//! LLM-facing memory tools — let the model query, mutate, and
//! prune the agent's long-term memory by emitting tool calls.
//!
//! Mirrors the Skills auto-wire pattern: operators construct memory
//! handles (a [`SemanticMemoryBackend`], an [`EntityMemory`], …) and
//! call [`install`] to register the corresponding tools into a
//! [`ToolRegistry`]. Tools the model can issue:
//!
//! Semantic-memory CRUD:
//! - `query_semantic_memory(query, top_k)` — vector search
//! - `save_to_semantic_memory(content, metadata)` — index a new document
//! - `update_in_semantic_memory(doc_id, content, metadata)` — replace an existing document
//! - `delete_from_semantic_memory(doc_id)` — drop a document by id
//!
//! Entity-memory CRUD:
//! - `set_entity_fact(entity, fact)` — write a fact (refreshes `last_seen`)
//! - `get_entity_fact(entity)` — read a fact
//! - `list_entity_facts()` — enumerate every recorded entity
//! - `clear_entity_fact(entity)` — drop a single entity
//!
//! The helper is **opt-in**: agents that should not expose memory
//! tools to the model simply skip the call. Mirroring
//! `skills::install`, the registry is consumed and a new one
//! returned with the tools appended.

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use entelix_core::tools::{Tool, ToolEffect, ToolMetadata};
use entelix_core::{AgentContext, Result, ToolRegistry};
use entelix_memory::{Document, EntityMemory, SemanticMemoryBackend};
use serde_json::{Value, json};

/// Bundle of memory handles to expose as LLM-facing tools.
///
/// Each field is optional — operators only register tools the agent
/// is supposed to use. Constructing every field is rare; typical
/// agents expose either entity OR semantic memory, not both.
///
/// ## LLM-facing exposure (invariant #16)
///
/// Memory tools expose their results to the model. Two opt-in knobs
/// keep noise out of the model's context by default:
///
/// - **Document metadata** is dropped from `query_semantic_memory`
///   results unless [`Self::expose_metadata_fields`] names specific
///   keys to surface. Backend internals (`namespace_key`,
///   `embedding_hash`, internal timestamps) never reach the model.
/// - **Entity temporal signals** (`created_at`, `last_seen`) are
///   omitted from `list_entity_facts` unless
///   [`Self::with_entity_temporal_signals`] is called.
pub struct MemoryToolConfig {
    /// Optional handle to a [`entelix_memory::SemanticMemory`].
    /// Wrapped behind a trait object so the tool is not generic
    /// over embedder/vector-store types.
    pub semantic: Option<Arc<dyn SemanticMemoryBackend>>,
    /// Optional handle to an [`EntityMemory`].
    pub entity: Option<Arc<EntityMemory>>,
    /// LLM-facing metadata field allowlist for `query_semantic_memory`.
    /// Empty (default) → no metadata reaches the model.
    metadata_allowlist: HashSet<String>,
    /// Whether `list_entity_facts` includes temporal signals
    /// (`first_seen_days_ago`, `last_seen_days_ago`). Default false
    /// — most agents only need the fact body.
    expose_entity_temporal: bool,
}

impl MemoryToolConfig {
    /// Empty bundle. Use builder-style methods to populate.
    #[must_use]
    pub fn new() -> Self {
        Self {
            semantic: None,
            entity: None,
            metadata_allowlist: HashSet::new(),
            expose_entity_temporal: false,
        }
    }

    /// Attach a semantic memory handle.
    #[must_use]
    pub fn with_semantic(mut self, handle: Arc<dyn SemanticMemoryBackend>) -> Self {
        self.semantic = Some(handle);
        self
    }

    /// Attach an entity-memory handle.
    #[must_use]
    pub fn with_entity(mut self, entity: Arc<EntityMemory>) -> Self {
        self.entity = Some(entity);
        self
    }

    /// Allow specific document-metadata keys to surface to the model
    /// in `query_semantic_memory` results. Default is the empty set
    /// — backend internals (`namespace_key`, `embedding_hash`,
    /// vendor-side timestamps) never reach the model. Operators
    /// opt-in fields the model can productively reason about
    /// (`source`, `title`, `kind`).
    #[must_use]
    pub fn expose_metadata_fields<I, S>(mut self, fields: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.metadata_allowlist = fields.into_iter().map(Into::into).collect();
        self
    }

    /// Include `first_seen_days_ago` / `last_seen_days_ago` integer
    /// fields in `list_entity_facts` output. Default off — RFC3339
    /// timestamps are 32-byte tokens for each entity that the model
    /// usually cannot act on (invariant #16). Opt in when an agent
    /// genuinely reasons about recency.
    #[must_use]
    pub const fn with_entity_temporal_signals(mut self, expose: bool) -> Self {
        self.expose_entity_temporal = expose;
        self
    }
}

impl Default for MemoryToolConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Append memory tools backed by `config` to `registry` and return
/// the new registry. Skipped silently when `config` is empty.
pub fn install(mut registry: ToolRegistry, config: MemoryToolConfig) -> Result<ToolRegistry> {
    let metadata_allowlist = Arc::new(config.metadata_allowlist);
    let expose_entity_temporal = config.expose_entity_temporal;
    if let Some(semantic) = config.semantic {
        registry = registry.register(Arc::new(QuerySemanticMemoryTool::new(
            Arc::clone(&semantic),
            Arc::clone(&metadata_allowlist),
        )))?;
        registry = registry.register(Arc::new(SaveToSemanticMemoryTool::new(Arc::clone(
            &semantic,
        ))))?;
        registry = registry.register(Arc::new(UpdateInSemanticMemoryTool::new(Arc::clone(
            &semantic,
        ))))?;
        registry = registry.register(Arc::new(DeleteFromSemanticMemoryTool::new(semantic)))?;
    }
    if let Some(entity) = config.entity {
        registry = registry.register(Arc::new(SetEntityFactTool::new(Arc::clone(&entity))))?;
        registry = registry.register(Arc::new(GetEntityFactTool::new(Arc::clone(&entity))))?;
        registry = registry.register(Arc::new(ListEntityFactsTool::new(
            Arc::clone(&entity),
            expose_entity_temporal,
        )))?;
        registry = registry.register(Arc::new(ClearEntityFactTool::new(entity)))?;
    }
    Ok(registry)
}

// ── semantic ─────────────────────────────────────────────────────────────

/// `query_semantic_memory(query, top_k)` — search the agent's
/// vector store for documents similar to `query`.
struct QuerySemanticMemoryTool {
    handle: Arc<dyn SemanticMemoryBackend>,
    metadata_allowlist: Arc<HashSet<String>>,
    metadata: ToolMetadata,
}

impl QuerySemanticMemoryTool {
    fn new(
        handle: Arc<dyn SemanticMemoryBackend>,
        metadata_allowlist: Arc<HashSet<String>>,
    ) -> Self {
        Self {
            handle,
            metadata_allowlist,
            metadata: ToolMetadata::function(
                "query_semantic_memory",
                "Search the agent's long-term semantic memory for documents \
                 similar to the supplied query. Results are ranked best-first.",
                json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
                    },
                    "required": ["query"],
                    "additionalProperties": false,
                }),
            )
            .with_effect(ToolEffect::ReadOnly)
            .with_idempotent(true),
        }
    }
}

#[async_trait]
impl Tool for QuerySemanticMemoryTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    async fn execute(&self, input: Value, ctx: &AgentContext<()>) -> Result<Value> {
        let query = input
            .get("query")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let top_k =
            usize::try_from(input.get("top_k").and_then(Value::as_u64).unwrap_or(5)).unwrap_or(5);
        let docs = self.handle.search(ctx.core(), query, top_k).await?;
        if let Some(handle) = ctx.audit_sink() {
            handle.as_sink().record_memory_recall(
                "semantic",
                &self.handle.namespace().render(),
                docs.len(),
            );
        }
        // Invariant #16 — raw `score: f32` (cosine distance) is
        // meaningless to the model. Surface a 1-based `rank` instead.
        // Document metadata is dropped to backend-internals (`namespace_key`,
        // `embedding_hash`, vendor timestamps) by default; only fields the
        // operator opted in via `MemoryToolConfig::expose_metadata_fields`
        // flow through.
        let allowlist = &*self.metadata_allowlist;
        let results: Vec<Value> = docs
            .iter()
            .enumerate()
            .map(|(idx, d)| {
                let mut row = serde_json::Map::new();
                row.insert("rank".into(), json!(idx + 1));
                row.insert("content".into(), Value::String(d.content.clone()));
                if !allowlist.is_empty()
                    && let Some(meta_obj) = d.metadata.as_object()
                {
                    let filtered: serde_json::Map<String, Value> = meta_obj
                        .iter()
                        .filter(|(k, _)| allowlist.contains(k.as_str()))
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();
                    if !filtered.is_empty() {
                        row.insert("metadata".into(), Value::Object(filtered));
                    }
                }
                Value::Object(row)
            })
            .collect();
        Ok(json!({ "results": results }))
    }
}

/// `save_to_semantic_memory(content, metadata)` — index a new
/// document into the agent's long-term semantic memory.
struct SaveToSemanticMemoryTool {
    handle: Arc<dyn SemanticMemoryBackend>,
    metadata: ToolMetadata,
}

impl SaveToSemanticMemoryTool {
    fn new(handle: Arc<dyn SemanticMemoryBackend>) -> Self {
        Self {
            handle,
            metadata: ToolMetadata::function(
                "save_to_semantic_memory",
                "Add a new document to the agent's long-term semantic memory. \
                 Use when the conversation surfaces a fact worth remembering for \
                 future turns.",
                json!({
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "metadata": {"type": "object"},
                    },
                    "required": ["content"],
                    "additionalProperties": false,
                }),
            )
            .with_effect(ToolEffect::Mutating),
        }
    }
}

#[async_trait]
impl Tool for SaveToSemanticMemoryTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    async fn execute(&self, input: Value, ctx: &AgentContext<()>) -> Result<Value> {
        let content = input
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let metadata = input.get("metadata").cloned().unwrap_or(Value::Null);
        let doc = Document::new(content).with_metadata(metadata);
        self.handle.add(ctx.core(), doc).await?;
        Ok(json!({"saved": true}))
    }
}

/// `update_in_semantic_memory(doc_id, content, metadata)` — replace
/// an existing document's content and metadata. Returns
/// `Error::Config` from the backend when the id is unknown.
struct UpdateInSemanticMemoryTool {
    handle: Arc<dyn SemanticMemoryBackend>,
    metadata: ToolMetadata,
}

impl UpdateInSemanticMemoryTool {
    fn new(handle: Arc<dyn SemanticMemoryBackend>) -> Self {
        Self {
            handle,
            metadata: ToolMetadata::function(
                "update_in_semantic_memory",
                "Replace an existing document in the agent's semantic memory \
                 (looked up by `doc_id`). Use to correct a fact you previously \
                 saved or to refresh stale metadata.",
                json!({
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string"},
                        "content": {"type": "string"},
                        "metadata": {"type": "object"},
                    },
                    "required": ["doc_id", "content"],
                    "additionalProperties": false,
                }),
            )
            .with_effect(ToolEffect::Mutating),
        }
    }
}

#[async_trait]
impl Tool for UpdateInSemanticMemoryTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    async fn execute(&self, input: Value, ctx: &AgentContext<()>) -> Result<Value> {
        let doc_id = input
            .get("doc_id")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let content = input
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let metadata = input.get("metadata").cloned().unwrap_or(Value::Null);
        let doc = Document::new(content).with_metadata(metadata);
        self.handle.update(ctx.core(), doc_id, doc).await?;
        Ok(json!({"updated": true, "doc_id": doc_id}))
    }
}

/// `delete_from_semantic_memory(doc_id)` — remove a document by id.
/// Idempotent at the backend's discretion.
struct DeleteFromSemanticMemoryTool {
    handle: Arc<dyn SemanticMemoryBackend>,
    metadata: ToolMetadata,
}

impl DeleteFromSemanticMemoryTool {
    fn new(handle: Arc<dyn SemanticMemoryBackend>) -> Self {
        Self {
            handle,
            metadata: ToolMetadata::function(
                "delete_from_semantic_memory",
                "Drop a document from the agent's semantic memory by its \
                 `doc_id` (the id surfaced in `query_semantic_memory` results). \
                 Use when a previously-saved fact is wrong or no longer relevant.",
                json!({
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string"},
                    },
                    "required": ["doc_id"],
                    "additionalProperties": false,
                }),
            )
            .with_effect(ToolEffect::Destructive),
        }
    }
}

#[async_trait]
impl Tool for DeleteFromSemanticMemoryTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    async fn execute(&self, input: Value, ctx: &AgentContext<()>) -> Result<Value> {
        let doc_id = input
            .get("doc_id")
            .and_then(Value::as_str)
            .unwrap_or_default();
        self.handle.delete(ctx.core(), doc_id).await?;
        Ok(json!({"deleted": true, "doc_id": doc_id}))
    }
}

// ── entity ───────────────────────────────────────────────────────────────

/// `set_entity_fact(entity, fact)` — write to the agent's entity
/// memory.
struct SetEntityFactTool {
    entity: Arc<EntityMemory>,
    metadata: ToolMetadata,
}

impl SetEntityFactTool {
    fn new(entity: Arc<EntityMemory>) -> Self {
        Self {
            entity,
            metadata: ToolMetadata::function(
                "set_entity_fact",
                "Record a fact about a named entity (person, place, project) \
                 so future turns can recall it. Existing facts for the same \
                 entity are overwritten.",
                json!({
                    "type": "object",
                    "properties": {
                        "entity": {"type": "string"},
                        "fact": {"type": "string"},
                    },
                    "required": ["entity", "fact"],
                    "additionalProperties": false,
                }),
            )
            .with_effect(ToolEffect::Mutating),
        }
    }
}

#[async_trait]
impl Tool for SetEntityFactTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    async fn execute(&self, input: Value, ctx: &AgentContext<()>) -> Result<Value> {
        let entity = input
            .get("entity")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let fact = input
            .get("fact")
            .and_then(Value::as_str)
            .unwrap_or_default();
        self.entity.set_entity(ctx.core(), entity, fact).await?;
        Ok(json!({"saved": true, "entity": entity}))
    }
}

/// `get_entity_fact(entity)` — read from the agent's entity memory.
struct GetEntityFactTool {
    entity: Arc<EntityMemory>,
    metadata: ToolMetadata,
}

impl GetEntityFactTool {
    fn new(entity: Arc<EntityMemory>) -> Self {
        Self {
            entity,
            metadata: ToolMetadata::function(
                "get_entity_fact",
                "Look up the recorded fact for a named entity. Returns null \
                 when no fact exists for that entity.",
                json!({
                    "type": "object",
                    "properties": {
                        "entity": {"type": "string"},
                    },
                    "required": ["entity"],
                    "additionalProperties": false,
                }),
            )
            .with_effect(ToolEffect::ReadOnly)
            .with_idempotent(true),
        }
    }
}

#[async_trait]
impl Tool for GetEntityFactTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    async fn execute(&self, input: Value, ctx: &AgentContext<()>) -> Result<Value> {
        let entity = input
            .get("entity")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let fact = self.entity.entity(ctx.core(), entity).await?;
        // Invariant #18 — entity-tier point lookup is a recall act
        // and must surface on the audit channel alongside the
        // semantic / list recalls. `hits` is 0 for a miss, 1 for a
        // present fact (the model saw exactly that many records).
        if let Some(handle) = ctx.audit_sink() {
            handle.as_sink().record_memory_recall(
                "entity",
                &self.entity.namespace().render(),
                usize::from(fact.is_some()),
            );
        }
        Ok(json!({"entity": entity, "fact": fact}))
    }
}

/// `list_entity_facts()` — enumerate every recorded entity in the
/// current namespace. Useful for the model to audit what it has
/// learned before deciding whether to overwrite or extend.
struct ListEntityFactsTool {
    entity: Arc<EntityMemory>,
    expose_temporal: bool,
    metadata: ToolMetadata,
}

impl ListEntityFactsTool {
    fn new(entity: Arc<EntityMemory>, expose_temporal: bool) -> Self {
        let description = if expose_temporal {
            "List every entity the agent has recorded a fact about, \
             including the fact and integer day-counts since it was \
             first observed and last confirmed."
        } else {
            "List every entity the agent has recorded a fact about, \
             including the fact body."
        };
        Self {
            entity,
            expose_temporal,
            metadata: ToolMetadata::function(
                "list_entity_facts",
                description,
                json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false,
                }),
            )
            .with_effect(ToolEffect::ReadOnly)
            .with_idempotent(true),
        }
    }
}

#[async_trait]
impl Tool for ListEntityFactsTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    async fn execute(&self, _input: Value, ctx: &AgentContext<()>) -> Result<Value> {
        let records = self.entity.all_records(ctx.core()).await?;
        if let Some(handle) = ctx.audit_sink() {
            handle.as_sink().record_memory_recall(
                "entity",
                &self.entity.namespace().render(),
                records.len(),
            );
        }
        let now = chrono::Utc::now();
        let entries: Vec<Value> = records
            .into_iter()
            .map(|(name, record)| {
                let mut row = serde_json::Map::new();
                row.insert("entity".into(), Value::String(name));
                row.insert("fact".into(), Value::String(record.fact));
                if self.expose_temporal {
                    // Model-friendly integer day counts beat RFC3339
                    // strings — fewer tokens, easier to reason about
                    // ("written 3 days ago" vs `2026-04-28T17:24:09Z`).
                    let first = (now - record.created_at).num_days().max(0);
                    let last = (now - record.last_seen).num_days().max(0);
                    row.insert("first_seen_days_ago".into(), json!(first));
                    row.insert("last_seen_days_ago".into(), json!(last));
                }
                Value::Object(row)
            })
            .collect();
        Ok(json!({"entities": entries}))
    }
}

/// `clear_entity_fact(entity)` — drop a single entity from the
/// agent's entity memory. Idempotent: clearing an unknown entity
/// is a no-op rather than an error.
struct ClearEntityFactTool {
    entity: Arc<EntityMemory>,
    metadata: ToolMetadata,
}

impl ClearEntityFactTool {
    fn new(entity: Arc<EntityMemory>) -> Self {
        Self {
            entity,
            metadata: ToolMetadata::function(
                "clear_entity_fact",
                "Drop the recorded fact for a single entity. Use when a fact \
                 is wrong or has become outdated. Clearing an entity that was \
                 never recorded is a no-op.",
                json!({
                    "type": "object",
                    "properties": {
                        "entity": {"type": "string"},
                    },
                    "required": ["entity"],
                    "additionalProperties": false,
                }),
            )
            .with_effect(ToolEffect::Destructive),
        }
    }
}

#[async_trait]
impl Tool for ClearEntityFactTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    async fn execute(&self, input: Value, ctx: &AgentContext<()>) -> Result<Value> {
        let entity = input
            .get("entity")
            .and_then(Value::as_str)
            .unwrap_or_default();
        self.entity.remove(ctx.core(), entity).await?;
        Ok(json!({"cleared": true, "entity": entity}))
    }
}
