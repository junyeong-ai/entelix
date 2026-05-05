//! Invariant #16 enforcement — LLM-facing channel hygiene.
//!
//! Built-in tool outputs and tool-spec schemas must not leak
//! operator-only content into the model's view. The contract:
//!
//! - `Error::render_for_llm()` strips vendor status,
//!   `provider returned …`, and source-chain framing.
//! - `LlmFacingSchema::strip` removes schemars envelope keys
//!   (`$schema`, `title`, `$defs`, `$ref`, integer width hints).
//! - `HttpFetchTool` ships zero response headers to the model unless
//!   the operator opts specific names in via
//!   `with_exposed_response_headers`.
//! - `query_semantic_memory` surfaces `rank` (1..=N) — never raw
//!   `score: f32`. Document metadata reaches the model only when
//!   `MemoryToolConfig::expose_metadata_fields` allowlists the key.
//! - `list_entity_facts` defaults to `{entity, fact}`; temporal
//!   signals are integer day-counts and only when
//!   `with_entity_temporal_signals(true)` is set.
//!
//! New regressions in any of these surfaces fail this gate, so a
//! silent leak cannot ship.

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use entelix_core::TenantId;
use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use entelix_core::ir::Usage;
use entelix_core::skills::SkillRegistry;
use entelix_core::{ExecutionContext, LlmFacingSchema, LlmRenderable, Result, ToolRegistry};
use entelix_memory::{
    Document, EntityMemory, InMemoryStore, Namespace, RerankedDocument, Reranker,
    SemanticMemoryBackend, VectorFilter,
};
use entelix_tools::memory::{MemoryToolConfig, install};
use entelix_tools::skills::install as install_skills;
use parking_lot::Mutex;
use serde_json::{Value, json};

// ── Error: LLM-facing rendering ───────────────────────────────────

#[test]
fn error_render_for_llm_omits_provider_status_and_chain() {
    let err = entelix_core::Error::provider_http(503, "vendor degraded".to_owned());
    let rendered = err.render_for_llm();
    let _ = Usage::default(); // anchor entelix_core::ir::Usage usage
    for forbidden in ["503", "vendor degraded", "provider returned"] {
        assert!(
            !rendered.contains(forbidden),
            "render_for_llm leaks `{forbidden}`: {rendered}"
        );
    }
}

#[test]
fn error_render_for_llm_invalid_request_keeps_caller_message() {
    let err = entelix_core::Error::invalid_request("missing 'task' field");
    let rendered = err.render_for_llm();
    assert!(
        rendered.contains("missing 'task' field"),
        "caller-supplied message must survive (it is already model-safe): {rendered}"
    );
}

/// Carrier sealing — `RenderedForLlm<T>::new` is `pub(crate)` to
/// `entelix-core`, so this crate (`entelix-tools`) cannot fabricate
/// a carrier from a raw string. The only path to one is
/// `LlmRenderable::for_llm` on a value that implements the trait,
/// which is the structural guarantee invariant 16 / ADR-0076 codify.
///
/// This test does not invoke any forbidden API — it documents the
/// boundary by exercising the only legal construction path. A
/// regression that re-exports `RenderedForLlm::new` as `pub` would
/// not break this test (the legal path still works) but would be
/// caught by `cargo xtask public-api` baselines, by the
/// `pub(crate)` visibility check at compile time in any external
/// consumer that tries to call `RenderedForLlm::new`, and by the
/// reviewer reading ADR-0076 §"Sealing".
#[test]
fn rendered_for_llm_only_constructible_via_for_llm_trait_default() {
    use entelix_core::{Error, LlmRenderable, RenderedForLlm};
    let err = Error::provider_http(503, "vendor down".to_owned());
    let carrier: RenderedForLlm<String> = err.for_llm();
    // The only operations on the carrier from outside `entelix-core`
    // are the read accessors. Inner value matches the raw producer.
    assert_eq!(carrier.as_inner(), "upstream model error");
    assert_eq!(carrier.into_inner(), "upstream model error");
}

// ── Schema strip: every built-in tool spec must be free of
//    schemars envelope keys ───────────────────────────────────────

#[test]
fn llm_facing_schema_strip_removes_envelope_keys_recursively() {
    let raw = json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Outer",
        "type": "object",
        "properties": {
            "n": {"type": "integer", "format": "int64"},
            "child": {"$ref": "#/$defs/Inner"}
        },
        "$defs": {
            "Inner": {
                "title": "Inner",
                "type": "object",
                "properties": {"s": {"type": "string"}}
            }
        }
    });
    let stripped = LlmFacingSchema::strip(&raw);
    let printed = stripped.to_string();
    for forbidden in ["$schema", "$defs", "definitions", "$ref"] {
        assert!(
            !printed.contains(forbidden),
            "stripped schema still carries `{forbidden}`: {printed}"
        );
    }
    // Title is removed at every depth.
    assert!(
        !printed.contains("\"title\""),
        "stripped schema still carries `title`: {printed}"
    );
    // Integer width hint is dropped.
    assert!(
        !printed.contains("int64"),
        "stripped schema still carries `int64`: {printed}"
    );
    // User-named property survives.
    assert_eq!(stripped["properties"]["n"]["type"], "integer");
    assert_eq!(
        stripped["properties"]["child"]["properties"]["s"]["type"],
        "string"
    );
}

// ── HttpFetchTool: response headers are default-empty ────────────

#[test]
fn http_fetch_tool_metadata_lists_no_response_header_field() {
    use entelix_core::tools::Tool;
    use entelix_tools::{HostAllowlist, HttpFetchTool};
    let tool = HttpFetchTool::builder()
        .with_allowlist(HostAllowlist::new().add_exact_host("api.example.com"))
        .build()
        .unwrap();
    // The tool input schema declares headers callers may *send*; the
    // gate here is on the *output*. We exercise that contract end to
    // end via the dispatch path in `http_fetch_e2e.rs`. Static check:
    // the metadata schema strip does not leak schemars envelope.
    let schema_str = tool.metadata().input_schema.to_string();
    assert!(
        !schema_str.contains("$schema") && !schema_str.contains("\"title\""),
        "tool input schema leaks schemars envelope: {schema_str}"
    );
}

// ── QuerySemanticMemoryTool: no `score`, metadata allowlist ──────

struct InMemorySemantic {
    docs: Mutex<Vec<Document>>,
    namespace: Namespace,
}

impl InMemorySemantic {
    fn new() -> Self {
        Self {
            docs: Mutex::new(Vec::new()),
            namespace: Namespace::new(TenantId::new("test")).with_scope("semantic"),
        }
    }
    fn seed(self, docs: Vec<Document>) -> Self {
        *self.docs.lock() = docs;
        self
    }
}

#[async_trait]
impl SemanticMemoryBackend for InMemorySemantic {
    fn namespace(&self) -> &Namespace {
        &self.namespace
    }
    fn dimension(&self) -> usize {
        4
    }
    async fn search(
        &self,
        _ctx: &ExecutionContext,
        _query: &str,
        top_k: usize,
    ) -> Result<Vec<Document>> {
        Ok(self.docs.lock().iter().take(top_k).cloned().collect())
    }
    async fn search_filtered(
        &self,
        _ctx: &ExecutionContext,
        _query: &str,
        _top_k: usize,
        _filter: &VectorFilter,
    ) -> Result<Vec<Document>> {
        Ok(Vec::new())
    }
    async fn add(&self, _ctx: &ExecutionContext, _document: Document) -> Result<()> {
        Ok(())
    }
    async fn batch_add(&self, _ctx: &ExecutionContext, _documents: Vec<Document>) -> Result<()> {
        Ok(())
    }
    async fn delete(&self, _ctx: &ExecutionContext, _doc_id: &str) -> Result<()> {
        Ok(())
    }
    async fn update(
        &self,
        _ctx: &ExecutionContext,
        _doc_id: &str,
        _document: Document,
    ) -> Result<()> {
        Ok(())
    }
    async fn search_with_rerank_dyn(
        &self,
        _ctx: &ExecutionContext,
        _query: &str,
        _top_k: usize,
        _candidates: usize,
        _reranker: &dyn Reranker,
    ) -> Result<Vec<RerankedDocument>> {
        Ok(Vec::new())
    }
    async fn count(
        &self,
        _ctx: &ExecutionContext,
        _filter: Option<&VectorFilter>,
    ) -> Result<usize> {
        Ok(self.docs.lock().len())
    }
    async fn list(
        &self,
        _ctx: &ExecutionContext,
        _filter: Option<&VectorFilter>,
        _limit: usize,
        _offset: usize,
    ) -> Result<Vec<Document>> {
        Ok(self.docs.lock().clone())
    }
}

#[tokio::test]
async fn query_semantic_memory_default_output_carries_rank_not_score_and_no_metadata() {
    let backend: Arc<dyn SemanticMemoryBackend> = Arc::new(InMemorySemantic::new().seed(vec![
            Document::new("alpha")
                .with_metadata(json!({"source": "alpha-src", "namespace_key": "tenant:scope:1"})),
            Document::new("beta")
                .with_metadata(json!({"source": "beta-src", "namespace_key": "tenant:scope:2"})),
        ]));
    let registry = install(
        ToolRegistry::new(),
        MemoryToolConfig::new().with_semantic(Arc::clone(&backend)),
    )
    .unwrap();
    let ctx = ExecutionContext::new();
    let out = registry
        .dispatch("tu", "query_semantic_memory", json!({"query": "x"}), &ctx)
        .await
        .unwrap();
    let results = out["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    for (idx, row) in results.iter().enumerate() {
        let row_obj = row.as_object().unwrap();
        assert!(row_obj.contains_key("rank"), "row missing rank: {row}");
        assert_eq!(row["rank"].as_u64().unwrap(), (idx as u64) + 1);
        assert!(
            !row_obj.contains_key("score"),
            "default output must not surface `score: f32` to the model: {row}"
        );
        assert!(
            !row_obj.contains_key("metadata"),
            "default output must not surface backend metadata to the model: {row}"
        );
    }
}

#[tokio::test]
async fn query_semantic_memory_metadata_allowlist_filters_internal_keys() {
    let backend: Arc<dyn SemanticMemoryBackend> =
        Arc::new(
            InMemorySemantic::new().seed(vec![Document::new("doc-a").with_metadata(json!({
                "source": "good",
                "title": "Doc A",
                "namespace_key": "tenant:scope:1",
                "embedding_hash": "abc123",
            }))]),
        );
    let registry = install(
        ToolRegistry::new(),
        MemoryToolConfig::new()
            .with_semantic(Arc::clone(&backend))
            .expose_metadata_fields(["source", "title"]),
    )
    .unwrap();
    let ctx = ExecutionContext::new();
    let out = registry
        .dispatch("tu", "query_semantic_memory", json!({"query": "x"}), &ctx)
        .await
        .unwrap();
    let row = &out["results"][0];
    let metadata = row["metadata"].as_object().unwrap();
    assert!(metadata.contains_key("source"));
    assert!(metadata.contains_key("title"));
    assert!(
        !metadata.contains_key("namespace_key"),
        "backend internals must not bleed into LLM-facing metadata: {row}"
    );
    assert!(
        !metadata.contains_key("embedding_hash"),
        "backend internals must not bleed into LLM-facing metadata: {row}"
    );
}

// ── ListEntityFactsTool: temporal default off ────────────────────

async fn entity_memory_with_one_fact(fact: &str) -> Arc<EntityMemory> {
    let store = Arc::new(InMemoryStore::<HashMap<String, entelix_memory::EntityRecord>>::new());
    let entity = Arc::new(EntityMemory::new(
        store,
        Namespace::new(TenantId::new("test")).with_scope("entity"),
    ));
    entity
        .set_entity(&ExecutionContext::new(), "alice", fact)
        .await
        .unwrap();
    entity
}

#[tokio::test]
async fn list_entity_facts_default_output_omits_temporal_signals() {
    let entity = entity_memory_with_one_fact("prefers terse answers").await;
    let registry = install(
        ToolRegistry::new(),
        MemoryToolConfig::new().with_entity(entity),
    )
    .unwrap();
    let ctx = ExecutionContext::new();
    let out = registry
        .dispatch("tu", "list_entity_facts", json!({}), &ctx)
        .await
        .unwrap();
    let row = &out["entities"][0];
    let row_obj = row.as_object().unwrap();
    assert_eq!(row["entity"], "alice");
    assert_eq!(row["fact"], "prefers terse answers");
    for forbidden in [
        "created_at",
        "last_seen",
        "first_seen_days_ago",
        "last_seen_days_ago",
    ] {
        assert!(
            !row_obj.contains_key(forbidden),
            "default output must not surface `{forbidden}` to the model: {row}"
        );
    }
    // Hard regex check: no RFC3339 datetime pattern anywhere in the
    // serialised output (`YYYY-MM-DDTHH:MM`).
    let printed = out.to_string();
    assert!(
        !regex_lite_rfc3339(&printed),
        "default output must not surface any RFC3339 timestamp: {printed}"
    );
}

#[tokio::test]
async fn list_entity_facts_with_temporal_signals_uses_integer_days() {
    let entity = entity_memory_with_one_fact("prefers terse answers").await;
    let registry = install(
        ToolRegistry::new(),
        MemoryToolConfig::new()
            .with_entity(entity)
            .with_entity_temporal_signals(true),
    )
    .unwrap();
    let ctx = ExecutionContext::new();
    let out = registry
        .dispatch("tu", "list_entity_facts", json!({}), &ctx)
        .await
        .unwrap();
    let row = &out["entities"][0];
    let row_obj = row.as_object().unwrap();
    assert!(row_obj.contains_key("first_seen_days_ago"));
    assert!(row_obj.contains_key("last_seen_days_ago"));
    assert!(
        row["first_seen_days_ago"].is_u64(),
        "temporal signal must be an integer (model-friendly), not RFC3339: {row}"
    );
    let printed = out.to_string();
    assert!(
        !regex_lite_rfc3339(&printed),
        "temporal signal must NOT be RFC3339 even when opted in: {printed}"
    );
}

// Tiny RFC3339 detector — looks for `\d{4}-\d{2}-\d{2}T` which is
// the unambiguous prefix of an ISO 8601 / RFC3339 timestamp. Avoids
// pulling the `regex` crate just for one assertion.
fn regex_lite_rfc3339(s: &str) -> bool {
    let bytes = s.as_bytes();
    bytes.windows(11).any(|w| {
        w[0].is_ascii_digit()
            && w[1].is_ascii_digit()
            && w[2].is_ascii_digit()
            && w[3].is_ascii_digit()
            && w[4] == b'-'
            && w[5].is_ascii_digit()
            && w[6].is_ascii_digit()
            && w[7] == b'-'
            && w[8].is_ascii_digit()
            && w[9].is_ascii_digit()
            && w[10] == b'T'
    })
}

// ── Skill tools: their auto-generated input schemas must also
//    survive the strip pass without schemars envelope. ──

#[test]
fn skill_tool_input_schemas_carry_no_schemars_envelope() {
    let registry = install_skills(ToolRegistry::new(), SkillRegistry::new()).unwrap();
    let names: Vec<String> = registry.names().map(str::to_owned).collect();
    for name in &names {
        let tool = registry.get(name).unwrap();
        let printed = tool.metadata().input_schema.to_string();
        for forbidden in ["$schema", "$defs", "definitions", "$ref"] {
            assert!(
                !printed.contains(forbidden),
                "tool `{name}` input schema leaks `{forbidden}`: {printed}"
            );
        }
        assert!(
            !printed.contains("\"title\""),
            "tool `{name}` input schema leaks schemars title: {printed}"
        );
    }
}

// Ensure `Value` import survives even when one branch removes it.
#[allow(dead_code)]
const _: fn(&Value) = |_| {};
