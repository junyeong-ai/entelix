//! End-to-end coverage for `entelix_tools::memory::install` —
//! constructs a fake `SemanticMemoryBackend`, registers the
//! generated tools into a `ToolRegistry`, and dispatches one of
//! them to verify the round-trip.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::iter_overeager_cloned
)]

use std::sync::Arc;

use async_trait::async_trait;
use entelix_core::{AuditSink, AuditSinkHandle, ExecutionContext, Result, TenantId, ToolRegistry};
use entelix_memory::{
    Document, EntityMemory, IdentityReranker, InMemoryStore, Namespace, RerankedDocument, Reranker,
    SemanticMemoryBackend, VectorFilter,
};
use entelix_tools::memory::{MemoryToolConfig, install};
use parking_lot::Mutex;
use serde_json::json;

/// Fake backend that records every `add` and returns the recorded
/// documents on `search`.
struct FakeBackend {
    docs: Mutex<Vec<Document>>,
    namespace: Namespace,
}

impl FakeBackend {
    fn new() -> Self {
        Self {
            docs: Mutex::new(Vec::new()),
            namespace: Namespace::new(TenantId::new("test-tenant")).with_scope("fake"),
        }
    }
}

#[async_trait]
impl SemanticMemoryBackend for FakeBackend {
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
        let stored = self.docs.lock();
        Ok(stored.iter().cloned().take(top_k).collect())
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

    async fn add(&self, _ctx: &ExecutionContext, document: Document) -> Result<()> {
        self.docs.lock().push(document);
        Ok(())
    }

    async fn batch_add(&self, _ctx: &ExecutionContext, documents: Vec<Document>) -> Result<()> {
        self.docs.lock().extend(documents);
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
        ctx: &ExecutionContext,
        query: &str,
        top_k: usize,
        candidates: usize,
        reranker: &dyn Reranker,
    ) -> Result<Vec<RerankedDocument>> {
        let pool = self.search(ctx, query, candidates.max(top_k)).await?;
        reranker.rerank(query, pool, top_k, ctx).await
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
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Document>> {
        let stored = self.docs.lock();
        Ok(stored.iter().skip(offset).take(limit).cloned().collect())
    }
}

#[tokio::test]
async fn semantic_backend_search_with_rerank_dyn_passes_through_identity() {
    let backend: Arc<dyn SemanticMemoryBackend> = Arc::new(FakeBackend::new());
    let ctx = ExecutionContext::new();
    backend.add(&ctx, Document::new("doc-a")).await.unwrap();
    let results = backend
        .search_with_rerank_dyn(&ctx, "anything", 1, 10, &IdentityReranker)
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].document.content, "doc-a");
}

#[tokio::test]
async fn install_registers_query_and_save_tools_for_semantic_backend() {
    let backend: Arc<dyn SemanticMemoryBackend> = Arc::new(FakeBackend::new());
    let registry = install(
        ToolRegistry::new(),
        MemoryToolConfig::new().with_semantic(Arc::clone(&backend)),
    )
    .unwrap();

    assert!(registry.get("query_semantic_memory").is_some());
    assert!(registry.get("save_to_semantic_memory").is_some());
    assert!(registry.get("update_in_semantic_memory").is_some());
    assert!(registry.get("delete_from_semantic_memory").is_some());

    let ctx = ExecutionContext::new();

    // Save → search round-trip via the LLM-facing tools.
    let _ = registry
        .dispatch(
            "tu1",
            "save_to_semantic_memory",
            json!({"content": "the user prefers brief replies", "metadata": {}}),
            &ctx,
        )
        .await
        .unwrap();
    let result = registry
        .dispatch(
            "tu2",
            "query_semantic_memory",
            json!({"query": "preferences", "top_k": 5}),
            &ctx,
        )
        .await
        .unwrap();
    let results = result.get("results").and_then(|v| v.as_array()).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].get("content").and_then(|v| v.as_str()).unwrap(),
        "the user prefers brief replies"
    );
}

#[derive(Default)]
struct RecordingAuditSink {
    recalls: Mutex<Vec<(String, String, usize)>>,
}

impl AuditSink for RecordingAuditSink {
    fn record_sub_agent_invoked(&self, _agent_id: &str, _sub_thread_id: &str) {}
    fn record_agent_handoff(&self, _from: Option<&str>, _to: &str) {}
    fn record_resumed(&self, _from_checkpoint: &str) {}
    fn record_memory_recall(&self, tier: &str, namespace_key: &str, hits: usize) {
        self.recalls
            .lock()
            .push((tier.to_owned(), namespace_key.to_owned(), hits));
    }
}

#[tokio::test]
async fn query_semantic_memory_emits_memory_recall() {
    let backend: Arc<dyn SemanticMemoryBackend> = Arc::new(FakeBackend::new());
    let registry = install(
        ToolRegistry::new(),
        MemoryToolConfig::new().with_semantic(Arc::clone(&backend)),
    )
    .unwrap();
    let sink = Arc::new(RecordingAuditSink::default());
    let ctx = ExecutionContext::new()
        .with_audit_sink(AuditSinkHandle::new(Arc::clone(&sink) as Arc<dyn AuditSink>));

    let _ = registry
        .dispatch(
            "tu1",
            "save_to_semantic_memory",
            json!({"content": "hello", "metadata": {}}),
            &ctx,
        )
        .await
        .unwrap();
    let _ = registry
        .dispatch(
            "tu2",
            "query_semantic_memory",
            json!({"query": "anything", "top_k": 5}),
            &ctx,
        )
        .await
        .unwrap();

    let recalls: Vec<_> = sink.recalls.lock().clone();
    assert_eq!(recalls.len(), 1);
    let (tier, ns, hits) = &recalls[0];
    assert_eq!(tier, "semantic");
    assert_eq!(ns, "test-tenant:fake");
    assert_eq!(*hits, 1);
}

#[tokio::test]
async fn list_entity_facts_emits_memory_recall() {
    let store = Arc::new(InMemoryStore::new());
    let namespace = Namespace::new(TenantId::new("test-tenant")).with_scope("entities");
    let entity = Arc::new(EntityMemory::new(store, namespace));
    let registry = install(
        ToolRegistry::new(),
        MemoryToolConfig::new().with_entity(Arc::clone(&entity)),
    )
    .unwrap();
    let sink = Arc::new(RecordingAuditSink::default());
    let ctx = ExecutionContext::new()
        .with_audit_sink(AuditSinkHandle::new(Arc::clone(&sink) as Arc<dyn AuditSink>));

    let _ = registry
        .dispatch(
            "tu1",
            "set_entity_fact",
            json!({"entity": "alice", "fact": "loves rust"}),
            &ctx,
        )
        .await
        .unwrap();
    let _ = registry
        .dispatch("tu2", "list_entity_facts", json!({}), &ctx)
        .await
        .unwrap();

    let recalls: Vec<_> = sink.recalls.lock().clone();
    assert_eq!(recalls.len(), 1);
    let (tier, ns, hits) = &recalls[0];
    assert_eq!(tier, "entity");
    assert_eq!(ns, "test-tenant:entities");
    assert_eq!(*hits, 1);
}

#[tokio::test]
async fn get_entity_fact_emits_memory_recall_for_present_and_absent_keys() {
    // Invariant #18 — entity-tier point lookup is a recall act and
    // surfaces on the audit channel. `hits` discriminates miss (0)
    // from present (1) so operators can read the model's lookup
    // success rate without re-correlating with the response body.
    let store = Arc::new(InMemoryStore::new());
    let namespace = Namespace::new(TenantId::new("test-tenant")).with_scope("entities");
    let entity = Arc::new(EntityMemory::new(store, namespace));
    let registry = install(
        ToolRegistry::new(),
        MemoryToolConfig::new().with_entity(Arc::clone(&entity)),
    )
    .unwrap();
    let sink = Arc::new(RecordingAuditSink::default());
    let ctx = ExecutionContext::new()
        .with_audit_sink(AuditSinkHandle::new(Arc::clone(&sink) as Arc<dyn AuditSink>));

    // Seed one fact, then look it up.
    let _ = registry
        .dispatch(
            "tu1",
            "set_entity_fact",
            json!({"entity": "alice", "fact": "prefers rust"}),
            &ctx,
        )
        .await
        .unwrap();
    let _ = registry
        .dispatch("tu2", "get_entity_fact", json!({"entity": "alice"}), &ctx)
        .await
        .unwrap();
    // Look up an absent key.
    let _ = registry
        .dispatch("tu3", "get_entity_fact", json!({"entity": "ghost"}), &ctx)
        .await
        .unwrap();

    let recalls: Vec<_> = sink.recalls.lock().clone();
    assert_eq!(recalls.len(), 2, "one emit per get_entity_fact call");
    let (tier, ns, hits) = &recalls[0];
    assert_eq!(tier, "entity");
    assert_eq!(ns, "test-tenant:entities");
    assert_eq!(*hits, 1, "present key → hits=1");
    let (_, _, miss_hits) = &recalls[1];
    assert_eq!(*miss_hits, 0, "absent key → hits=0");
}

#[tokio::test]
async fn query_semantic_memory_without_sink_is_noop() {
    let backend: Arc<dyn SemanticMemoryBackend> = Arc::new(FakeBackend::new());
    let registry = install(
        ToolRegistry::new(),
        MemoryToolConfig::new().with_semantic(Arc::clone(&backend)),
    )
    .unwrap();
    let ctx = ExecutionContext::new();

    let result = registry
        .dispatch(
            "tu1",
            "query_semantic_memory",
            json!({"query": "anything"}),
            &ctx,
        )
        .await
        .unwrap();
    assert!(
        result
            .get("results")
            .and_then(|v| v.as_array())
            .unwrap()
            .is_empty()
    );
}
