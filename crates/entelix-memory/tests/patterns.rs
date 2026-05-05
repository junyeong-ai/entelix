//! `SummaryMemory` + `EntityMemory` + `SemanticMemory` tests.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::missing_const_for_fn
)]

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use entelix_core::{ExecutionContext, Result};
use entelix_memory::{
    Document, Embedder, Embedding, EmbeddingUsage, EntityMemory, EntityRecord, InMemoryStore,
    Namespace, SemanticMemory, Store, SummaryMemory, VectorStore,
};
use parking_lot::Mutex;

// ── SummaryMemory ──────────────────────────────────────────────────────────

#[tokio::test]
async fn summary_set_and_get_round_trip() -> Result<()> {
    let store: Arc<dyn Store<String>> = Arc::new(InMemoryStore::<String>::new());
    let mem = SummaryMemory::new(store, Namespace::new(TenantId::new("t")));
    let ctx = ExecutionContext::new();
    assert!(mem.get(&ctx).await?.is_none());
    mem.set(&ctx, "user prefers brief replies").await?;
    assert_eq!(
        mem.get(&ctx).await?.as_deref(),
        Some("user prefers brief replies")
    );
    Ok(())
}

#[tokio::test]
async fn summary_append_concatenates_with_blank_line() -> Result<()> {
    let store: Arc<dyn Store<String>> = Arc::new(InMemoryStore::<String>::new());
    let mem = SummaryMemory::new(store, Namespace::new(TenantId::new("t")));
    let ctx = ExecutionContext::new();
    mem.append(&ctx, "turn 1: hello").await?;
    mem.append(&ctx, "turn 2: hi back").await?;
    let summary = mem.get(&ctx).await?.unwrap();
    assert_eq!(summary, "turn 1: hello\n\nturn 2: hi back");
    Ok(())
}

#[tokio::test]
async fn summary_append_to_empty_starts_fresh() -> Result<()> {
    let store: Arc<dyn Store<String>> = Arc::new(InMemoryStore::<String>::new());
    let mem = SummaryMemory::new(store, Namespace::new(TenantId::new("t")));
    let ctx = ExecutionContext::new();
    mem.append(&ctx, "first content").await?;
    assert_eq!(mem.get(&ctx).await?.as_deref(), Some("first content"));
    Ok(())
}

#[tokio::test]
async fn summary_clear_removes_value() -> Result<()> {
    let store: Arc<dyn Store<String>> = Arc::new(InMemoryStore::<String>::new());
    let mem = SummaryMemory::new(store, Namespace::new(TenantId::new("t")));
    let ctx = ExecutionContext::new();
    mem.set(&ctx, "anything").await?;
    mem.clear(&ctx).await?;
    assert!(mem.get(&ctx).await?.is_none());
    Ok(())
}

// ── EntityMemory ───────────────────────────────────────────────────────────

#[tokio::test]
async fn entity_set_and_get() -> Result<()> {
    let store: Arc<dyn Store<HashMap<String, EntityRecord>>> =
        Arc::new(InMemoryStore::<HashMap<String, EntityRecord>>::new());
    let mem = EntityMemory::new(store, Namespace::new(TenantId::new("t")));
    let ctx = ExecutionContext::new();
    mem.set_entity(&ctx, "Alice", "vegetarian").await?;
    mem.set_entity(&ctx, "Bob", "prefers Korean").await?;
    assert_eq!(
        mem.entity(&ctx, "Alice").await?.as_deref(),
        Some("vegetarian")
    );
    assert_eq!(
        mem.entity(&ctx, "Bob").await?.as_deref(),
        Some("prefers Korean")
    );
    assert!(mem.entity(&ctx, "Carol").await?.is_none());
    Ok(())
}

#[tokio::test]
async fn entity_all_returns_full_map() -> Result<()> {
    let store: Arc<dyn Store<HashMap<String, EntityRecord>>> =
        Arc::new(InMemoryStore::<HashMap<String, EntityRecord>>::new());
    let mem = EntityMemory::new(store, Namespace::new(TenantId::new("t")));
    let ctx = ExecutionContext::new();
    mem.set_entity(&ctx, "Alice", "v").await?;
    mem.set_entity(&ctx, "Bob", "k").await?;
    let all = mem.all(&ctx).await?;
    assert_eq!(all.len(), 2);
    assert_eq!(all.get("Alice").map(String::as_str), Some("v"));
    Ok(())
}

#[tokio::test]
async fn entity_remove_drops_one() -> Result<()> {
    let store: Arc<dyn Store<HashMap<String, EntityRecord>>> =
        Arc::new(InMemoryStore::<HashMap<String, EntityRecord>>::new());
    let mem = EntityMemory::new(store, Namespace::new(TenantId::new("t")));
    let ctx = ExecutionContext::new();
    mem.set_entity(&ctx, "A", "1").await?;
    mem.set_entity(&ctx, "B", "2").await?;
    mem.remove(&ctx, "A").await?;
    assert!(mem.entity(&ctx, "A").await?.is_none());
    assert_eq!(mem.entity(&ctx, "B").await?.as_deref(), Some("2"));
    Ok(())
}

#[tokio::test]
async fn entity_remove_idempotent_when_absent() -> Result<()> {
    let store: Arc<dyn Store<HashMap<String, EntityRecord>>> =
        Arc::new(InMemoryStore::<HashMap<String, EntityRecord>>::new());
    let mem = EntityMemory::new(store, Namespace::new(TenantId::new("t")));
    let ctx = ExecutionContext::new();
    mem.remove(&ctx, "ghost").await?; // no-op, no panic
    Ok(())
}

#[tokio::test]
async fn entity_clear_drops_everything() -> Result<()> {
    let store: Arc<dyn Store<HashMap<String, EntityRecord>>> =
        Arc::new(InMemoryStore::<HashMap<String, EntityRecord>>::new());
    let mem = EntityMemory::new(store, Namespace::new(TenantId::new("t")));
    let ctx = ExecutionContext::new();
    mem.set_entity(&ctx, "A", "1").await?;
    mem.set_entity(&ctx, "B", "2").await?;
    mem.clear(&ctx).await?;
    assert!(mem.all(&ctx).await?.is_empty());
    Ok(())
}

#[tokio::test]
async fn entity_set_preserves_created_at_on_update() -> Result<()> {
    let store: Arc<dyn Store<HashMap<String, EntityRecord>>> =
        Arc::new(InMemoryStore::<HashMap<String, EntityRecord>>::new());
    let mem = EntityMemory::new(store, Namespace::new(TenantId::new("t")));
    let ctx = ExecutionContext::new();
    mem.set_entity(&ctx, "Alice", "v1").await?;
    let initial = mem.entity_record(&ctx, "Alice").await?.unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(15)).await;
    mem.set_entity(&ctx, "Alice", "v2").await?;
    let updated = mem.entity_record(&ctx, "Alice").await?.unwrap();
    assert_eq!(updated.fact, "v2");
    assert_eq!(
        updated.created_at, initial.created_at,
        "created_at must be preserved across updates"
    );
    assert!(
        updated.last_seen > initial.last_seen,
        "last_seen must advance on update"
    );
    Ok(())
}

#[tokio::test]
async fn entity_touch_refreshes_last_seen_without_changing_fact() -> Result<()> {
    let store: Arc<dyn Store<HashMap<String, EntityRecord>>> =
        Arc::new(InMemoryStore::<HashMap<String, EntityRecord>>::new());
    let mem = EntityMemory::new(store, Namespace::new(TenantId::new("t")));
    let ctx = ExecutionContext::new();
    mem.set_entity(&ctx, "Alice", "v1").await?;
    let initial = mem.entity_record(&ctx, "Alice").await?.unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(15)).await;
    let touched = mem.touch(&ctx, "Alice").await?;
    assert!(touched);
    let after = mem.entity_record(&ctx, "Alice").await?.unwrap();
    assert_eq!(after.fact, initial.fact);
    assert!(after.last_seen > initial.last_seen);
    Ok(())
}

#[tokio::test]
async fn entity_touch_returns_false_for_absent_entity() -> Result<()> {
    let store: Arc<dyn Store<HashMap<String, EntityRecord>>> =
        Arc::new(InMemoryStore::<HashMap<String, EntityRecord>>::new());
    let mem = EntityMemory::new(store, Namespace::new(TenantId::new("t")));
    let ctx = ExecutionContext::new();
    let touched = mem.touch(&ctx, "ghost").await?;
    assert!(!touched);
    Ok(())
}

#[tokio::test]
async fn entity_prune_drops_records_past_ttl() -> Result<()> {
    let store: Arc<dyn Store<HashMap<String, EntityRecord>>> =
        Arc::new(InMemoryStore::<HashMap<String, EntityRecord>>::new());
    let mem = EntityMemory::new(store, Namespace::new(TenantId::new("t")));
    let ctx = ExecutionContext::new();
    mem.set_entity(&ctx, "old", "stale").await?;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    mem.set_entity(&ctx, "new", "fresh").await?;
    // TTL = 25ms — the "old" record's last_seen is 50+ms ago, the
    // "new" record's last_seen is well within the window.
    let pruned = mem
        .prune_older_than(&ctx, std::time::Duration::from_millis(25))
        .await?;
    assert_eq!(pruned, 1);
    assert!(mem.entity(&ctx, "old").await?.is_none());
    assert_eq!(mem.entity(&ctx, "new").await?.as_deref(), Some("fresh"));
    Ok(())
}

#[tokio::test]
async fn entity_prune_returns_zero_when_namespace_is_empty() -> Result<()> {
    let store: Arc<dyn Store<HashMap<String, EntityRecord>>> =
        Arc::new(InMemoryStore::<HashMap<String, EntityRecord>>::new());
    let mem = EntityMemory::new(store, Namespace::new(TenantId::new("t")));
    let ctx = ExecutionContext::new();
    let pruned = mem
        .prune_older_than(&ctx, std::time::Duration::from_secs(1))
        .await?;
    assert_eq!(pruned, 0);
    Ok(())
}

// ── SemanticMemory ─────────────────────────────────────────────────────────

/// Stub embedder: deterministic fingerprint based on the input length and
/// first byte (no ML, just enough to verify wiring).
struct StubEmbedder {
    dimension: usize,
}

#[async_trait]
impl Embedder for StubEmbedder {
    fn dimension(&self) -> usize {
        self.dimension
    }
    async fn embed(&self, text: &str, _ctx: &ExecutionContext) -> Result<Embedding> {
        let seed = text.bytes().next().unwrap_or(0);
        let vector: Vec<f32> = (0..self.dimension)
            .map(|i| f32::from(seed) + i as f32)
            .collect();
        Ok(Embedding::new(vector).with_usage(EmbeddingUsage::new(text.len() as u32)))
    }
}

/// Stub vector store: holds (Document, Vec<f32>) pairs in a Mutex<Vec>,
/// returns documents in insertion order on search.
struct StubVectorStore {
    dimension: usize,
    rows: Mutex<Vec<(Namespace, Document, Vec<f32>)>>,
}

impl StubVectorStore {
    fn new(dimension: usize) -> Self {
        Self {
            dimension,
            rows: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl VectorStore for StubVectorStore {
    fn dimension(&self) -> usize {
        self.dimension
    }
    async fn add(
        &self,
        _ctx: &ExecutionContext,
        ns: &Namespace,
        document: Document,
        vector: Vec<f32>,
    ) -> Result<()> {
        let entry = (ns.clone(), document, vector);
        {
            let mut g = self.rows.lock();
            g.push(entry);
        }
        Ok(())
    }
    async fn search(
        &self,
        _ctx: &ExecutionContext,
        ns: &Namespace,
        _query_vector: &[f32],
        top_k: usize,
    ) -> Result<Vec<Document>> {
        let out = {
            let g = self.rows.lock();
            g.iter()
                .filter(|(stored_ns, _, _)| stored_ns == ns)
                .map(|(_, doc, _)| doc.clone())
                .take(top_k)
                .collect::<Vec<_>>()
        };
        Ok(out)
    }
    async fn count(
        &self,
        _ctx: &ExecutionContext,
        ns: &Namespace,
        _filter: Option<&entelix_memory::VectorFilter>,
    ) -> Result<usize> {
        let g = self.rows.lock();
        Ok(g.iter().filter(|(stored_ns, _, _)| stored_ns == ns).count())
    }
    async fn list(
        &self,
        _ctx: &ExecutionContext,
        ns: &Namespace,
        _filter: Option<&entelix_memory::VectorFilter>,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Document>> {
        let g = self.rows.lock();
        Ok(g.iter()
            .filter(|(stored_ns, _, _)| stored_ns == ns)
            .map(|(_, doc, _)| doc.clone())
            .skip(offset)
            .take(limit)
            .collect())
    }
}

#[tokio::test]
async fn semantic_memory_add_and_search_round_trip() -> Result<()> {
    let embedder: Arc<StubEmbedder> = Arc::new(StubEmbedder { dimension: 8 });
    let vector_store: Arc<StubVectorStore> = Arc::new(StubVectorStore::new(8));
    let ns = Namespace::new(TenantId::new("t")).with_scope("agent");

    let mem = SemanticMemory::new(embedder, vector_store, ns)?;
    let ctx = ExecutionContext::new();
    mem.add(&ctx, Document::new("alpha")).await?;
    mem.add(&ctx, Document::new("beta")).await?;

    let results = mem.search(&ctx, "query", 5).await?;
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].content, "alpha");
    assert_eq!(results[1].content, "beta");
    Ok(())
}

/// Misbehaving embedder that returns one fewer vector than texts —
/// the kind of bug `SemanticMemory::batch_add` must surface rather
/// than silently zip-truncating the document set.
struct ShortBatchEmbedder {
    dimension: usize,
}

#[async_trait]
impl Embedder for ShortBatchEmbedder {
    fn dimension(&self) -> usize {
        self.dimension
    }
    async fn embed(&self, _text: &str, _ctx: &ExecutionContext) -> Result<Embedding> {
        Ok(Embedding::new(vec![0.0; self.dimension]))
    }
    async fn embed_batch(
        &self,
        texts: &[String],
        _ctx: &ExecutionContext,
    ) -> Result<Vec<Embedding>> {
        Ok(texts
            .iter()
            .skip(1)
            .map(|_| Embedding::new(vec![0.0; self.dimension]))
            .collect())
    }
}

#[tokio::test]
async fn semantic_memory_batch_add_surfaces_embedder_count_mismatch() -> Result<()> {
    // Embedder returns N-1 vectors for N documents — must NOT
    // silently drop the trailing document via zip-truncation.
    let embedder: Arc<ShortBatchEmbedder> = Arc::new(ShortBatchEmbedder { dimension: 4 });
    let vector_store: Arc<StubVectorStore> = Arc::new(StubVectorStore::new(4));
    let mem = SemanticMemory::new(embedder, vector_store, Namespace::new(TenantId::new("t")))?;
    let ctx = ExecutionContext::new();
    let docs = vec![
        Document::new("alpha"),
        Document::new("beta"),
        Document::new("gamma"),
    ];
    match mem.batch_add(&ctx, docs).await {
        Err(entelix_core::Error::Config(msg)) => {
            assert!(
                msg.contains("returned 2 vectors for 3 documents"),
                "expected count-mismatch detail in error message, got: {msg}"
            );
        }
        Ok(()) => panic!("expected Error::Config from count mismatch, got Ok"),
        Err(other) => panic!("expected Error::Config, got {other:?}"),
    }
    Ok(())
}

#[tokio::test]
async fn semantic_memory_dimension_mismatch_returns_config_error() {
    let embedder: Arc<StubEmbedder> = Arc::new(StubEmbedder { dimension: 8 });
    let vector_store: Arc<StubVectorStore> = Arc::new(StubVectorStore::new(16)); // mismatch
    match SemanticMemory::new(embedder, vector_store, Namespace::new(TenantId::new("t"))) {
        Err(entelix_core::Error::Config(_)) => {}
        Ok(_) => panic!("expected dimension-mismatch error, got Ok"),
        Err(other) => panic!("expected Config error, got {other:?}"),
    }
}

#[tokio::test]
async fn semantic_memory_namespaces_isolate_documents() -> Result<()> {
    let embedder: Arc<StubEmbedder> = Arc::new(StubEmbedder { dimension: 4 });
    let vector_store: Arc<StubVectorStore> = Arc::new(StubVectorStore::new(4));

    let alpha = SemanticMemory::new(
        embedder.clone(),
        vector_store.clone(),
        Namespace::new(TenantId::new("tenant")).with_scope("alpha"),
    )?;
    let beta = SemanticMemory::new(
        embedder,
        vector_store,
        Namespace::new(TenantId::new("tenant")).with_scope("beta"),
    )?;

    let ctx = ExecutionContext::new();
    alpha.add(&ctx, Document::new("alpha-doc")).await?;
    beta.add(&ctx, Document::new("beta-doc")).await?;

    let alpha_hits = alpha.search(&ctx, "q", 5).await?;
    let beta_hits = beta.search(&ctx, "q", 5).await?;
    assert_eq!(alpha_hits.len(), 1);
    assert_eq!(alpha_hits[0].content, "alpha-doc");
    assert_eq!(beta_hits.len(), 1);
    assert_eq!(beta_hits[0].content, "beta-doc");
    Ok(())
}

#[tokio::test]
async fn semantic_memory_count_and_list_pass_through_to_vector_store() -> Result<()> {
    let embedder: Arc<StubEmbedder> = Arc::new(StubEmbedder { dimension: 4 });
    let vector_store: Arc<StubVectorStore> = Arc::new(StubVectorStore::new(4));
    let mem = SemanticMemory::new(
        embedder,
        vector_store,
        Namespace::new(TenantId::new("tenant")).with_scope("agent"),
    )?;
    let ctx = ExecutionContext::new();
    for label in ["a", "b", "c", "d"] {
        mem.add(&ctx, Document::new(label)).await?;
    }

    assert_eq!(mem.count(&ctx, None).await?, 4);

    let page_1 = mem.list(&ctx, None, 2, 0).await?;
    assert_eq!(page_1.len(), 2);
    assert_eq!(page_1[0].content, "a");
    assert_eq!(page_1[1].content, "b");

    let page_2 = mem.list(&ctx, None, 2, 2).await?;
    assert_eq!(page_2.len(), 2);
    assert_eq!(page_2[0].content, "c");
    assert_eq!(page_2[1].content, "d");
    Ok(())
}

#[tokio::test]
async fn vector_store_default_count_and_list_surface_config_error() {
    use entelix_memory::VectorStore;
    let store = StubAddOnly::new(4);
    let ctx = ExecutionContext::new();
    let ns = Namespace::new(TenantId::new("t"));
    match store.count(&ctx, &ns, None).await {
        Err(entelix_core::Error::Config(msg)) => assert!(msg.contains("count")),
        other => panic!("expected Config error from default count, got {other:?}"),
    }
    match store.list(&ctx, &ns, None, 10, 0).await {
        Err(entelix_core::Error::Config(msg)) => assert!(msg.contains("list")),
        other => panic!("expected Config error from default list, got {other:?}"),
    }
}

/// Minimal `VectorStore` that overrides only the required methods —
/// exercises the default `count`/`list` paths returning `Error::Config`.
struct StubAddOnly {
    dimension: usize,
}
impl StubAddOnly {
    const fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}
#[async_trait]
impl entelix_memory::VectorStore for StubAddOnly {
    fn dimension(&self) -> usize {
        self.dimension
    }
    async fn add(
        &self,
        _ctx: &ExecutionContext,
        _ns: &Namespace,
        _document: Document,
        _vector: Vec<f32>,
    ) -> Result<()> {
        Ok(())
    }
    async fn search(
        &self,
        _ctx: &ExecutionContext,
        _ns: &Namespace,
        _query_vector: &[f32],
        _top_k: usize,
    ) -> Result<Vec<Document>> {
        Ok(Vec::new())
    }
}

#[tokio::test]
async fn vector_store_default_batch_add_bails_on_cancellation() {
    use entelix_core::cancellation::CancellationToken;
    use entelix_memory::VectorStore;

    // The default `batch_add` impl loops over `add` — pre-cancel the
    // context so the loop bails on the very first iteration with
    // `Error::Cancelled` rather than draining the full batch.
    let store = StubAddOnly::new(4);
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let ctx = ExecutionContext::with_cancellation(cancellation);
    let ns = Namespace::new(TenantId::new("t"));
    let items: Vec<(Document, Vec<f32>)> = (0..50)
        .map(|i| (Document::new(format!("d{i}")), vec![0.0; 4]))
        .collect();
    match store.batch_add(&ctx, &ns, items).await {
        Err(entelix_core::Error::Cancelled) => {}
        other => panic!("expected Error::Cancelled, got {other:?}"),
    }
}

#[tokio::test]
async fn embedder_default_batch_bails_on_cancellation() {
    use entelix_core::cancellation::CancellationToken;
    use entelix_memory::Embedder;

    let embedder = StubEmbedder { dimension: 4 };
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let ctx = ExecutionContext::with_cancellation(cancellation);
    let texts: Vec<String> = (0..50).map(|i| format!("t{i}")).collect();
    match embedder.embed_batch(&texts, &ctx).await {
        Err(entelix_core::Error::Cancelled) => {}
        other => panic!("expected Error::Cancelled, got {other:?}"),
    }
}
