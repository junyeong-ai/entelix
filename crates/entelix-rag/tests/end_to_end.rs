//! End-to-end integration test for the RAG primitive surface.
//!
//! Verifies that the `IngestionPipeline` → `EmbeddingRetriever` →
//! `Agent<CorrectiveRagState>` path composes cleanly with no live
//! API dependency. A single deterministic embedder + grader keeps
//! the test hermetic; the goal is regression coverage on the
//! composition (not on retrieval quality, which is the corpus and
//! embedder's domain).

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use entelix_agents::{Agent, AgentBuilder, AgentEvent, CaptureSink};
use entelix_core::ir::{ContentPart, Message, Role};
use entelix_core::{ByteCountTokenCounter, ExecutionContext, Result, TenantId, TokenCounter};
use entelix_memory::{
    Document as RetrievedDocument, Embedder, Embedding, EmbeddingRetriever, InMemoryVectorStore,
    Namespace, Retriever,
};
use entelix_rag::{
    CORRECTIVE_RAG_AGENT_NAME, CorrectiveRagState, CragConfig, Document, DocumentLoader,
    DocumentStream, GradeVerdict, IngestionPipeline, QueryRewriter, RetrievalGrader, Source,
    TokenCountSplitter, build_corrective_rag_graph, create_corrective_rag_agent,
};
use entelix_runnable::Runnable;
use futures::stream;

const CORPUS: &[(&str, &str)] = &[
    (
        "rust-async",
        "Rust async uses futures and the .await syntax. The runtime — typically tokio \
         — drives the futures to completion.",
    ),
    (
        "tokio-runtime",
        "Tokio is an async Rust runtime. Multi-threaded executor, async sockets, timers, \
         channels.",
    ),
    (
        "rust-graphs",
        "Graph algorithms in Rust commonly use petgraph. DAG support and topological sort.",
    ),
];

const VOCAB: &[&str] = &[
    "rust", "async", "tokio", "runtime", "executor", "future", "futures", "graph", "graphs",
    "petgraph", "dag",
];

#[tokio::test]
async fn rag_pipeline_to_crag_agent_composes_end_to_end() -> Result<()> {
    let ctx = ExecutionContext::new();
    let namespace = Namespace::new(TenantId::new("acme"));

    let counter: Arc<dyn TokenCounter> = Arc::new(ByteCountTokenCounter::new());
    let splitter = TokenCountSplitter::new(counter)
        .with_chunk_size(40)
        .with_chunk_overlap(8);
    let embedder = Arc::new(BowEmbedder::new(VOCAB));
    let store = Arc::new(InMemoryVectorStore::new(embedder.dimension()));

    let pipeline = IngestionPipeline::builder(
        InlineLoader::new(CORPUS, &namespace),
        splitter,
        Arc::clone(&embedder),
        Arc::clone(&store),
        namespace.clone(),
    )
    .build();

    let report = pipeline.run(&ctx).await?;
    assert_eq!(report.documents_loaded, 3);
    assert!(report.chunks_indexed >= 3);
    assert_eq!(report.embedding_calls, 3);
    assert!(
        report.errors.is_empty(),
        "ingestion errors: {:?}",
        report.errors
    );
    assert_eq!(store.total_slots() as u64, report.chunks_indexed);

    let retriever = Arc::new(EmbeddingRetriever::new(
        Arc::clone(&embedder),
        Arc::clone(&store),
        namespace.clone(),
    ));

    let agent = create_corrective_rag_agent(
        retriever,
        KeywordGrader,
        StaticRewriter::new(&["tokio runtime async"]),
        EchoGenerator,
        CragConfig::new()
            .with_retrieval_top_k(3)
            .with_min_correct_fraction(0.3)
            .with_max_rewrite_attempts(2),
    )?;

    let final_state = agent
        .execute(
            CorrectiveRagState::from_query("How does async Rust work with tokio?"),
            &ctx,
        )
        .await?
        .state;

    assert!(
        !final_state.graded.is_empty(),
        "CRAG must produce graded docs"
    );
    assert!(
        !final_state.correct_documents.is_empty(),
        "keyword overlap must yield Correct verdicts"
    );
    assert_eq!(
        final_state.answer.as_deref(),
        Some("answer"),
        "generator output must land verbatim on state.answer"
    );
    assert_eq!(
        final_state.original_query, "How does async Rust work with tokio?",
        "original query must survive every pass"
    );
    assert_eq!(
        final_state.attempt, 0,
        "BoW retrieval ranks the relevant doc above the keyword threshold; no rewrite needed"
    );
    Ok(())
}

/// CRAG agent must integrate with the `Agent<S>` lifecycle — the
/// slice's design claim. Wire a `CaptureSink` and verify every run
/// emits `Started` + `Complete` events bound to the canonical
/// `corrective-rag` agent name. Regression gate against future
/// refactors that bypass the standard envelope.
#[tokio::test]
async fn crag_agent_emits_lifecycle_events_through_capture_sink() -> Result<()> {
    let ctx = ExecutionContext::new();
    let namespace = Namespace::new(TenantId::new("acme"));

    let counter: Arc<dyn TokenCounter> = Arc::new(ByteCountTokenCounter::new());
    let embedder = Arc::new(BowEmbedder::new(VOCAB));
    let store = Arc::new(InMemoryVectorStore::new(embedder.dimension()));
    IngestionPipeline::builder(
        InlineLoader::new(CORPUS, &namespace),
        TokenCountSplitter::new(Arc::clone(&counter)).with_chunk_size(40),
        Arc::clone(&embedder),
        Arc::clone(&store),
        namespace.clone(),
    )
    .build()
    .run(&ctx)
    .await?;

    let retriever = Arc::new(EmbeddingRetriever::new(
        Arc::clone(&embedder),
        Arc::clone(&store),
        namespace.clone(),
    ));

    // Build the graph by hand, then wrap in a custom Agent with a
    // CaptureSink — exercises the lower-level builder path the
    // canonical create_corrective_rag_agent abstracts.
    let graph = build_corrective_rag_graph(
        retriever,
        KeywordGrader,
        StaticRewriter::new(&["unused"]),
        EchoGenerator,
        CragConfig::new()
            .with_retrieval_top_k(3)
            .with_min_correct_fraction(0.3),
    )?;
    let sink: CaptureSink<CorrectiveRagState> = CaptureSink::new();
    let captured = sink.clone();
    let agent: Agent<CorrectiveRagState> = AgentBuilder::default()
        .with_name(CORRECTIVE_RAG_AGENT_NAME)
        .with_runnable(graph)
        .add_sink(sink)
        .build()?;

    let _ = agent
        .execute(CorrectiveRagState::from_query("rust async tokio?"), &ctx)
        .await?;

    let events = captured.events();
    assert!(
        !events.is_empty(),
        "agent dispatch must emit at least one event"
    );
    assert!(
        events.iter().any(
            |e| matches!(e, AgentEvent::Started { agent, .. } if agent == CORRECTIVE_RAG_AGENT_NAME)
        ),
        "Started event must fire with the canonical agent name; got {events:?}",
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::Complete { .. })),
        "Complete event must fire on success; got {events:?}",
    );
    Ok(())
}

#[tokio::test]
async fn rag_pipeline_namespace_isolates_across_tenants() -> Result<()> {
    let ctx = ExecutionContext::new();
    let alice = Namespace::new(TenantId::new("alice"));
    let bob = Namespace::new(TenantId::new("bob"));

    let counter: Arc<dyn TokenCounter> = Arc::new(ByteCountTokenCounter::new());
    let embedder = Arc::new(BowEmbedder::new(VOCAB));
    let store = Arc::new(InMemoryVectorStore::new(embedder.dimension()));

    // Alice's corpus reaches the store; Bob's namespace stays empty.
    IngestionPipeline::builder(
        InlineLoader::new(CORPUS, &alice),
        TokenCountSplitter::new(Arc::clone(&counter))
            .with_chunk_size(40)
            .with_chunk_overlap(0),
        Arc::clone(&embedder),
        Arc::clone(&store),
        alice.clone(),
    )
    .build()
    .run(&ctx)
    .await?;

    let bob_retriever = EmbeddingRetriever::new(Arc::clone(&embedder), Arc::clone(&store), bob);
    let hits = bob_retriever
        .retrieve(
            entelix_memory::RetrievalQuery::new("rust async tokio", 10),
            &ctx,
        )
        .await?;
    assert!(
        hits.is_empty(),
        "tenant boundary must hold: bob saw {hits:?}"
    );
    Ok(())
}

// ── Components ───────────────────────────────────────────────────

struct InlineLoader {
    docs: Vec<Document>,
}

impl InlineLoader {
    fn new(corpus: &[(&str, &str)], namespace: &Namespace) -> Self {
        let docs = corpus
            .iter()
            .map(|(id, content)| {
                Document::root(
                    *id,
                    *content,
                    Source::now("inline://", "inline"),
                    namespace.clone(),
                )
            })
            .collect();
        Self { docs }
    }
}

#[async_trait]
impl DocumentLoader for InlineLoader {
    fn name(&self) -> &'static str {
        "inline"
    }

    async fn load<'a>(&'a self, _ctx: &'a ExecutionContext) -> Result<DocumentStream<'a>> {
        let items: Vec<Result<Document>> = self.docs.iter().cloned().map(Ok).collect();
        Ok(Box::pin(stream::iter(items)))
    }
}

struct BowEmbedder {
    vocab: HashMap<String, usize>,
    dimension: usize,
}

impl BowEmbedder {
    fn new(words: &[&str]) -> Self {
        let dimension = words.len();
        let vocab = words
            .iter()
            .enumerate()
            .map(|(i, w)| ((*w).to_owned(), i))
            .collect();
        Self { vocab, dimension }
    }
}

#[async_trait]
impl Embedder for BowEmbedder {
    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn embed(&self, text: &str, _ctx: &ExecutionContext) -> Result<Embedding> {
        let mut v = vec![0.0_f32; self.dimension];
        for word in text.to_lowercase().split(|c: char| !c.is_alphanumeric()) {
            if let Some(&idx) = self.vocab.get(word)
                && let Some(slot) = v.get_mut(idx)
            {
                *slot += 1.0;
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        Ok(Embedding::new(v))
    }
}

struct KeywordGrader;

#[async_trait]
impl RetrievalGrader for KeywordGrader {
    fn name(&self) -> &'static str {
        "keyword"
    }

    async fn grade(
        &self,
        query: &str,
        doc: &RetrievedDocument,
        _ctx: &ExecutionContext,
    ) -> Result<GradeVerdict> {
        let query_words: Vec<String> = query
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 2)
            .map(str::to_owned)
            .collect();
        let content = doc.content.to_lowercase();
        let overlap = query_words.iter().any(|w| content.contains(w.as_str()));
        Ok(if overlap {
            GradeVerdict::Correct
        } else {
            GradeVerdict::Ambiguous
        })
    }
}

struct StaticRewriter {
    replies: Vec<String>,
    cursor: AtomicUsize,
}

impl StaticRewriter {
    fn new(replies: &[&str]) -> Self {
        Self {
            replies: replies.iter().map(|s| (*s).to_owned()).collect(),
            cursor: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl QueryRewriter for StaticRewriter {
    fn name(&self) -> &'static str {
        "static"
    }

    async fn rewrite(
        &self,
        _original: &str,
        _previous: &[String],
        _ctx: &ExecutionContext,
    ) -> Result<String> {
        let idx = self.cursor.fetch_add(1, Ordering::Relaxed);
        Ok(self
            .replies
            .get(idx)
            .cloned()
            .unwrap_or_else(|| "<exhausted>".to_owned()))
    }
}

struct EchoGenerator;

#[async_trait]
impl Runnable<Vec<Message>, Message> for EchoGenerator {
    async fn invoke(&self, _input: Vec<Message>, _ctx: &ExecutionContext) -> Result<Message> {
        Ok(Message::new(
            Role::Assistant,
            vec![ContentPart::text("answer")],
        ))
    }
}
