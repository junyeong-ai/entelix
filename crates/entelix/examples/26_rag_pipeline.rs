//! Example 26 — RAG ingestion + corrective retrieval, end-to-end.
//!
//! Build: `cargo run --example 26_rag_pipeline -p entelix`
//! Run: same.
//!
//! Demonstrates the full RAG path the SDK ships:
//!
//! 1. **`IngestionPipeline`** — `DocumentLoader` → `TokenCountSplitter`
//!    → `Embedder` → `VectorStore` composed end-to-end. Drains a
//!    small in-memory corpus, splits each doc into token-budgeted
//!    chunks, embeds, stores.
//! 2. **`EmbeddingRetriever`** — wraps the `VectorStore` +
//!    `Embedder` pair into the `Retriever` shape every retrieval-
//!    aware agent consumes. Filter + min-score handling for free.
//! 3. **`create_corrective_rag_agent`** — the CRAG (Yan et al. 2024)
//!    recipe. Returns a standard `Agent<CorrectiveRagState>` that
//!    retrieves, grades, rewrites-on-failure, generates. Runs over
//!    operator-supplied `RetrievalGrader` / `QueryRewriter` /
//!    generator primitives; this example wires deterministic
//!    keyword-based stand-ins so the demo is hermetic.
//!
//! No external API dependency — runs deterministically in CI.

#![allow(clippy::print_stdout)]

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use entelix::ir::{ContentPart, Message, Role};
use entelix::{
    ByteCountTokenCounter, CorrectiveRagState, CragConfig, Document, DocumentLoader,
    DocumentStream, Embedder, Embedding, EmbeddingRetriever, ExecutionContext, GradeVerdict,
    InMemoryVectorStore, IngestionPipeline, Namespace, QueryRewriter, Result, RetrievalGrader,
    RetrievedDocument, Runnable, Source, TenantId, TokenCountSplitter, TokenCounter,
    create_corrective_rag_agent,
};
use futures::stream;

const CORPUS: &[(&str, &str)] = &[
    (
        "rust-async",
        "Rust async uses futures and .await syntax. Async functions desugar into state \
         machines that the runtime — typically tokio — drives to completion.",
    ),
    (
        "tokio-runtime",
        "Tokio is a popular runtime for async Rust. It provides a multi-threaded executor, \
         async sockets, timers, and synchronisation primitives like Mutex and channels.",
    ),
    (
        "entelix-agents",
        "entelix is an agentic AI SDK in Rust. It supports ReAct, Supervisor, and \
         Hierarchical agent recipes with token-aware chunking and vendor-portable IR.",
    ),
    (
        "rust-graphs",
        "Graph algorithms in Rust often use petgraph. Common shapes are adjacency-list \
         and adjacency-matrix; petgraph offers DAG support, shortest-path, topological sort.",
    ),
];

#[tokio::main]
async fn main() -> Result<()> {
    let ctx = ExecutionContext::new();
    let namespace = Namespace::new(TenantId::new("acme"));

    // ── 1. Ingestion ─────────────────────────────────────────────

    let loader = InlineLoader::new(CORPUS, &namespace);
    let counter: Arc<dyn TokenCounter> = Arc::new(ByteCountTokenCounter::new());
    let splitter = TokenCountSplitter::new(Arc::clone(&counter))
        .with_chunk_size(40)
        .with_chunk_overlap(8);
    let embedder = Arc::new(BowEmbedder::corpus_vocab());
    let store = Arc::new(InMemoryVectorStore::new(embedder.dimension()));

    let pipeline = IngestionPipeline::builder(
        loader,
        splitter,
        Arc::clone(&embedder),
        Arc::clone(&store),
        namespace.clone(),
    )
    .build();

    let report = pipeline.run(&ctx).await?;
    println!("=== Ingestion ===");
    println!("documents loaded: {}", report.documents_loaded);
    println!("chunks indexed:   {}", report.chunks_indexed);
    println!("embedding calls:  {}", report.embedding_calls);
    println!("errors:           {}", report.errors.len());

    // ── 2. Retrieval ─────────────────────────────────────────────

    let retriever = Arc::new(EmbeddingRetriever::new(
        Arc::clone(&embedder),
        Arc::clone(&store),
        namespace.clone(),
    ));

    // ── 3. CRAG agent ────────────────────────────────────────────

    let grader = KeywordGrader;
    let rewriter =
        StaticRewriter::new(&["tokio runtime async executor", "rust async tokio futures"]);
    let generator = TemplatedGenerator;

    let agent = create_corrective_rag_agent(
        retriever,
        grader,
        rewriter,
        generator,
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

    // ── 4. Output ────────────────────────────────────────────────

    println!();
    println!("=== CRAG agent ===");
    println!("rewrite attempts: {}", final_state.attempt);
    println!("graded documents: {}", final_state.graded.len());
    println!("correct subset:   {}", final_state.correct_documents.len());
    println!();
    println!(
        "answer: {}",
        final_state
            .answer
            .unwrap_or_else(|| "<generator produced no answer>".into())
    );
    Ok(())
}

// ── Components ───────────────────────────────────────────────────

/// Loader that yields a fixed corpus.
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
                    Source::now("inline://", "inline-corpus"),
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

/// `BoW` embedder — every recognised word increments one basis
/// component, L2-normalised. Deterministic, no IO.
struct BowEmbedder {
    vocab: HashMap<String, usize>,
    dimension: usize,
}

impl BowEmbedder {
    fn corpus_vocab() -> Self {
        let words: &[&str] = &[
            "rust",
            "async",
            "tokio",
            "runtime",
            "executor",
            "future",
            "futures",
            "agent",
            "agents",
            "entelix",
            "react",
            "supervisor",
            "graph",
            "graphs",
            "petgraph",
            "dag",
            "path",
            "memory",
            "chunking",
            "rag",
            "retrieval",
        ];
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

/// Grader: Correct iff doc content shares any > 2-letter keyword with
/// the query; Ambiguous otherwise. Real deployments wire
/// `LlmRetrievalGrader`.
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
        let query_words: HashSet<_> = query
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

/// Rewriter: returns the next pre-canned query per call. The cursor
/// rides on an atomic counter so the rewriter stays `Send + Sync`
/// without locking. Real deployments wire `LlmQueryRewriter`.
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

/// Generator: echoes the user message it received as the answer.
/// Real deployments wire a `ChatModel` here.
struct TemplatedGenerator;

#[async_trait]
impl Runnable<Vec<Message>, Message> for TemplatedGenerator {
    async fn invoke(&self, input: Vec<Message>, _ctx: &ExecutionContext) -> Result<Message> {
        let user_text = input
            .last()
            .and_then(|m| {
                m.content.iter().find_map(|p| match p {
                    ContentPart::Text { text, .. } => Some(text.clone()),
                    _ => None,
                })
            })
            .unwrap_or_else(|| "(no input)".to_owned());
        let summary: String = user_text.chars().take(160).collect();
        Ok(Message::new(
            Role::Assistant,
            vec![ContentPart::text(format!(
                "Based on the retrieved documents: {summary}..."
            ))],
        ))
    }
}
