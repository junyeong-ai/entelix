//! `IngestionPipeline` — typed composition of every RAG primitive
//! into one runnable end-to-end ingestion path.
//!
//! Wires [`DocumentLoader`] → [`TextSplitter`] → optional
//! [`Chunker`] chain → [`Embedder`] → [`VectorStore`] with one
//! [`Namespace`] supplying the multi-tenant boundary. One
//! [`Self::run`] call drains the source, processes every document
//! the loader produces, and surfaces an [`IngestReport`] summarising
//! what landed in the index and what failed.
//!
//! ## Why a typed composition (not a `Vec<Arc<dyn …>>` plumbing)
//!
//! Each pipeline component is generic in this crate's surface
//! (`L: DocumentLoader, S: TextSplitter, E: Embedder, V:
//! VectorStore`) so monomorphisation produces an inlined hot path —
//! the chunk loop never pays a vtable dispatch per invocation. The
//! [`Chunker`] chain alone is `Vec<Arc<dyn Chunker>>` because
//! chains are runtime-variable in length and content; the
//! single-occurrence components are typed.
//!
//! ## Partial-success contract
//!
//! Per-document failures don't abort the pipeline. The report
//! collects them as [`IngestError`] entries and `run()` keeps
//! draining; the loader-level `Result<Document>` items, the splitter
//! and chunker passes, and the embedder + store calls all
//! contribute. Operators decide whether a failure count is
//! actionable from the report — `run()` itself only returns
//! `Err` for *structural* failures (loader open rejection,
//! cancellation, deadline).

use std::sync::Arc;

use entelix_core::{Error, ExecutionContext, Result};
use entelix_memory::{
    Document as RetrievedDocument, DocumentId as RetrievedDocumentId, Embedder, Namespace,
    VectorStore,
};
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::chunker::Chunker;
use crate::document::Document;
use crate::loader::DocumentLoader;
use crate::splitter::TextSplitter;

/// Reserved key on the persisted `metadata` map under which the
/// pipeline stamps `Source` + `Lineage` + `namespace`. Carries the
/// `entelix` prefix so an operator's own metadata fields never
/// collide. Retrieval-side consumers reach back to provenance
/// through this nested object.
pub const PROVENANCE_METADATA_KEY: &str = "entelix";

/// Outcome counters and per-document failure list a single
/// [`IngestionPipeline::run`] produces.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct IngestReport {
    /// Documents the loader yielded successfully.
    pub documents_loaded: u64,
    /// Documents that reached the embedder + store stages
    /// successfully (one entry per chunk produced by splitting).
    pub chunks_indexed: u64,
    /// Embedding API calls the pipeline made — useful for cost
    /// reconciliation against vendor-side dashboards. One entry per
    /// `embed_batch` invocation, regardless of batch size.
    pub embedding_calls: u64,
    /// Per-document errors. The pipeline does NOT abort on these —
    /// the report accumulates them and `run` drains the rest of
    /// the source. Operators decide whether a non-empty list is
    /// actionable.
    pub errors: Vec<IngestError>,
}

impl IngestReport {
    /// Whether the pipeline indexed every document the loader
    /// produced without per-document errors. `true` does NOT mean
    /// the loader hit zero documents — pair with
    /// [`Self::documents_loaded`] for the "found anything"
    /// distinction.
    #[must_use]
    pub const fn is_clean(&self) -> bool {
        self.errors.is_empty()
    }
}

/// One per-document failure recorded during ingestion. Carries the
/// originating document id (when known) and a stage label
/// identifying which pipeline phase failed.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub struct IngestError {
    /// Stage label. One of `"load"`, `"chunk"`, `"embed"`,
    /// `"store"`. Carried as `String` (rather than `&'static str`)
    /// so persisted reports reconstruct via serde without forcing
    /// every stage label to be a string literal at deserialise
    /// time.
    pub stage: String,
    /// Document id of the item that failed, when the failure
    /// happened after an id was stamped. `"<unknown>"` for
    /// loader-side failures that never produced a document.
    pub document_id: String,
    /// LLM-/operator-facing message. Renders the originating
    /// `entelix_core::Error` through its `Display` impl.
    pub message: String,
}

impl IngestError {
    fn from_error(stage: impl Into<String>, document_id: impl Into<String>, err: &Error) -> Self {
        Self {
            stage: stage.into(),
            document_id: document_id.into(),
            message: err.to_string(),
        }
    }
}

/// Builder for [`IngestionPipeline`]. Required components
/// (loader / splitter / embedder / store) come in via [`Self::new`];
/// optional [`Chunker`] entries accumulate via [`Self::add_chunker`].
pub struct IngestionPipelineBuilder<L, S, E: ?Sized, V: ?Sized> {
    loader: L,
    splitter: S,
    embedder: Arc<E>,
    store: Arc<V>,
    chunkers: Vec<Arc<dyn Chunker>>,
    namespace: Namespace,
}

impl<L, S, E, V> IngestionPipelineBuilder<L, S, E, V>
where
    L: DocumentLoader,
    S: TextSplitter,
    E: Embedder + ?Sized,
    V: VectorStore + ?Sized,
{
    /// Append a [`Chunker`] to the chain. Multiple chunkers run in
    /// registration order — `add_chunker(contextual).add_chunker(hyde)`
    /// runs `contextual` first, then `hyde` over its output.
    #[must_use]
    pub fn add_chunker(mut self, chunker: Arc<dyn Chunker>) -> Self {
        self.chunkers.push(chunker);
        self
    }

    /// Finalise into a runnable pipeline.
    #[must_use]
    pub fn build(self) -> IngestionPipeline<L, S, E, V> {
        IngestionPipeline {
            loader: self.loader,
            splitter: self.splitter,
            embedder: self.embedder,
            store: self.store,
            chunkers: self.chunkers,
            namespace: self.namespace,
        }
    }
}

/// End-to-end RAG ingestion pipeline. Construct via
/// [`Self::builder`]; finalise with
/// [`IngestionPipelineBuilder::build`]; drive with [`Self::run`].
pub struct IngestionPipeline<L, S, E: ?Sized, V: ?Sized> {
    loader: L,
    splitter: S,
    embedder: Arc<E>,
    store: Arc<V>,
    chunkers: Vec<Arc<dyn Chunker>>,
    namespace: Namespace,
}

impl<L, S, E, V> IngestionPipeline<L, S, E, V>
where
    L: DocumentLoader,
    S: TextSplitter,
    E: Embedder + ?Sized,
    V: VectorStore + ?Sized,
{
    /// Start a builder bound to the supplied components and
    /// [`Namespace`]. Embedder and store are `Arc<E>` / `Arc<V>`
    /// because production deployments share single instances
    /// across many pipelines (one connection pool, one tenant
    /// scope).
    #[must_use]
    pub fn builder(
        loader: L,
        splitter: S,
        embedder: Arc<E>,
        store: Arc<V>,
        namespace: Namespace,
    ) -> IngestionPipelineBuilder<L, S, E, V> {
        IngestionPipelineBuilder {
            loader,
            splitter,
            embedder,
            store,
            chunkers: Vec::new(),
            namespace,
        }
    }

    /// Drain the loader, process every document, return the
    /// outcome report. Cancellation polls between every loader
    /// item — a long-running ingestion bails within one chunk
    /// boundary of `ctx.cancel()`.
    pub async fn run(&self, ctx: &ExecutionContext) -> Result<IngestReport> {
        let mut report = IngestReport::default();
        let mut stream = self.loader.load(ctx).await?;
        while let Some(item) = stream.next().await {
            if ctx.is_cancelled() {
                return Err(Error::Cancelled);
            }
            match item {
                Err(err) => {
                    report
                        .errors
                        .push(IngestError::from_error("load", "<unknown>", &err));
                }
                Ok(document) => {
                    report.documents_loaded = report.documents_loaded.saturating_add(1);
                    self.process_document(document, ctx, &mut report).await;
                }
            }
        }
        Ok(report)
    }

    /// Per-document pipeline: split → chunker chain → embed →
    /// store. Errors are recorded on the report; the pipeline
    /// continues with the next document.
    async fn process_document(
        &self,
        document: Document,
        ctx: &ExecutionContext,
        report: &mut IngestReport,
    ) {
        let document_id = document.id.as_str().to_owned();
        let mut chunks = self.splitter.split(&document);
        if chunks.is_empty() {
            return;
        }

        // Chunker chain — each chunker runs over the whole vector,
        // batching its work where supported.
        for chunker in &self.chunkers {
            match chunker.process(chunks, ctx).await {
                Ok(transformed) => chunks = transformed,
                Err(err) => {
                    report
                        .errors
                        .push(IngestError::from_error("chunk", &document_id, &err));
                    return;
                }
            }
        }

        if chunks.is_empty() {
            return;
        }

        // Embedder pass — one batch per document so the cost meter
        // sees one call regardless of chunk count.
        let texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
        let embeddings = match self.embedder.embed_batch(&texts, ctx).await {
            Ok(v) => v,
            Err(err) => {
                report
                    .errors
                    .push(IngestError::from_error("embed", &document_id, &err));
                return;
            }
        };
        report.embedding_calls = report.embedding_calls.saturating_add(1);
        if embeddings.len() != chunks.len() {
            report.errors.push(IngestError {
                stage: "embed".to_owned(),
                document_id: document_id.clone(),
                message: format!(
                    "embedder returned {} vectors for {} chunks",
                    embeddings.len(),
                    chunks.len()
                ),
            });
            return;
        }

        // Bridge to the retrieval-side `entelix_memory::Document`
        // shape — provenance lands in `metadata.entelix`.
        let items: Vec<(RetrievedDocument, Vec<f32>)> = chunks
            .into_iter()
            .zip(embeddings)
            .map(|(chunk, emb)| (to_retrieved(chunk), emb.vector))
            .collect();

        if let Err(err) = self
            .store
            .add_batch(ctx, &self.namespace, items.clone())
            .await
        {
            report
                .errors
                .push(IngestError::from_error("store", &document_id, &err));
            return;
        }
        let count = items.len() as u64;
        report.chunks_indexed = report.chunks_indexed.saturating_add(count);
    }
}

/// Convert an ingestion-shape [`Document`] into the retrieval-shape
/// [`entelix_memory::Document`] the vector store stores. Provenance
/// (source + lineage + tenant) lives under the
/// [`PROVENANCE_METADATA_KEY`] reserved key on the persisted
/// `metadata` so retrieval consumers reach back to origin without a
/// second round-trip.
fn to_retrieved(chunk: Document) -> RetrievedDocument {
    let provenance = serde_json::json!({
        "source": chunk.source,
        "lineage": chunk.lineage,
        "namespace": chunk.namespace.render(),
    });
    let mut metadata = match chunk.metadata {
        serde_json::Value::Object(map) => map,
        serde_json::Value::Null => serde_json::Map::new(),
        other => {
            let mut map = serde_json::Map::new();
            map.insert("value".to_owned(), other);
            map
        }
    };
    metadata.insert(PROVENANCE_METADATA_KEY.to_owned(), provenance);
    RetrievedDocument::new(chunk.content)
        .with_doc_id(RetrievedDocumentId::from(chunk.id.as_str()))
        .with_metadata(serde_json::Value::Object(metadata))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::Source;
    use crate::splitter::RecursiveCharacterSplitter;
    use async_trait::async_trait;
    use entelix_memory::{Embedding, EmbeddingUsage, InMemoryVectorStore};
    use std::sync::Mutex;

    fn ns() -> Namespace {
        Namespace::new(entelix_core::TenantId::new("acme"))
    }

    /// In-process loader yielding pre-canned documents.
    struct StubLoader {
        documents: Mutex<Vec<Document>>,
    }

    impl StubLoader {
        fn new(documents: Vec<Document>) -> Self {
            Self {
                documents: Mutex::new(documents),
            }
        }
    }

    #[async_trait]
    impl DocumentLoader for StubLoader {
        fn name(&self) -> &'static str {
            "stub"
        }

        async fn load<'a>(
            &'a self,
            _ctx: &'a ExecutionContext,
        ) -> Result<crate::loader::DocumentStream<'a>> {
            let docs = std::mem::take(&mut *self.documents.lock().unwrap());
            Ok(Box::pin(futures::stream::iter(docs.into_iter().map(Ok))))
        }
    }

    /// Loader that yields one document and one error — verifies
    /// partial-success contract.
    struct PartialLoader;

    #[async_trait]
    impl DocumentLoader for PartialLoader {
        fn name(&self) -> &'static str {
            "partial"
        }

        async fn load<'a>(
            &'a self,
            _ctx: &'a ExecutionContext,
        ) -> Result<crate::loader::DocumentStream<'a>> {
            let ok = Document::root("d1", "alpha", Source::now("test://", "test"), ns());
            let stream =
                futures::stream::iter(vec![Ok(ok), Err(Error::invalid_request("bad item"))]);
            Ok(Box::pin(stream))
        }
    }

    /// Deterministic fixed-dimension embedder that returns `[byte_count, 0, 0, …]`.
    struct StubEmbedder {
        dimension: usize,
    }

    #[async_trait]
    impl Embedder for StubEmbedder {
        fn dimension(&self) -> usize {
            self.dimension
        }

        async fn embed(&self, text: &str, _ctx: &ExecutionContext) -> Result<Embedding> {
            let mut v = vec![0.0_f32; self.dimension];
            #[allow(clippy::cast_precision_loss)]
            if let Some(first) = v.first_mut() {
                *first = text.len() as f32;
            }
            Ok(Embedding {
                vector: v,
                usage: Some(EmbeddingUsage::new(1)),
            })
        }
    }

    #[tokio::test]
    async fn empty_loader_produces_zero_indexed_clean_report() {
        let loader = StubLoader::new(vec![]);
        let pipeline = IngestionPipeline::builder(
            loader,
            RecursiveCharacterSplitter::new(),
            Arc::new(StubEmbedder { dimension: 4 }),
            Arc::new(InMemoryVectorStore::new(4)),
            ns(),
        )
        .build();
        let report = pipeline.run(&ExecutionContext::new()).await.unwrap();
        assert_eq!(report.documents_loaded, 0);
        assert_eq!(report.chunks_indexed, 0);
        assert_eq!(report.embedding_calls, 0);
        assert!(report.is_clean());
    }

    #[tokio::test]
    async fn single_document_flows_load_split_embed_store() {
        let doc = Document::root(
            "doc-1",
            "alpha\n\nbeta\n\ngamma",
            Source::now("test://doc-1", "test"),
            ns(),
        );
        let loader = StubLoader::new(vec![doc]);
        let store = Arc::new(InMemoryVectorStore::new(4));
        let pipeline = IngestionPipeline::builder(
            loader,
            RecursiveCharacterSplitter::new()
                .with_chunk_size(10)
                .with_chunk_overlap(0),
            Arc::new(StubEmbedder { dimension: 4 }),
            Arc::clone(&store),
            ns(),
        )
        .build();
        let report = pipeline.run(&ExecutionContext::new()).await.unwrap();
        assert_eq!(report.documents_loaded, 1);
        assert_eq!(report.embedding_calls, 1, "one batch per document");
        assert!(report.chunks_indexed >= 3);
        assert!(report.is_clean());

        // Round-trip through retrieval — store carries the chunks
        // and provenance landed under the reserved metadata key.
        let mut probe_vec = vec![0.0_f32; 4];
        probe_vec[0] = 5.0;
        let hits = store
            .search(&ExecutionContext::new(), &ns(), &probe_vec, 10)
            .await
            .unwrap();
        assert!(!hits.is_empty(), "store must contain the indexed chunks");
        let metadata = &hits[0].metadata;
        assert!(
            metadata.get(PROVENANCE_METADATA_KEY).is_some(),
            "provenance metadata stamped under reserved key"
        );
        let provenance = &metadata[PROVENANCE_METADATA_KEY];
        assert!(provenance.get("source").is_some());
        assert!(provenance.get("lineage").is_some());
        assert!(provenance.get("namespace").is_some());
    }

    #[tokio::test]
    async fn partial_loader_failure_recorded_but_run_completes() {
        let store = Arc::new(InMemoryVectorStore::new(4));
        let pipeline = IngestionPipeline::builder(
            PartialLoader,
            RecursiveCharacterSplitter::new(),
            Arc::new(StubEmbedder { dimension: 4 }),
            Arc::clone(&store),
            ns(),
        )
        .build();
        let report = pipeline.run(&ExecutionContext::new()).await.unwrap();
        assert_eq!(report.documents_loaded, 1, "successful item still indexed");
        assert_eq!(report.errors.len(), 1, "loader-side error recorded");
        assert_eq!(report.errors[0].stage, "load");
        assert!(!report.is_clean());
        assert!(report.chunks_indexed >= 1);
    }

    /// Chunker that drops every chunk except those at even
    /// indices — verifies the chunker chain runs and feeds the
    /// next stage.
    struct EvenIndexChunker;

    #[async_trait]
    impl Chunker for EvenIndexChunker {
        fn name(&self) -> &'static str {
            "even-only"
        }

        async fn process(
            &self,
            chunks: Vec<Document>,
            _ctx: &ExecutionContext,
        ) -> Result<Vec<Document>> {
            // Tag every kept chunk's lineage so the pipeline records
            // the chunker chain entry per design contract.
            let kept: Vec<Document> = chunks
                .into_iter()
                .enumerate()
                .filter(|(idx, _)| idx % 2 == 0)
                .map(|(_, mut chunk)| {
                    if let Some(lineage) = chunk.lineage.as_mut() {
                        lineage.push_chunker("even-only");
                    }
                    chunk
                })
                .collect();
            Ok(kept)
        }
    }

    #[tokio::test]
    async fn chunker_chain_filters_chunks_before_storage() {
        // The chunker drops every odd-indexed chunk; the embedder +
        // store therefore see strictly fewer chunks than the
        // splitter emitted. Provenance ("even-only") survives on
        // every retained chunk's lineage.
        let doc = Document::root(
            "doc",
            "alpha\n\nbeta\n\ngamma\n\ndelta\n\nepsilon",
            Source::now("test://", "test"),
            ns(),
        );
        let store = Arc::new(InMemoryVectorStore::new(4));
        let raw_chunks = RecursiveCharacterSplitter::new()
            .with_chunk_size(7)
            .with_chunk_overlap(0)
            .split(&doc);
        let raw_count = raw_chunks.len();
        assert!(
            raw_count >= 3,
            "splitter must produce at least three chunks for the test to be meaningful: got {raw_count}"
        );
        let pipeline = IngestionPipeline::builder(
            StubLoader::new(vec![doc]),
            RecursiveCharacterSplitter::new()
                .with_chunk_size(7)
                .with_chunk_overlap(0),
            Arc::new(StubEmbedder { dimension: 4 }),
            Arc::clone(&store),
            ns(),
        )
        .add_chunker(Arc::new(EvenIndexChunker))
        .build();
        let report = pipeline.run(&ExecutionContext::new()).await.unwrap();
        assert!(report.is_clean());
        let kept = raw_count.div_ceil(2); // even-index count
        let kept_u64 = kept as u64;
        assert_eq!(
            report.chunks_indexed, kept_u64,
            "chunker dropped odd indices; expected {kept} of {raw_count} indexed"
        );

        // Inspect the store — provenance carries the chunker chain.
        let mut probe = vec![0.0_f32; 4];
        probe[0] = 5.0;
        let hits = store
            .search(&ExecutionContext::new(), &ns(), &probe, 100)
            .await
            .unwrap();
        for hit in hits {
            let chain = &hit.metadata[PROVENANCE_METADATA_KEY]["lineage"]["chunker_chain"];
            let arr = chain.as_array().unwrap();
            assert_eq!(arr.len(), 1);
            assert_eq!(arr[0].as_str(), Some("even-only"));
        }
    }
}
