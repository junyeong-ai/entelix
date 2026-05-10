# entelix-rag

Algorithmic RAG primitives + the corrective-RAG (CRAG) recipe. Concrete source connectors and tokenizer-accurate splitters ship as companion crates.

## Surface

- **`Document`** + **`DocumentId`** / **`Lineage`** / **`Source`** — ingestion-side document with provenance (where it came from) and lineage (split / chunk ancestry). Distinct from the retrieval-side `entelix_memory::Document` (which carries similarity score).
- **`DocumentLoader` trait** — async source-side trait. `load(ctx) -> DocumentStream` so ingestion stays memory-bounded over arbitrarily large corpora.
- **`TextSplitter` trait** — sync algorithmic primitive. Concrete impls: `RecursiveCharacterSplitter` (char-budget), `TokenCountSplitter<C>` (token-budget, generic over `C: TokenCounter + ?Sized`), `MarkdownStructureSplitter` (heading-aware). All share the same recursive-merge core under `splitter::common`.
- **`Chunker` trait** + **`ContextualChunker`** — async transform over a chunk sequence. LLM-call capable (Anthropic Contextual Retrieval, HyDE, query decomposition).
- **`IngestionPipeline<L, S, E, V>`** — typed composition `DocumentLoader → TextSplitter → Chunker* → Embedder → VectorStore`. Generic params accept `?Sized` so concrete and `dyn` shapes interchange. Per-document partial-success contract: per-item failures land on `IngestReport::errors`; `run` only returns `Err` on structural failures.
- **Corrective-RAG recipe** — `build_corrective_rag_graph<Ret, G, R, M>` returns `CompiledGraph<CorrectiveRagState>`; `create_corrective_rag_agent` wraps that into the standard `Agent<CorrectiveRagState>` so lifecycle (sink fan-out, observer, supervisor handoff, audit emission) integrates uniformly. `RetrievalGrader` / `QueryRewriter` traits + LLM-driven reference impls (`LlmRetrievalGrader<M>` / `LlmQueryRewriter<M>`). `CORRECTIVE_RAG_AGENT_NAME` is the stable agent identifier on `AgentEvent` + OTel spans.

## Crate-local rules

- **Splitter algorithm core lives in `splitter::common`** — `recurse_with_metric` / `merge_with_overlap_metric` / `split_keeping_separator` are metric-agnostic. New splitters inject closures (`measure`, `take_tail`, `fallback`) and call into common; do not reimplement the recursion.
- **`chunk_size` is a soft cap** — overlap seeding deliberately allows the chunk that follows a flush to exceed `chunk_size` by up to `chunk_overlap` units before the next split point. Token-budget metrics additionally tolerate a small BPE seam-effect drift. Operators wanting strict adherence configure `chunk_overlap = 0`.
- **Shared trait-bound components are `Arc<T>` with `?Sized`** — `IngestionPipeline` embedder/store, `build_corrective_rag_graph` retriever, every `make_*_node` helper. Operators pass concrete `Arc<MyEmbedder>` for monomorphisation or `Arc<dyn Embedder>` for type-erasure interchangeably.
- **CRAG router is internal + state-based, no LLM call** — `route_after_grade` decides on the `Correct`-verdict fraction vs `CragConfig::min_correct_fraction`. The CRAG paper's three-way decision is intentionally collapsed to Correct-vs-not-Correct since the SDK ships no built-in web-search primitive; operators add web-search as a fallback inside their custom `Retriever`.
- **Recipe entry points return `Agent<S>`** — `create_corrective_rag_agent` returns `Result<Agent<CorrectiveRagState>>`, mirroring `create_react_agent` / `create_chat_agent` / `create_supervisor_agent`. Custom sink/observer/approver wiring goes through `build_corrective_rag_graph` + manual `Agent::builder()`.

## Forbidden

- New splitter that reimplements the recursive walk instead of calling into `splitter::common`.
- Default-injecting a token count when the wired `TokenCounter` returns an error (silent fallback — invariant 15).
- Synthesising a `ToolPair` outside `entelix_session::CompactedHistory::group(events)` if a chunker manipulates session events (invariant 21).
- Source connectors landing in this crate. They go in `entelix-rag-<connector>` companion crates.

## References

- Tokenizer companions: `entelix-tokenizer-tiktoken` (OpenAI BPE), `entelix-tokenizer-hf` (HuggingFace tokenizer.json).
