//! # entelix
//!
//! Web-service-native Rust agentic AI SDK — `LangChain` + `LangGraph` parity,
//! Anthropic managed-agent shape, production-grade observability. This crate
//! is the **facade**: it re-exports the sub-crates behind feature flags
//! (see `Cargo.toml`).
//!
//! Surface includes the composition spine (`Runnable`, `.pipe()`), prompt
//! primitives (`PromptTemplate`, `ChatPromptTemplate`,
//! `MessagesPlaceholder`), the IR (`ir`), every codec
//! ([`codecs::AnthropicMessagesCodec`] and siblings), transports
//! (`DirectTransport` plus optional cloud transports under feature flags),
//! the `ChatModel` bundle ([`ChatModel`]), the `Tool` hand contract, and
//! optional sub-crates for memory, persistence, MCP, policy,
//! observability, the HTTP server, and agent recipes.
//!
//! Architectural canon: see `CLAUDE.md` (22 invariants + naming taxonomy)
//! and `docs/architecture/principles.md`.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_root_url = "https://docs.rs/entelix/0.4.1")]
#![deny(missing_docs)]

mod tokenizer;

pub use tokenizer::default_token_counter_registry;

// ── Module re-exports — every type's canonical facade path lives
// inside the entelix-core submodule it was authored in. Reach a
// codec via `entelix::codecs::AnthropicMessagesCodec`, an
// auth provider via `entelix::auth::ApiKeyProvider`, a tool trait
// via `entelix::tools::Tool`, and so on. One path per type.
pub use entelix_core::auth;
pub use entelix_core::cancellation;
pub use entelix_core::codecs;
pub use entelix_core::events;
pub use entelix_core::ir;
pub use entelix_core::sandbox;
pub use entelix_core::service;
pub use entelix_core::skills;
pub use entelix_core::stream;
pub use entelix_core::tools;
pub use entelix_core::transports;

// ── Headline top-level types — those without a natural facade
// module home. Every user touches Error / Result / ExecutionContext
// on every call; ChatModel is the 5-line agent path; ThreadKey is
// the persistence addressing primitive. AuditSink + Extensions +
// LlmRenderable + ProviderErrorKind sit alongside because every
// consumer lifecycle (audit emission, scoped task-locals, error
// matching, LLM-facing error rendering) reaches them. Keep them at
// the top so callers don't need to memorise an internal module name.
pub use entelix_core::{
    AgentContext, ApprovalDecision, AuditSink, AuditSinkHandle, BudgetCostEstimator,
    ByteCountTokenCounter, ChatModel, ChatModelConfig, Clock, CostCalculator, DEFAULT_TENANT_ID,
    Error, ErrorClass, ExecutionContext, Extensions, InterruptionKind, InterruptionPhase,
    LlmFacingSchema, LlmRenderable, OutputValidator, PendingApprovalDecisions, ProviderErrorKind,
    RenderedForLlm, RequestOverrides, Result, RunBudget, RunOverrides, SystemClock, TenantId,
    ThreadKey, TokenCounter, TokenCounterRegistry, TokenCounterResolution, ToolCostCalculator,
    TypedModelStream, UsageLimitBreach, UsageSnapshot, install_default_tls, interrupt,
    interrupt_with,
};

// ── Sub-crate re-exports — the 90% surface for crates that don't
// share entelix-core's module structure. One canonical flat path
// per type.
pub use entelix_agents::{
    Agent, AgentBuilder, AgentEntry, AgentEvent, AgentEventSink, AgentObserver, AgentRunResult,
    AlwaysApprove, ApprovalLayer, ApprovalRequest, ApprovalService, Approver, BroadcastSink,
    CaptureSink, ChannelApprover, ChannelApproverConfig, ChannelSink, ChatState,
    DEFAULT_SUMMARY_KEEP_RECENT_TURNS, DEFAULT_SUMMARY_SYSTEM_PROMPT, DroppingSink, DynObserver,
    EffectGate, ExecutionMode, FailOpenSink, FanOutSink, MessageRunnableCompactionExt,
    PendingApproval, ReActAgentBuilder, ReActState, RunnableCompacting,
    RunnableToSummarizerAdapter, Subagent, SubagentBuilder, SubagentMetadata, SubagentTool,
    SummaryCompactor, SupervisorDecision, SupervisorState, ToolApprovalEventSink,
    ToolApprovalEventSinkHandle, ToolEventLayer, ToolEventService, ToolHook, ToolHookDecision,
    ToolHookLayer, ToolHookRegistry, ToolHookRequest, ToolHookService, build_chat_graph,
    build_react_graph, build_supervisor_graph, create_chat_agent, create_react_agent,
    create_supervisor_agent, team_from_supervisor,
};
pub use entelix_cloud::CloudError;
#[cfg(feature = "aws")]
#[cfg_attr(docsrs, doc(cfg(feature = "aws")))]
pub use entelix_cloud::bedrock::{
    BedrockAuth, BedrockCredentialProvider, BedrockSigner, BedrockTransport,
    BedrockTransportBuilder,
};
#[cfg(feature = "azure")]
#[cfg_attr(docsrs, doc(cfg(feature = "azure")))]
pub use entelix_cloud::foundry::{
    FOUNDRY_SCOPE, FoundryAuth, FoundryCredentialProvider, FoundryTransport,
    FoundryTransportBuilder,
};
pub use entelix_cloud::refresh::{
    CachedTokenProvider, DEFAULT_REFRESH_BUFFER as CLOUD_DEFAULT_REFRESH_BUFFER, TokenRefresher,
    TokenSnapshot,
};
#[cfg(feature = "gcp")]
#[cfg_attr(docsrs, doc(cfg(feature = "gcp")))]
pub use entelix_cloud::vertex::{
    VERTEX_SCOPE, VertexCredentialProvider, VertexTransport, VertexTransportBuilder,
};
pub use entelix_graph::{
    Annotated, Append, Checkpoint, CheckpointGranularity, CheckpointId, Checkpointer, Command,
    CompiledGraph, ConditionalEdge, ContributingNodeAdapter, DEFAULT_RECURSION_LIMIT, END,
    EdgeSelector, InMemoryCheckpointer, Max, MergeMap, MergeNodeAdapter, Reducer, Replace,
    SendEdge, SendMerger, SendSelector, StateGraph, StateMerge,
};
#[cfg(feature = "graphmemory-pg")]
#[cfg_attr(docsrs, doc(cfg(feature = "graphmemory-pg")))]
pub use entelix_graphmemory_pg::{
    PgGraphMemory, PgGraphMemoryBuilder, PgGraphMemoryError, PgGraphMemoryResult,
};
#[cfg(feature = "mcp-chatmodel")]
#[cfg_attr(docsrs, doc(cfg(feature = "mcp-chatmodel")))]
pub use entelix_mcp::ChatModelSamplingProvider;
#[cfg(feature = "mcp")]
#[cfg_attr(docsrs, doc(cfg(feature = "mcp")))]
pub use entelix_mcp::{
    DEFAULT_IDLE_TTL as MCP_DEFAULT_IDLE_TTL,
    DEFAULT_LISTENER_CONCURRENCY as MCP_DEFAULT_LISTENER_CONCURRENCY,
    DEFAULT_MAX_FRAME_BYTES as MCP_DEFAULT_MAX_FRAME_BYTES, DEFAULT_TIMEOUT as MCP_DEFAULT_TIMEOUT,
    ElicitationProvider, ElicitationRequest, ElicitationResponse, HttpMcpClient, IncludeContext,
    McpClient, McpClientState, McpCompletionArgument, McpCompletionReference, McpCompletionResult,
    McpError, McpManager, McpManagerBuilder, McpPrompt, McpPromptArgument, McpPromptContent,
    McpPromptInvocation, McpPromptMessage, McpPromptResourceRef, McpResource, McpResourceContent,
    McpResult, McpRoot, McpServerConfig, McpToolAdapter, McpToolDefinition, ModelHint,
    ModelPreferences, PROTOCOL_VERSION as MCP_PROTOCOL_VERSION, RequestDecorator,
    ResourceBoundKind, RootsProvider, SamplingContent, SamplingMessage, SamplingProvider,
    SamplingRequest, SamplingResponse, StaticElicitationProvider, StaticRootsProvider,
    StaticSamplingProvider, qualified_name, validate_server_name,
};
pub use entelix_memory::{
    BufferMemory, ConsolidatingBufferMemory, ConsolidationContext, ConsolidationPolicy,
    CostCalculatorAdapter, Direction, Document as RetrievedDocument,
    DocumentId as RetrievedDocumentId, EdgeId, Embedder, Embedding, EmbeddingCostCalculator,
    EmbeddingRetriever, EmbeddingUsage, EntityMemory, EntityRecord, Episode, EpisodeId,
    EpisodicMemory, GraphHop, GraphMemory, IdentityReranker, InMemoryGraphMemory, InMemoryStore,
    InMemoryVectorStore, MeteredEmbedder, MmrReranker, Namespace, NamespacePrefix,
    NeverConsolidate, NodeId, OnMessageCount, OnTokenBudget, PolicyExtras, PutOptions,
    RerankedDocument, Reranker, RetrievalQuery, Retriever, SemanticMemory, SemanticMemoryBackend,
    Store, Summarizer, SummaryMemory, VectorFilter, VectorStore, first_non_finite_vector_value,
    validate_vector_shape,
};
#[cfg(feature = "embedders-openai")]
#[cfg_attr(docsrs, doc(cfg(feature = "embedders-openai")))]
pub use entelix_memory_openai::{
    DEFAULT_BASE_URL as OPENAI_EMBEDDINGS_BASE_URL, OpenAiEmbedder, OpenAiEmbedderBuilder,
    OpenAiEmbedderError, OpenAiEmbedderResult, TEXT_EMBEDDING_3_LARGE,
    TEXT_EMBEDDING_3_LARGE_DIMENSION, TEXT_EMBEDDING_3_SMALL, TEXT_EMBEDDING_3_SMALL_DIMENSION,
};
#[cfg(feature = "vectorstores-pgvector")]
#[cfg_attr(docsrs, doc(cfg(feature = "vectorstores-pgvector")))]
pub use entelix_memory_pgvector::{
    DistanceMetric as PgVectorDistanceMetric, IndexKind as PgVectorIndexKind, PgVectorStore,
    PgVectorStoreBuilder, PgVectorStoreError, PgVectorStoreResult,
};
#[cfg(feature = "vectorstores-qdrant")]
#[cfg_attr(docsrs, doc(cfg(feature = "vectorstores-qdrant")))]
pub use entelix_memory_qdrant::{
    CONTENT_KEY as QDRANT_CONTENT_KEY, DOC_ID_KEY as QDRANT_DOC_ID_KEY,
    DistanceMetric as QdrantDistanceMetric, METADATA_KEY as QDRANT_METADATA_KEY,
    NAMESPACE_KEY as QDRANT_NAMESPACE_KEY, QdrantStoreError, QdrantStoreResult, QdrantVectorStore,
    QdrantVectorStoreBuilder,
};
#[cfg(feature = "otel")]
#[cfg_attr(docsrs, doc(cfg(feature = "otel")))]
pub use entelix_otel::{
    DEFAULT_TOOL_IO_TRUNCATION, GenAiMetrics, OperationKind, OtelLayer, OtelService,
    ToolIoCaptureMode, TraceContextTransport, trace_context_injector,
};
#[cfg(feature = "postgres")]
#[cfg_attr(docsrs, doc(cfg(feature = "postgres")))]
pub use entelix_persistence::postgres::{
    PostgresCheckpointer, PostgresLock, PostgresPersistence, PostgresPersistenceBuilder,
    PostgresSessionLog, PostgresStore,
};
#[cfg(feature = "redis")]
#[cfg_attr(docsrs, doc(cfg(feature = "redis")))]
pub use entelix_persistence::redis::{
    RedisCheckpointer, RedisLock, RedisPersistence, RedisPersistenceBuilder, RedisSessionLog,
    RedisStore,
};
#[cfg(any(feature = "postgres", feature = "redis"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "postgres", feature = "redis"))))]
pub use entelix_persistence::{
    AdvisoryKey, DistributedLock, LockGuard, PersistenceError, PersistenceResult,
    SessionSchemaVersion, with_session_lock,
};
#[cfg(feature = "policy")]
#[cfg_attr(docsrs, doc(cfg(feature = "policy")))]
pub use entelix_policy::{
    Budget, CostMeter, DEFAULT_MAX_TENANTS as POLICY_DEFAULT_MAX_TENANTS,
    MAX_WARNED_MODELS as POLICY_MAX_WARNED_MODELS, ModelPricing, PiiPattern, PiiRedactor,
    PolicyError, PolicyLayer, PolicyRegistry, PolicyResult, PolicyService, PricingTable,
    QuotaLimiter, RateLimiter, RegexRedactor, TenantPolicy, TenantPolicyBuilder,
    TokenBucketLimiter, UnknownModelPolicy, default_pii_patterns, luhn_valid,
};
pub use entelix_prompt::{
    ChatFewShotPromptTemplate, ChatPromptPart, ChatPromptTemplate, Example, ExampleSelector,
    FewShotPromptTemplate, FixedExampleSelector, LengthBasedExampleSelector, MessagesPlaceholder,
    PromptTemplate, PromptValue, PromptVars, SharedExampleSelector,
};
pub use entelix_rag::{
    CONTEXTUAL_CHUNKER_DEFAULT_INSTRUCTION, CORRECTIVE_RAG_AGENT_NAME, Chunker, ContextualChunker,
    ContextualChunkerBuilder, CorrectiveRagState, CragConfig, DEFAULT_CHUNK_OVERLAP_CHARS,
    DEFAULT_CHUNK_OVERLAP_TOKENS, DEFAULT_CHUNK_SIZE_CHARS, DEFAULT_CHUNK_SIZE_TOKENS,
    DEFAULT_GENERATOR_SYSTEM_PROMPT, DEFAULT_GRADER_INSTRUCTION, DEFAULT_MARKDOWN_HEADING_LEVELS,
    DEFAULT_MAX_REWRITE_ATTEMPTS, DEFAULT_MIN_CORRECT_FRACTION, DEFAULT_RECURSIVE_SEPARATORS,
    DEFAULT_RETRIEVAL_TOP_K, DEFAULT_REWRITER_INSTRUCTION, Document, DocumentId, DocumentLoader,
    DocumentStream, FailurePolicy, GradeVerdict, IngestError, IngestReport, IngestionPipeline,
    IngestionPipelineBuilder, Lineage, LlmQueryRewriter, LlmQueryRewriterBuilder,
    LlmRetrievalGrader, LlmRetrievalGraderBuilder, MarkdownStructureSplitter,
    PROVENANCE_METADATA_KEY, QueryRewriter, RecursiveCharacterSplitter, RetrievalGrader, Source,
    TextSplitter, TokenCountSplitter, build_corrective_rag_graph, create_corrective_rag_agent,
};
pub use entelix_runnable::{
    AnyRunnable, AnyRunnableHandle, BoxStream, ChatModelExt, Configured, DebugEvent, Fallback,
    JsonOutputParser, Mapping, Retrying, Runnable, RunnableEvent, RunnableExt, RunnableLambda,
    RunnableParallel, RunnablePassthrough, RunnableRouter, RunnableSequence, StreamChunk,
    StreamMode, StructuredOutputAdapter, Timed, ToolToRunnableAdapter, erase,
};
#[cfg(feature = "server")]
#[cfg_attr(docsrs, doc(cfg(feature = "server")))]
pub use entelix_server::{
    AgentRouterBuilder, BuildError as ServerBuildError, BuildResult as ServerBuildResult,
    DEFAULT_TENANT_HEADER as SERVER_DEFAULT_TENANT_HEADER, ServerError, ServerResult, TenantMode,
};
pub use entelix_session::{
    CompactedHistory, Compactor, GraphEvent, HeadDropCompactor, InMemorySessionLog,
    SessionAuditSink, SessionGraph, SessionLog, ToolPair, Turn, messages_char_size,
    messages_to_events,
};
#[cfg(feature = "tokenizer-hf")]
#[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-hf")))]
pub use entelix_tokenizer_hf::{HfTokenCounter, HfTokenizerError};
#[cfg(feature = "tokenizer-tiktoken")]
#[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-tiktoken")))]
pub use entelix_tokenizer_tiktoken::{TiktokenCounter, TiktokenEncoding, TiktokenError};
pub use entelix_tools::{
    ActivateSkillTool, Calculator, CalculatorInput, CalculatorOutput,
    DEFAULT_FETCH_TIMEOUT as HTTP_FETCH_DEFAULT_TIMEOUT,
    DEFAULT_MAX_REDIRECTS as HTTP_FETCH_DEFAULT_MAX_REDIRECTS,
    DEFAULT_MAX_RESPONSE_BYTES as HTTP_FETCH_DEFAULT_MAX_RESPONSE_BYTES,
    DEFAULT_MAX_RESULTS as SEARCH_DEFAULT_MAX_RESULTS, HostAllowlist, HostRule, HttpFetchTool,
    HttpFetchToolBuilder, InMemorySkill, InMemorySkillBuilder, ListSkillsTool,
    ReadSkillResourceTool, SchemaTool, SchemaToolAdapter, SchemaToolExt, SearchProvider,
    SearchResult, SearchTool, SsrfSafeDnsResolver, StaticResource, ToolError, ToolResult,
    is_ssrf_blocked, tool,
};
/// Common imports for typical SDK usage.
///
/// ```ignore
/// use entelix::prelude::*;
/// ```
pub mod prelude {
    pub use entelix_core::ir::{ContentPart, Message, Role};
    pub use entelix_core::{AgentContext, ChatModel, Error, ExecutionContext, Result};
    pub use entelix_prompt::{ChatPromptPart, ChatPromptTemplate, PromptValue, PromptVars};
    pub use entelix_rag::{
        CONTEXTUAL_CHUNKER_DEFAULT_INSTRUCTION, CORRECTIVE_RAG_AGENT_NAME, Chunker,
        ContextualChunker, ContextualChunkerBuilder, CorrectiveRagState, CragConfig,
        DEFAULT_CHUNK_OVERLAP_CHARS, DEFAULT_CHUNK_OVERLAP_TOKENS, DEFAULT_CHUNK_SIZE_CHARS,
        DEFAULT_CHUNK_SIZE_TOKENS, DEFAULT_GENERATOR_SYSTEM_PROMPT, DEFAULT_GRADER_INSTRUCTION,
        DEFAULT_MARKDOWN_HEADING_LEVELS, DEFAULT_MAX_REWRITE_ATTEMPTS,
        DEFAULT_MIN_CORRECT_FRACTION, DEFAULT_RECURSIVE_SEPARATORS, DEFAULT_RETRIEVAL_TOP_K,
        DEFAULT_REWRITER_INSTRUCTION, Document, DocumentId, DocumentLoader, DocumentStream,
        FailurePolicy, GradeVerdict, IngestError, IngestReport, IngestionPipeline,
        IngestionPipelineBuilder, Lineage, LlmQueryRewriter, LlmQueryRewriterBuilder,
        LlmRetrievalGrader, LlmRetrievalGraderBuilder, MarkdownStructureSplitter,
        PROVENANCE_METADATA_KEY, QueryRewriter, RecursiveCharacterSplitter, RetrievalGrader,
        Source, TextSplitter, TokenCountSplitter, build_corrective_rag_graph,
        create_corrective_rag_agent,
    };
    pub use entelix_runnable::{JsonOutputParser, Runnable, RunnableExt};
}
