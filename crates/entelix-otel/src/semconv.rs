//! `gen_ai.*` attribute names — snapshot of the OpenTelemetry
//! GenAI semantic conventions (0.31). Tracking the registry rather
//! than the experimental detail catalogue keeps this layer stable
//! across minor convention updates.
//!
//! Reference: <https://opentelemetry.io/docs/specs/semconv/gen-ai/>
//!
//! Constants are duplicated rather than re-exported from
//! `opentelemetry_semantic_conventions::trace::*` because the
//! upstream crate moves namespaces between minor versions; pinning
//! the names here decouples entelix's contract from the upstream
//! API churn.

// ── Operation ────────────────────────────────────────────────────

/// Provider system name. Common values: `"anthropic"`, `"openai"`,
/// `"gemini"`, `"bedrock"`, `"vertex"`, `"foundry"`.
pub const SYSTEM: &str = "gen_ai.system";

/// High-level operation. Values: `"chat"`, `"text_completion"`,
/// `"embeddings"`, `"execute_tool"`.
pub const OPERATION_NAME: &str = "gen_ai.operation.name";

// ── Request ──────────────────────────────────────────────────────

/// Model name as the caller specified it (pre-alias resolution).
pub const REQUEST_MODEL: &str = "gen_ai.request.model";

/// Sampling temperature.
pub const REQUEST_TEMPERATURE: &str = "gen_ai.request.temperature";

/// Top-p (nucleus sampling).
pub const REQUEST_TOP_P: &str = "gen_ai.request.top_p";

/// Top-k.
pub const REQUEST_TOP_K: &str = "gen_ai.request.top_k";

/// Maximum tokens to generate.
pub const REQUEST_MAX_TOKENS: &str = "gen_ai.request.max_tokens";

/// Stop sequences (string array).
pub const REQUEST_STOP_SEQUENCES: &str = "gen_ai.request.stop_sequences";

/// Frequency penalty.
pub const REQUEST_FREQUENCY_PENALTY: &str = "gen_ai.request.frequency_penalty";

/// Presence penalty.
pub const REQUEST_PRESENCE_PENALTY: &str = "gen_ai.request.presence_penalty";

// ── Response ─────────────────────────────────────────────────────

/// Vendor-assigned response ID.
pub const RESPONSE_ID: &str = "gen_ai.response.id";

/// Model name the vendor reported (post-alias resolution).
pub const RESPONSE_MODEL: &str = "gen_ai.response.model";

/// Reason the model halted (`"end_turn"`, `"max_tokens"`, …).
pub const RESPONSE_FINISH_REASONS: &str = "gen_ai.response.finish_reasons";

// ── Usage ────────────────────────────────────────────────────────

/// Prompt tokens consumed.
pub const USAGE_INPUT_TOKENS: &str = "gen_ai.usage.input_tokens";

/// Completion tokens produced.
pub const USAGE_OUTPUT_TOKENS: &str = "gen_ai.usage.output_tokens";

/// Tokens served from the prompt cache (Anthropic
/// `cache_read_input_tokens`, OpenAI `cached_tokens`, Bedrock
/// `cacheReadInputTokens`). Vendors typically discount these
/// heavily; the metric lets operators measure cache hit rate
/// directly.
pub const USAGE_CACHED_INPUT_TOKENS: &str = "gen_ai.usage.cached_input_tokens";

/// Tokens written to the prompt cache (Anthropic
/// `cache_creation_input_tokens`, Bedrock
/// `cacheWriteInputTokens`). Vendors typically charge a
/// premium for these; the metric lets operators amortise the
/// premium against subsequent cache reads.
pub const USAGE_CACHE_CREATION_INPUT_TOKENS: &str = "gen_ai.usage.cache_creation_input_tokens";

/// Tokens spent on internal reasoning (Anthropic thinking,
/// OpenAI o-series reasoning, Gemini thinking budget). Often
/// billed at the output rate but not surfaced in the
/// assistant text — separate metric so operators can isolate
/// reasoning cost from visible completion cost.
pub const USAGE_REASONING_TOKENS: &str = "gen_ai.usage.reasoning_tokens";

// ── Token type tag values (for `gen_ai.token.type` on the metric) ──

/// `gen_ai.token.type` attribute key.
pub const TOKEN_TYPE: &str = "gen_ai.token.type";

/// `gen_ai.token.type` value for prompt input tokens.
pub const TOKEN_TYPE_INPUT: &str = "input";

/// `gen_ai.token.type` value for completion output tokens.
pub const TOKEN_TYPE_OUTPUT: &str = "output";

/// `gen_ai.token.type` value for cache-read input tokens.
/// Aligned with the OpenTelemetry GenAI semconv 0.32 addition.
pub const TOKEN_TYPE_CACHED: &str = "cached";

/// `gen_ai.token.type` value for cache-write input tokens.
/// Entelix-specific extension — semconv has no standard yet
/// for cache writes; operators filter on this string to
/// isolate the premium spend.
pub const TOKEN_TYPE_CACHE_CREATION: &str = "cache_creation";

/// `gen_ai.token.type` value for internal-reasoning tokens.
/// Entelix-specific extension — semconv has no standard yet
/// for reasoning budgets; operators filter on this string to
/// isolate reasoning spend from visible output spend.
pub const TOKEN_TYPE_REASONING: &str = "reasoning";

// ── Tools ────────────────────────────────────────────────────────

/// Tool name (set on the tool-invocation span entry by a
/// `tower::Layer<Service<ToolInvocation>>` adapter).
pub const TOOL_NAME: &str = "gen_ai.tool.name";

/// Vendor-assigned tool-call ID, if any.
pub const TOOL_CALL_ID: &str = "gen_ai.tool.call.id";

// ── Tenant (entelix extension) ───────────────────────────────────

/// `tenant_id` carried by [`entelix_core::context::ExecutionContext`].
/// **Not** part of the upstream semconv — entelix-specific extension
/// (invariant 11). Sits in the `entelix.*` namespace to avoid
/// collision with future official additions.
pub const TENANT_ID: &str = "entelix.tenant_id";

/// `thread_id` carried by [`entelix_core::context::ExecutionContext`].
/// Also entelix-specific.
pub const THREAD_ID: &str = "entelix.thread_id";

/// `run_id` carried by [`entelix_core::context::ExecutionContext`] —
/// the per-execute correlation key the agent runtime stamps on every
/// `AgentEvent` and OTel span. Also entelix-specific.
pub const RUN_ID: &str = "entelix.run_id";

// ── Metric instrument names ──────────────────────────────────────

/// Per-call token-usage histogram.
pub const METRIC_TOKEN_USAGE: &str = "gen_ai.client.token.usage";

/// Per-call latency histogram (seconds).
pub const METRIC_OPERATION_DURATION: &str = "gen_ai.client.operation.duration";
