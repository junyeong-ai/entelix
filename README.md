# entelix

[![Rust 1.95](https://img.shields.io/badge/rust-1.95.0%2B-orange?style=flat-square&logo=rust)](https://www.rust-lang.org)
[![Edition 2024](https://img.shields.io/badge/edition-2024-blueviolet?style=flat-square)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

> **English** | **[한국어](README.ko.md)**

> **Where every agent realizes its purpose.**

**General-purpose agentic-AI SDK in Rust.** Build any agent that runs an LLM-driven control loop with tools, memory, and durable state. Multi-tenant from day 1. OpenTelemetry GenAI semconv-native. Architectural invariants enforced by CI. Anthropic *managed-agents* shape. LangChain + LangGraph parity.

---

## Why entelix?

- **Multi-tenant first** — `TenantId` at every persistence boundary; Postgres `FORCE ROW LEVEL SECURITY` is defense-in-depth on every backend; cross-tenant data leakage is structurally impossible by API design.
- **Type-enforced architecture** — sealed `RenderedForLlm<T>` carrier (compile-time LLM / operator channel separation); sealed `CompactedHistory` (`tool_call` / `tool_result` pair invariant impossible to break); `TenantId` newtype with validating constructor.
- **Production observability from day 1** — OpenTelemetry GenAI semconv `gen_ai.*` attributes; cost emits transactionally (`Ok` branch only); typed `AuditSink` channel with six `record_*` verbs (`sub_agent_invoked` / `agent_handoff` / `resumed` / `memory_recall` / `usage_limit_exceeded` / `context_compacted`).
- **Five codecs × four transports** — Anthropic Messages, OpenAI Chat, OpenAI Responses, Gemini, Bedrock Converse over Direct (any HTTPS), Bedrock (SigV4), Vertex (gcp_auth), Foundry (api-key + AAD). Capability honesty: `LossyEncode` warnings on every coerced field, `Other { raw }` for unknown vendor signals — no silent fallback.
- **MCP first-class** — all three server-initiated channels (`Roots`, `Elicitation`, `Sampling`); per-tenant `(tenant_id, server_name)` pool isolation; HTTP-only by design.
- **CI-enforced invariants** — `cargo xtask invariants` runs typed-AST visitors per push for filesystem-free, naming taxonomy, silent-fallback, lock-ordering, public-API drift, supply-chain, feature-matrix gates.

---

## Quick Start

```bash
cargo add entelix
```

```rust
use std::sync::Arc;
use entelix::auth::ApiKeyProvider;
use entelix::codecs::AnthropicMessagesCodec;
use entelix::ir::Message;
use entelix::transports::DirectTransport;
use entelix::{ChatModel, ExecutionContext};

#[tokio::main]
async fn main() -> entelix::Result<()> {
    entelix::install_default_tls();
    let creds = Arc::new(ApiKeyProvider::anthropic(std::env::var("ANTHROPIC_API_KEY")?));
    let transport = DirectTransport::anthropic(creds)?;
    let model = ChatModel::new(AnthropicMessagesCodec, transport, "claude-opus-4-7")
        .with_system("Answer in one sentence.");

    let reply = model
        .complete(vec![Message::user("Define entelechy.")], &ExecutionContext::new())
        .await?;
    println!("{reply:?}");
    Ok(())
}
```

---

## Key Features

### LLM call — five codecs through one IR

```rust
// Same `ModelRequest` IR routes through every codec.
// `LossyEncode` warnings expose every coerced field.
let model = ChatModel::new(AnthropicMessagesCodec, transport, "claude-opus-4-7")
    .with_validation_retries(2);                     // schema-mismatch retry budget

let reply: Translation = model
    .complete_typed::<Translation>(messages, &ctx)   // typed structured output
    .await?;
```

### LCEL composition

```rust
use entelix::{ChatPromptTemplate, JsonOutputParser, RunnableExt};

let chain = prompt.pipe(model).pipe(parser);       // compile-time I/O checking
let result = chain.invoke(input, &ctx).await?;
```

### StateGraph control flow (LangGraph parity)

```rust
use entelix::{Annotated, Append, Max, StateGraph, StateMerge, RunnableLambda};

#[derive(Clone, Default, StateMerge)]
struct AgentState {
    log: Annotated<Vec<String>, Append<String>>,    // accumulated across nodes
    score: Annotated<i32, Max<i32>>,                // best-of across branches
    last_phase: String,                             // last-write-wins
}

let graph = StateGraph::<AgentState>::new()
    .add_contributing_node("plan", plan)            // typed delta + per-field merge
    .add_send_edges("plan", ["a", "b", "c"], scatter, "score")  // parallel fan-out
    .add_conditional_edges("score", router, [("plan", "plan"), ("done", entelix::END)])
    .set_entry_point("plan")
    .with_checkpointer(Arc::new(postgres_checkpointer))         // resume after crash
    .compile()?;
```

### Auto-compaction (Claude Agent SDK parity)

```rust
use entelix::{HeadDropCompactor, MessageRunnableCompactionExt, SummaryCompactor};

// Drop oldest turns when context approaches threshold.
let model = my_model.with_compaction(Arc::new(HeadDropCompactor), 8_192);

// Or LLM-summary compaction (LangChain SummarizationMiddleware parity).
let summariser = SummaryCompactor::new(Arc::new(summary_model));
let model = my_model.with_compaction(Arc::new(summariser), 8_192);
```

### Production HTTP server

```rust
use entelix::{AgentRouterBuilder, SERVER_DEFAULT_TENANT_HEADER};

let router = AgentRouterBuilder::new(agent)
    .with_checkpointer(Arc::clone(&postgres_checkpointer))
    .with_tenant_header(SERVER_DEFAULT_TENANT_HEADER)   // multi-tenant strict mode
    .build()?;

let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
axum::serve(listener, router).await?;
```

Endpoints: `POST /v1/threads/{thread_id}/runs`, `GET /v1/threads/{thread_id}/stream`, `POST /v1/threads/{thread_id}/wake`, `GET /v1/health`.

### Tools — built-in + `#[tool]` macro

```rust
use entelix::tool;

#[tool]
/// Add two numbers.                              // first paragraph → LLM-facing description
async fn add(_ctx: &AgentContext<()>, a: i64, b: i64) -> Result<i64> {
    Ok(a + b)
}
// Generates `Add` unit struct + `SchemaTool` impl + JSON Schema.
```

Built-ins: `HttpFetchTool` (3-layer SSRF defense), `Calculator`, `SchemaTool` typed-I/O adapter. Sandboxed shell / file / code / list-dir tools live in the `entelix-tools-coding` companion crate and delegate syscalls through the `Sandbox` trait.

### Memory patterns

`BufferMemory` (sliding window), `SummaryMemory` (rolling LLM summary), `ConsolidatingBufferMemory` (auto-summary on size threshold), `EntityMemory` (entity-keyed facts), `EpisodicMemory<V>` (time-ordered episodes), `SemanticMemory<E, V>` (vector retrieval), `GraphMemory<N, E>` (typed nodes + timestamped edges, BFS traversal + shortest-path).

### MCP — all three server-initiated channels

```rust
use entelix::{ChatModelSamplingProvider, McpServerConfig, StaticRootsProvider};

let server = McpServerConfig::http("https://server.example/mcp")
    .with_roots_provider(Arc::new(StaticRootsProvider::new(roots)))
    .with_sampling_provider(Arc::new(ChatModelSamplingProvider::new(chat)));
```

---

## How entelix differs

| Feature | LangChain | LangGraph | Claude Agent SDK | pydantic-ai | **entelix** |
|---|---|---|---|---|---|
| Language | Python | Python | TypeScript | Python | **Rust** |
| Multi-tenant first-class | ✗ | △ | ✗ | ✗ | **✓** |
| Postgres FORCE RLS | ✗ | ✗ | ✗ | ✗ | **✓** |
| Graph memory (typed nodes/edges + BFS) | ✗ | ✗ | ✗ | ✗ | **✓** |
| Auto-compaction (drop + LLM-summary) | △ | △ | ✓ | ✗ | **✓** |
| Typed structured output + retry budget | ✓ | ✓ | △ | ✓ | **✓** |
| Sealed LLM/operator channel carrier | ✗ | ✗ | ✗ | ✗ | **✓** |
| OTel GenAI semconv native | △ | △ | △ | ✓ | **✓** |
| Typed audit sink (`record_*` verbs) | ✗ | ✗ | △ | ✗ | **✓** |
| Run budget axes (incl. USD cost) | ✗ | △ | ✗ | ✓ (5-axis, no USD) | **✓ (6-axis incl. USD)** |
| Vendor-specific extension escape-hatch (`*Ext`) | △ | △ | ✗ | ✓ | **✓** |
| Distributed session lock (Postgres + Redis) | ✗ | ✗ | ✗ | ✗ | **✓** |
| MCP all 3 server-initiated channels | △ | △ | ✓ | △ | **✓** |
| CI-enforced architectural invariants | ✗ | ✗ | ✗ | ✗ | **✓** |

---

## What entelix is NOT

- **No direct filesystem / shell calls in first-party crates** — sandboxed wrappers (`SandboxedShellTool`, `SandboxedReadFileTool`, …) ship in the `entelix-tools-coding` companion crate and every syscall delegates through the `Sandbox` trait. Concrete `Sandbox` impls (Landlock / Seatbelt / e2b / modal) live as companion crates, not in core.
- **No local inference** — application-layer SDK; pair with `candle` / `mistral.rs` if you need it.
- **No vector DB reimplementation** — production `VectorStore` impls ship as `entelix-memory-qdrant` / `entelix-memory-pgvector`; bring your own via the trait.
- **No document loaders** — that's `swiftide`'s job.
- **No Python interop** — Rust-first.

---

## Workspace

```
entelix                  — facade (re-exports gated by feature flags)
entelix-core             — IR, Codec, Transport, Tool, Auth, ChatModel + tower::Service spine
entelix-runnable         — Runnable trait + LCEL .pipe() + Sequence/Parallel/Router/Lambda
entelix-prompt           — PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, FewShot
entelix-graph            — StateGraph, Reducer, StateMerge trait, Dispatch, Checkpointer, interrupts
entelix-graph-derive     — proc-macro: #[derive(StateMerge)] emits Contribution + builders + impl
entelix-tool-derive      — proc-macro: #[tool] generates SchemaTool impl from an async fn signature
entelix-session          — SessionGraph event log + Compactor + SessionAuditSink, fork, archival watermark
entelix-memory           — Store + Embedder/Retriever/EmbeddingRetriever/VectorStore/GraphMemory traits + memory patterns
entelix-memory-openai    — OpenAI Embeddings concrete Embedder (companion)
entelix-memory-qdrant    — qdrant gRPC concrete VectorStore (companion)
entelix-memory-pgvector  — Postgres + pgvector concrete VectorStore with row-level security (companion)
entelix-graphmemory-pg   — Postgres concrete GraphMemory with WITH RECURSIVE BFS + UNNEST bulk insert (companion)
entelix-rag              — RAG primitives (Document/Lineage, splitters, Chunker, IngestionPipeline) + corrective-RAG (CRAG) recipe
entelix-persistence      — Postgres + Redis Checkpointer/Store/SessionLog with row-level security + advisory lock
entelix-tokenizer-tiktoken — Vendor-accurate TokenCounter wrapping tiktoken-rs (OpenAI BPE: cl100k_base / o200k_base / p50k_base / r50k_base)
entelix-tokenizer-hf     — Vendor-accurate TokenCounter wrapping HuggingFace `tokenizers` (Llama / Qwen / Mistral / DeepSeek / Gemma / Phi)
entelix-tools            — HttpFetchTool, Calculator, SchemaTool, skills, memory tools
entelix-tools-coding     — Sandbox-trait-backed shell / code / fs tools + Anthropic Skills layout (vertical companion)
entelix-mcp              — native JSON-RPC 2.0 over MCP streamable-http; Roots + Elicitation + Sampling channels; ChatModelSamplingProvider behind `chatmodel-sampling` feature
entelix-cloud            — Bedrock (SigV4) / Vertex (gcp_auth) / Foundry (AAD) transports
entelix-policy           — TenantPolicy, RateLimiter, PiiRedactor, CostMeter, QuotaLimiter, PolicyLayer
entelix-otel             — OpenTelemetry GenAI semconv tower::Layer + cache token telemetry + agent root span
entelix-server           — axum HTTP + 5-mode SSE + tenant middleware
entelix-auth-claude-code — Claude.ai OAuth credential provider (Claude Code CLI shared storage)
entelix-agents           — ReAct, Supervisor, Hierarchical, Chat recipes + Subagent
```

`entelix-core` depends on no other entelix crate. The DAG is enforced at workspace level.

The facade `entelix` crate gates optional sub-crates behind feature flags so you don't pay for layers you don't use. Canonical list in [`crates/entelix/Cargo.toml`](crates/entelix/Cargo.toml) — `full` enables every feature.

---

## Examples

Working examples under [`crates/entelix/examples/`](crates/entelix/examples/) — quickstart through end-to-end production workflow, covering LCEL composition, StateGraph control flow (`16_state_merge_pipeline` shows `derive(StateMerge)` + `add_contributing_node` + `add_send_edges` end-to-end), HITL graph interrupts (`04_hitl`) and HITL tool-dispatch approval pause-and-resume (`18_tool_approval`), memory, multi-agent supervisor / hierarchical recipes, every streaming mode, every codec × transport pair, MCP per-tenant isolation, MCP sampling via `ChatModelSamplingProvider` (`17_mcp_sampling_provider`), auto-compaction (`25_auto_compaction`), and the axum `AgentRouterBuilder`.

---

## Inspirations

- **Anthropic [Managed Agents](https://www.anthropic.com/engineering/managed-agents)** — Session / Harness / Hand decoupling, cattle-not-pets, lazy provisioning
- **LangChain LCEL** — `Runnable` + `.pipe()` composition primitive
- **LangGraph** — typed state-graph control flow + `Annotated[T, reducer]` + Checkpointer + HITL
- **OpenTelemetry [GenAI semconv](https://github.com/open-telemetry/semantic-conventions/tree/main/docs/gen-ai)** — vendor-neutral observability vocabulary

---

## Reading order

1. [`CLAUDE.md`](CLAUDE.md) — invariants, lock ordering, error conventions, managed-agent shape
2. [`docs/architecture/principles.md`](docs/architecture/principles.md) — living design contract
3. Per-crate `crates/<name>/CLAUDE.md` — surface, crate-local rules, forbidden patterns
4. [`docs/public-api/`](docs/public-api/) — frozen per-crate API baselines (facade excluded by design)

---

## Support

- [GitHub Issues](https://github.com/junyeong-ai/entelix/issues)
- [Architecture Principles](docs/architecture/principles.md)
- [Provider Capabilities Matrix](docs/architecture/provider-capabilities.md)

---

## License

MIT.

---

<div align="center">

**English** | **[한국어](README.ko.md)**

Made with Rust 🦀

</div>
