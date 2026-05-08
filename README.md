# entelix

> **Where every agent realizes its purpose.**
>
> **General-purpose agentic-AI SDK.** Build any agent that runs an LLM-driven control loop with tools, memory, and durable state. Not coding-agent-specific, not vertical-specific — every domain-shaped concern enters as a `trait` your code implements, not a flag you toggle.
>
> Web-service-native Rust SDK. LangChain + LangGraph parity, Anthropic *managed-agents* shape, MCP first-class (all server-initiated channels, transport hardened against frame-flood + dispatch-flood by default), multi-tenant primitives, OpenTelemetry GenAI semconv. Architectural invariants — stateless harness, event-sourced session, three-tier state model, Postgres row-level security on every persistence backend, typed audit channel, no filesystem / shell in first-party crates — are CI-enforced.

[![Rust 1.94](https://img.shields.io/badge/rust-1.94-orange)](https://www.rust-lang.org)
[![Edition 2024](https://img.shields.io/badge/edition-2024-blueviolet)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green)]()

---

## What entelix gives you

- **LCEL composition** — `Runnable<I, O>` trait with `.pipe()`. Compile-time I/O checking; the chain rejects shape mismatches at `cargo build`.
- **StateGraph control flow** — typed state, conditional edges, `add_send_edges` parallel fan-out, recursion limit, subgraphs, time-travel via `update_state`, `interrupt()` HITL.
- **`#[derive(StateMerge)]` per-field reducers** — LangGraph TypedDict parity. State struct declares `Annotated<T, R>` per field (`Append<T>` / `Max<T>` / `MergeMap<K, V>` / `Replace`); the derive emits a `<Name>Contribution` companion + builder methods. Nodes return only the slots they touched; the framework folds via `merge_contribution`. `add_send_edges` parallel-branch joins also use `S::merge` automatically — no per-call reducer wiring.
- **Durable everything** — `Checkpointer` trait + `InMemoryCheckpointer` + `PostgresCheckpointer` + `RedisCheckpointer`. Distributed session lock via `with_session_lock`. Resume `thread_id` after a pod restart. Postgres backends enforce row-level security via `current_setting('entelix.tenant_id', true)` — defense-in-depth on top of the application-level `Namespace` scoping.
- **Five memory tiers** — `BufferMemory`, `SummaryMemory`, `EntityMemory`, `SemanticMemory<E, V>`, `EpisodicMemory<E, V>` (time-ordered episode log). All compose over the `Store<V>` trait. Multi-tenant `Namespace` with mandatory non-empty `tenant_id` (compile-time + runtime enforced).
- **`GraphMemory<N, E>`** — typed nodes + timestamped edges, BFS traversal + shortest-path. `PgGraphMemory` folds BFS into one `WITH RECURSIVE` round-trip and bulk insert into one `INSERT … SELECT FROM UNNEST(…)` call.
- **Codecs through one IR** — Anthropic Messages, OpenAI Chat, OpenAI Responses, Gemini, Bedrock Converse — capability honesty + `LossyEncode` warnings on every coerced field + `Other { raw }` for unknown vendor signals (no silent fallback) + token-level streaming.
- **Transports** — Direct (any HTTPS), Bedrock (SigV4 + bearer), Vertex (gcp_auth), Foundry (api-key + AAD). Sparse codec×transport matrix (ADR-0018).
- **MCP first-class with all 3 server-initiated channels** — `Roots` + `Elicitation` + `Sampling`. Wire your `ChatModel` as the sampling backend in 5 lines via `entelix-mcp-chatmodel`. Per-tenant `(tenant_id, server_name)` pool isolation. HTTP-only by design (invariant 9).
- **Multi-tenant policy** — `RateLimiter`, `PiiRedactor` (bidirectional), `CostMeter` (`rust_decimal`, transactional, charged only inside the `Ok` branch), `QuotaLimiter`, `TenantPolicy` aggregate.
- **OpenTelemetry GenAI semconv 0.32** — `OtelLayer` (tower middleware on both model and tool invocations) emits `gen_ai.*` events including cache token telemetry (`cached_input_tokens`, `cache_creation_input_tokens`, `reasoning_tokens`). Tool I/O capture mode (`Off` / `Truncated{4096}` / `Full`). `Agent::execute` opens an `entelix.agent.run` root span so trace UIs show agent → model → tool as one tree.
- **Typed audit channel** — `entelix::AuditSink` with 4 `record_*` verbs (`sub_agent_invoked` / `agent_handoff` / `resumed` / `memory_recall`). `entelix::SessionAuditSink` maps onto `GraphEvent` so replays reconstruct managed-agent lifecycle without re-running the dispatch path.
- **axum HTTP server** — `AgentRouterBuilder` with sync / 5-mode SSE / wake-from-checkpoint endpoints + tenant header extraction.
- **`Tool<D = ()>` trait + built-ins + `#[tool]` macro** — `HttpFetchTool` (3-layer SSRF defense), `CalculatorTool`, `SearchTool` adapter, `SchemaTool` typed-I/O adapter (auto-generated schema goes through `LlmFacingSchema::strip` so envelope keys never reach the model), sandboxed shell / file / code / list-dir tools delegating syscalls through the `Sandbox` trait. The `#[tool]` proc-macro (in `entelix-tool-derive`) turns an `async fn(ctx: &AgentContext<D>, ...args) -> Result<O>` into a fully-wired `SchemaTool` impl — no boilerplate. Operators thread typed dependencies through `D`; the layer ecosystem (`PolicyLayer`, `OtelLayer`, `RetryService`, `ApprovalLayer`) stays D-free and composes unchanged regardless of operator deps.
- **Typed structured output with semantic validation** — `complete_typed::<O>()` parses the model's response into your typed `O`; `complete_typed_validated` runs an `OutputValidator<O>` (any `Fn(&O) -> Result<()>` works) so cross-field invariants the JSON schema can't express get caught and corrected. Schema-mismatch and validator failures share one budget (`ChatModelConfig::with_validation_retries`) and one typed retry channel (`Error::ModelRetry { hint: RenderedForLlm<String>, ... }`); the loop reflects the corrective hint to the model and re-invokes within budget.
- **Type-enforced conversation compaction** — `Compactor` trait + sealed `CompactedHistory`. The `tool_call` / `tool_result` pair invariant is structurally impossible to violate: `ToolPair` fields are private, so external compactors can drop or pass through tool round-trips obtained from `CompactedHistory::group(events)` but cannot synthesize unmatched ones. `HeadDropCompactor` ships as the reference "drop oldest until fits" strategy.
- **Pre-built recipes** — `create_react_agent`, `create_supervisor_agent`, `create_chat_agent`. Nested-supervisor topologies wire `team_from_supervisor` into a parent `create_supervisor_agent`.

## Quickstart — single API call

```rust
use std::sync::Arc;
use entelix::auth::ApiKeyProvider;
use entelix::codecs::AnthropicMessagesCodec;
use entelix::ir::Message;
use entelix::transports::DirectTransport;
use entelix::{ChatModel, ExecutionContext};

#[tokio::main]
async fn main() -> entelix::Result<()> {
    let creds = Arc::new(ApiKeyProvider::anthropic(std::env::var("ANTHROPIC_API_KEY")?));
    let transport = DirectTransport::anthropic(creds)?;
    let model = ChatModel::new(AnthropicMessagesCodec::new(), transport, "claude-opus-4-7")
        .with_system("Answer in one sentence.");

    let reply = model
        .complete(vec![Message::user("Define entelechy.")], &ExecutionContext::new())
        .await?;
    println!("{reply:?}");
    Ok(())
}
```

## LCEL chain

```rust
use entelix::{ChatPromptTemplate, JsonOutputParser, RunnableExt};

let prompt = ChatPromptTemplate::from_messages([
    ("system", "Translate to {language}. Reply as JSON {{\"text\": \"...\"}}."),
    ("user", "{question}"),
]);
let parser = JsonOutputParser::<Translation>::new();

let chain = prompt.pipe(model).pipe(parser);
let result: Translation = chain.invoke(input, &ExecutionContext::new()).await?;
```

## StateGraph (LangGraph TypedDict parity)

```rust
use entelix::{Annotated, Append, Max, StateGraph, StateMerge, RunnableLambda};

#[derive(Clone, Default, StateMerge)]
struct AgentState {
    log: Annotated<Vec<String>, Append<String>>,   // accumulated across nodes
    score: Annotated<i32, Max<i32>>,                // best-of across branches
    last_phase: String,                             // last-write-wins
}

let plan = RunnableLambda::new(|s: AgentState, _ctx| async move {
    // Touched: log + last_phase. score left as None → preserved.
    Ok(AgentStateContribution::default()
        .with_log(vec![format!("planning at score={}", s.score.value)])
        .with_last_phase("plan".into()))
});

let graph = StateGraph::<AgentState>::new()
    .add_contributing_node("plan", plan)
    .add_contributing_node("score", scorer)
    .add_send_edges(                                // parallel fan-out, S::merge folds
        "plan",
        ["a", "b", "c"],
        |s| vec![("a".into(), s.clone()), ("b".into(), s.clone()), ("c".into(), s.clone())],
        "score",
    )
    .add_conditional_edges(
        "score",
        |s: &AgentState| (if s.score.value >= 80 { "done" } else { "plan" }).to_owned(),
        [("plan", "plan"), ("done", entelix::END)],
    )
    .set_entry_point("plan")
    .with_checkpointer(Arc::new(postgres_checkpointer))
    .compile()?;

let result = graph.invoke(AgentState::default(), &ctx).await?;
```

## Production HTTP server

```rust
use std::sync::Arc;
use entelix::{
    AgentRouterBuilder, ChatModel, HostAllowlist, HttpFetchTool, OtelLayer, PolicyLayer,
    SERVER_DEFAULT_TENANT_HEADER, ToolIoCaptureMode, ToolRegistry, create_react_agent,
};

let model = ChatModel::new(codec, transport, "claude-opus-4-7")
    .layer(PolicyLayer::new(policy_manager))
    .layer(OtelLayer::new("anthropic")
        .with_tool_io_capture(ToolIoCaptureMode::Truncated { max_bytes: 4096 }));

let tools = ToolRegistry::new()
    .register(Arc::new(HttpFetchTool::builder()
        .with_allowlist(HostAllowlist::new().add_subdomain_root("*.api.example.com"))
        .build()?))?;
let agent = create_react_agent(model, tools)?;

// `with_tenant_header` opts the router into multi-tenant strict mode:
// every request MUST carry the named header. Missing → 400 + typed
// `ServerError::MissingTenantHeader`. Omit the call entirely to run
// single-tenant under `entelix::DEFAULT_TENANT_ID`.
let router = AgentRouterBuilder::new(agent)
    .with_checkpointer(Arc::clone(&postgres_checkpointer))
    .with_tenant_header(SERVER_DEFAULT_TENANT_HEADER)
    .build()?;

let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
axum::serve(listener, router).await?;
```

See [`crates/entelix/examples/14_serve_agent.rs`](crates/entelix/examples/14_serve_agent.rs) for an end-to-end CI-deterministic demo (in-memory tower stack — no socket, no LLM).

Endpoints:
- `POST /v1/threads/{thread_id}/runs` — synchronous invoke.
- `GET  /v1/threads/{thread_id}/stream?mode=values|updates|messages|debug|events` — SSE.
- `POST /v1/threads/{thread_id}/wake` — resume from checkpoint with `Command::{Resume, Update}`.
- `GET  /v1/health` — liveness.

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
entelix-memory           — Store + Embedder/Retriever/VectorStore/GraphMemory traits + 5 LC-style memory patterns
entelix-memory-openai    — OpenAI Embeddings concrete Embedder (companion)
entelix-memory-qdrant    — qdrant gRPC concrete VectorStore (companion)
entelix-memory-pgvector  — Postgres + pgvector concrete VectorStore with row-level security (companion)
entelix-graphmemory-pg   — Postgres concrete GraphMemory with WITH RECURSIVE BFS + UNNEST bulk insert (companion)
entelix-persistence      — Postgres + Redis Checkpointer/Store/SessionLog with row-level security + advisory lock
entelix-tools            — HttpFetchTool, CalculatorTool, SchemaTool, sandboxed tools, skills, memory tools
entelix-mcp              — native JSON-RPC 2.0 over MCP streamable-http; Roots + Elicitation + Sampling channels
entelix-mcp-chatmodel    — bridges MCP sampling/createMessage onto a ChatModel<C, T> (companion)
entelix-cloud            — Bedrock (SigV4) / Vertex (gcp_auth) / Foundry (AAD) transports
entelix-policy           — TenantPolicy, RateLimiter, PiiRedactor, CostMeter, QuotaLimiter, PolicyLayer
entelix-otel             — OpenTelemetry GenAI semconv tower::Layer + cache token telemetry + agent root span
entelix-server           — axum HTTP + 5-mode SSE + tenant middleware
entelix-agents           — ReAct, Supervisor, Hierarchical, Chat recipes + Subagent (F7 mitigation)
```

`entelix-core` depends on no other entelix crate. The DAG is enforced at workspace level.

The facade `entelix` crate gates optional sub-crates behind feature flags so you don't pay for layers you don't use. Canonical list in [`crates/entelix/Cargo.toml`](crates/entelix/Cargo.toml) — `full` enables every feature.

## Examples

Working examples under [`crates/entelix/examples/`](crates/entelix/examples/) — quickstart through end-to-end production workflow, covering LCEL composition, StateGraph control flow (`16_state_merge_pipeline` shows `derive(StateMerge)` + `add_contributing_node` + `add_send_edges` end-to-end), HITL graph interrupts (`04_hitl`) and HITL tool-dispatch approval pause-and-resume (`18_tool_approval`), memory, multi-agent supervisor / hierarchical recipes, every streaming mode, every codec × transport pair, MCP per-tenant isolation, MCP sampling via `ChatModelSamplingProvider` (`17_mcp_sampling_provider`), and the axum `AgentRouterBuilder`.

## Migrating from another SDK?

- [`docs/migrations/langgraph-python.md`](docs/migrations/langgraph-python.md) — Python LangGraph users
- [`docs/migrations/rig.md`](docs/migrations/rig.md) — `rig` Rust users

## What entelix is NOT

- No direct filesystem / shell calls in first-party crates — sandboxed tool wrappers exist (`SandboxedShellTool`, `SandboxedReadFileTool`, …) but every syscall delegates through the `Sandbox` trait; concrete `Sandbox` impls (Landlock / Seatbelt-backed) ship as 1.x companion crates, not in core.
- No local inference — application-layer SDK; pair with candle / mistral.rs.
- No Vector DB reimplementation — production `VectorStore` impls ship as `entelix-memory-qdrant` / `entelix-memory-pgvector`; plug your own via the trait.
- No Document Loaders — that's swiftide's job.
- No Python interop — Rust-first.

## Inspirations

- **Anthropic [Managed Agents](https://www.anthropic.com/engineering/managed-agents)** — Session/Harness/Hand decoupling, cattle-not-pets, lazy provisioning
- **LangChain LCEL** — `Runnable` + `.pipe()` composition primitive
- **LangGraph** — typed state-graph control flow + `Annotated[T, reducer]` + Checkpointer + HITL
- **OpenTelemetry GenAI semconv** — vendor-neutral observability vocabulary

## Reading order

1. [`docs/architecture/overview.md`](docs/architecture/overview.md) — big picture
2. [`docs/architecture/state-graph.md`](docs/architecture/state-graph.md) — LangGraph parity
3. [`docs/architecture/runnable-and-lcel.md`](docs/architecture/runnable-and-lcel.md) — LangChain parity
4. [`docs/architecture/session-and-memory.md`](docs/architecture/session-and-memory.md) — three-tier state
5. [`docs/architecture/managed-agents.md`](docs/architecture/managed-agents.md) — Anthropic shape
6. [`docs/adr/`](docs/adr/) — architecture decision records
7. [`docs/public-api/`](docs/public-api/) — frozen 1.0-candidate per-crate API baselines (facade excluded by design)
8. [`docs/migrations/`](docs/migrations/) — porting guides

## License

MIT.
