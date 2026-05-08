# Migration Guide — `rig` → entelix

> Audience: existing [`rig`](https://github.com/0xPlaygrounds/rig)
> users. `rig` is a focused Rust SDK for "talk to an LLM and call
> tools"; entelix is a wider control-flow + multi-tenant + HTTP
> service SDK. The two overlap on the model-call surface and
> diverge sharply at the agent layer. This guide focuses on the
> overlap.

## At a glance

| Concept | rig | entelix |
|---|---|---|
| Provider client | `providers::openai::Client` | `ChatModel<C, T>` (codec + transport) |
| Completion call | `client.completion_request(...).send().await` | `model.complete(messages, &ctx).await` |
| Streaming | `.stream().await` (provider-shaped) | `model.stream_deltas(messages, &ctx).await` (`StreamDelta` IR) |
| Tools | `#[tool]` macro on `impl Tool` | `#[tool]` proc-macro (entelix-tool-derive) on an `async fn` |
| Agents | `Agent::new(model).preamble(...).tool(...)` | `create_react_agent(model, tools)?` (or build a `StateGraph`) |
| Memory | (loose conventions in user code) | `entelix::{BufferMemory, EntityMemory, …}` |
| Multi-step graphs | (none) | `entelix::StateGraph<S>` |
| Multi-tenant | (none) | `ExecutionContext::with_tenant_id` + `entelix-policy` |
| HTTP service | (none) | `entelix::AgentRouterBuilder` |

The biggest mental shift: rig is **flat** — call → response,
optionally with a tool loop. entelix has **layers** — composition
(`Runnable`), control flow (`StateGraph`), persistence
(`Checkpointer`), observability (`Hook`), HTTP (`AgentRouterBuilder`).
You don't pay for layers you don't use, but they're available
when you scale up.

## Provider client → ChatModel

```rust
// rig
use rig::providers::anthropic::ClientBuilder;
let client = ClientBuilder::new().build();
let agent = client.agent("claude-opus-4-7").build();
let reply = agent.prompt("Hello").await?;
```

```rust
// entelix
use std::sync::Arc;
use entelix::{AnthropicMessagesCodec, ChatModel};
use entelix::auth::ApiKeyProvider;
use entelix::transports::DirectTransport;

let creds = Arc::new(ApiKeyProvider::anthropic("sk-..."));
let transport = DirectTransport::anthropic(creds)?;
let model = ChatModel::new(AnthropicMessagesCodec::new(), transport, "claude-opus-4-7");
let reply = model
    .complete(
        vec![entelix::ir::Message::user("Hello")],
        &entelix::ExecutionContext::new(),
    )
    .await?;
```

The two-layer split (codec + transport) is more verbose than rig's
single `Client`, but it's how entelix supports the same Claude on
the Anthropic API, AWS Bedrock, GCP Vertex, and Azure Foundry — a
one-line transport swap rather than a new provider integration. See
`examples/12_compat_matrix.rs` for the 10-cell sparse matrix.

## Tools

```rust
// rig
use rig::tool::Tool;
#[derive(Deserialize, JsonSchema)]
struct LookupArgs { id: String }

#[tool]
async fn lookup(args: LookupArgs) -> Result<String> {
    Ok(fetch(&args.id).await?)
}
```

```rust
// entelix
use entelix::{AgentContext, Result};
use entelix_tool_derive::tool;

#[tool]
/// Look up a record by id.
async fn lookup(ctx: &AgentContext<()>, id: String) -> Result<String> {
    fetch(&id, ctx.core()).await
}
```

entelix's `#[tool]` proc-macro generates a `SchemaTool` impl from
the function signature: an `Input` struct (`Deserialize +
JsonSchema` over the params), a unit struct named after the fn
(snake_case → PascalCase), and the routing impl. The first
doc-comment paragraph becomes the LLM-facing description; the
JSON schema is auto-generated. Operators that need the manual
form (effect classes, custom routing, generic tools) implement
`Tool` directly — the macro is sugar over the underlying trait,
not a replacement.

The `ctx: &AgentContext<D>` parameter carries an operator-supplied
typed `D` (defaults to `()`); pass `&AgentContext<MyDeps>` for
typed handles (DB connection, auth client, …). The layer ecosystem
(`PolicyLayer`, `OtelLayer`, `RetryService`, `ApprovalLayer`) stays
`D`-free so existing layer impls compile unchanged regardless of
your deps choice.

## Agent loop

rig's `Agent` is a built-in tool-calling loop that bakes the
preamble into every request:

```rust
// rig
let agent = client.agent("claude-opus-4-7")
    .preamble("You are a helpful assistant.")
    .tool(lookup)
    .build();
let reply = agent.prompt("Find record xyz").await?;
```

The entelix equivalent is `create_react_agent` from
`entelix-agents` — a [`StateGraph`] with two nodes (planner /
tools) and a conditional edge:

```rust
// entelix
use std::sync::Arc;
use entelix::create_react_agent;

let model = ChatModel::new(codec, transport, "claude-opus-4-7")
    .with_system("You are a helpful assistant.");
let agent = create_react_agent(model, vec![Arc::new(LookupTool)])?;
let final_state = agent.invoke(initial_state, &ctx).await?;
```

The graph is observable: you can `agent.stream(state, mode, &ctx)`
to watch the tool-calling loop tick by tick. See
`examples/06_supervisor.rs` and `08_streaming_modes.rs`.

## What entelix adds (not in rig)

- **Multi-step state machines** — `StateGraph<S>` with conditional
  edges, `add_send_edges` parallel fan-out, recursion limit,
  subgraphs, time-travel via `update_state`.
- **Per-field state reducers via derive macro** — `#[derive(StateMerge)]`
  + `Annotated<T, R>` per field. Nodes return only the slots they
  touched; the framework folds via `merge_contribution`. Direct
  Rust port of LangGraph's `Annotated[T, reducer]` pattern.
- **Crash-resume** — `Checkpointer` + `CompiledGraph::resume`. A
  pod can die mid-conversation; another pod can pick up the
  `thread_id` from durable storage. `InMemoryCheckpointer` for
  tests, `PostgresCheckpointer` / `RedisCheckpointer` for
  production (the Postgres backend enforces row-level security
  via `current_setting('entelix.tenant_id', true)`).
- **Memory tiers** — `BufferMemory`, `SummaryMemory`,
  `EntityMemory`, `SemanticMemory<E, V>`, `EpisodicMemory<E, V>`
  (time-ordered episode log). All compose over the `Store<V>`
  trait. `GraphMemory<N, E>` for typed nodes + timestamped edges
  (`PgGraphMemory` folds BFS into one `WITH RECURSIVE` round-trip).
- **HTTP service** — `entelix::AgentRouterBuilder` mounts any
  `CompiledGraph` under `/v1/threads/{id}/...` with sync, SSE
  streaming, and resume endpoints.
- **Multi-tenant primitives** — `entelix::{RateLimiter,
  PiiRedactor, CostMeter, QuotaLimiter}` keyed by
  `ExecutionContext::tenant_id`. `Namespace::new(tenant_id)`
  enforces non-empty `tenant_id` at compile time + runtime
  (invariant 11 / F2 mitigation).
- **OpenTelemetry GenAI semconv 0.32** — `entelix::OtelLayer`
  (tower middleware) emits the standard `gen_ai.*` span/event
  attributes including cache token telemetry. `Agent::execute`
  opens an `entelix.agent.run` root span so trace UIs show
  agent → model → tool as one tree.
- **Typed audit channel** — `entelix::AuditSink` with 4
  `record_*` verbs (`sub_agent_invoked` / `agent_handoff` /
  `resumed` / `memory_recall`) plus `entelix::SessionAuditSink`
  that maps onto `GraphEvent` for replay.
- **MCP first-class with all 3 server-initiated channels** —
  `entelix::McpManager` with per-tenant pool isolation, plus
  `RootsProvider` / `ElicitationProvider` / `SamplingProvider`
  trait surfaces. Wire your `ChatModel` as the sampling backend
  in 5 lines via `entelix-mcp-chatmodel`'s
  `ChatModelSamplingProvider`.
- **Human-in-the-loop tool approval** — `entelix::ApprovalLayer`
  (Tower middleware) gates every `Tool::execute` through an
  operator-supplied `Approver`. Two flows compose: in-band
  `ChannelApprover` for fast decisions (operator on a Slack
  channel responds in seconds) and out-of-band
  `Command::ApproveTool { tool_use_id, decision }` for long
  reviews (`AwaitExternal` decisions raise `Error::Interrupted`
  so the agent run pauses cleanly via the existing
  graph-checkpoint infrastructure; resume re-enters with the
  operator's eventual decision threaded through the typed
  resume command). ADRs 0070/0071/0072, example
  `18_tool_approval.rs`. rig has no equivalent surface.
- **Per-call ChatModel + graph overrides** — `entelix::RunOverrides`
  rides on `ExecutionContext::extension` so per-call knobs
  (`with_model("haiku")` for cheap classification routes,
  `with_system_prompt(...)` for tenant-specific personas,
  `with_max_iterations(8)` for clamped exploratory runs) reach
  every dispatch without rebuilding the `ChatModel`. ADR-0069.
- **Tool-dispatch ambient state hook** — `entelix::tools::ToolDispatchScope`
  + `ScopedToolLayer` let operators wrap every `Tool::execute`
  future with task-locals or RLS `SET LOCAL` settings the SDK
  cannot supply through `ExecutionContext` directly. Crucial
  for multi-tenant Postgres-RLS deployments where the tool's
  query path reads `current_setting('entelix.tenant_id', true)`.
  ADR-0068.

## What rig has that entelix doesn't

- **Larger provider matrix at the surface** — rig integrates
  Cohere, Mistral, etc. directly; entelix routes through codec +
  transport (Anthropic / OpenAI Chat / OpenAI Responses / Gemini /
  BedrockConverse). New codecs land in `entelix-core::codecs` per
  provider; community-supplied codecs implement the same trait.

## Porting strategy

1. **Replace the provider client** with a `ChatModel<C, T>` matching
   your provider (codec) and route (transport).
2. **Translate tools** to `#[tool]` proc-macro on an `async fn`
   — the JSON schema is auto-generated from the param types via
   `schemars::JsonSchema`. Operators with custom-effect or
   generic tools implement `Tool` directly (the macro is sugar
   over the trait).
3. **For ReAct-shaped agents**: swap `Agent::new(...).build()` for
   `create_react_agent(model, tools)?`. The same prompt/system
   string flows through `ChatModel::with_system`.
4. **For more complex flows**: build a `StateGraph` directly. See
   `examples/03_state_graph.rs` and `examples/06_supervisor.rs`.
5. **Add the layers you need**: persistence, policy, otel, server,
   MCP. Each is a separate crate behind a feature flag — opt in as
   your deployment grows.
6. **For RAG / knowledge-graph patterns**: pick one of the five
   memory patterns over `Store<V>` (`SemanticMemory<E, V>` for
   vector search, `EpisodicMemory<E, V>` for time-ordered traces)
   or the `GraphMemory<N, E>` tier (typed relationships, separate
   trait — not a `Store` pattern) and wire the matching backend
   (`entelix-memory-pgvector`, `entelix-memory-qdrant`,
   `entelix-graphmemory-pg`).

## See also

- `docs/migrations/langgraph-python.md` — for users coming from
  Python LangGraph (covers the `derive(StateMerge)` ergonomic in
  depth).
- `docs/architecture/runnable-and-lcel.md` — the composition layer
  rig users will be most familiar with.
- `examples/01_quickstart.rs`, `examples/02_lcel_chain.rs` — the
  closest analogues to a rig `agent.prompt(...)`.
- `examples/16_state_merge_pipeline.rs` — `derive(StateMerge)` +
  `add_contributing_node` + `add_send_edges` end-to-end.
- `examples/17_mcp_sampling_provider.rs` —
  `ChatModelSamplingProvider` wiring against a stub Codec/Transport.
