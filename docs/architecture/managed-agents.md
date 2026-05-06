# Managed-Agent Shape — Anthropic Mapping

Source: [Anthropic Engineering — Managed Agents](https://www.anthropic.com/engineering/managed-agents)

This document maps Anthropic's managed-agents architecture to entelix's workspace. The mapping is **load-bearing**: it is the reason entelix exists in the shape it does.

## Anthropic's three components

> *"Decoupling the brain from the hands"*

| Component | Anthropic role | One-line definition |
|---|---|---|
| **Session** | the persisted state | append-only event log, externally stored, outlives any harness |
| **Harness** | the brain | Claude + a loop, **stateless**, reads session → produces next event |
| **Sandbox** | the hands | tool execution environment, accessed via `execute(name, input) → string` |

> *"Cattle, not pets"* — every component is replaceable. Only events persist.

## entelix mapping

| Anthropic | entelix crate | Concrete type | Notes |
|---|---|---|---|
| Session (event log) | `entelix-session` | `SessionGraph { events: Vec<GraphEvent> }` | append-only; nodes/branches derived |
| Session storage | `entelix-persistence` | `PostgresSessionLog`, `RedisSessionLog` | externalized as Anthropic prescribes |
| Working state | `entelix-graph` | `StateGraph<S>` + `Checkpointer<S>` | LangGraph-style typed state per thread |
| Harness | `entelix-core` | `Agent` + `Codec` + `Transport` (Runnable<I,O>) | stateless per-request |
| Sandbox | `entelix-core::tools::Tool` + `entelix-mcp` | `dyn Tool` trait | single `execute()` method |
| Sandbox lifecycle | external (caller's deployment) | container/restate/serverless | entelix does NOT manage |
| Credential isolation | `entelix-core::auth` + `entelix-policy` | `CredentialProvider` + `PiiRedactor` hook | tokens never reach Tool input |
| Cross-thread memory | `entelix-memory` | `Store<V>` + `Namespace { tenant_id, scope }` | not in Anthropic spec — entelix addition for full LG parity |

## Interface mapping

Anthropic's prescribed interfaces:

```
emitEvent(sessionId, event)   # harness → session
getEvents(sessionId, query)   # harness ← session
wake(sessionId)               # external → harness
execute(name, input)          # harness → hand
```

entelix equivalents:

```rust
// emit (per event, write-through)
SessionGraph::append(event)
  → Persistence::append_event(session_id, event).await   // entelix-persistence

// getEvents (selective replay)
SessionGraph::events_since(cursor)
SessionGraph::current_branch_messages()
SessionGraph::events_for_node(node_id)

// wake / resume
Agent::wake(session_id, persistence)
  → load events → SessionGraph → resume from last event

// execute (Tool dispatch)
ToolRegistry::dispatch(name, input, ctx)
  → match registered Tool / MCP tool / sub-agent delegate
```

## Cattle-not-pets — concrete consequences

| What can crash | What survives |
|---|---|
| Agent process | SessionGraph in Postgres |
| Pod hosting agent | session_id; rebind on any pod |
| Tool execution container | tool error event, agent retries or recovers |
| MCP server | entelix-mcp emits ToolError, agent decides |
| One LLM call (timeout, 5xx) | retry inside harness loop, no session impact |

| What MUST survive | Where it lives |
|---|---|
| Every user input | `GraphEvent::UserMessage` in events |
| Every assistant token | aggregated `AssistantMessage` event after stream end |
| Every tool call & result | `GraphEvent::ToolCall` + `GraphEvent::ToolResult` |
| Branch / fork structure | derived from `BranchCreate` events |
| Checkpoint markers | `GraphEvent::Checkpoint` events |

## Lazy provisioning

Anthropic emphasizes provisioning sandboxes only at first tool call. entelix mirrors this:

- **MCP servers**: `entelix-mcp::McpManager` does NOT connect at agent build time. Connection opens at first dispatch to a tool of that namespace; idle connections close after configurable TTL.
- **Cloud transports**: `gcp_auth`, `aws-config`, `azure_identity` cache credentials but do not pre-acquire tokens; token acquisition is on first request.
- **Persistence**: connection pool initialized at agent build, but session row creation deferred to first event append.

Implication: a brain with 10 hands and no current task uses ~10kB of memory and 0 active connections.

## Transport resource bounds

Server-initiated traffic on the MCP streamable-HTTP transport is bounded by default — a hostile or malfunctioning peer cannot pin client memory. Two knobs on `McpServerConfig`:

- **`with_max_frame_bytes(n)`** (default: 1 MiB) — caps a single SSE frame's accumulated byte length. The listener closes the connection if any frame grows past the cap without a `\n\n` terminator. JSON-RPC frames over MCP are typically kilobytes; the default has 100×–10000× headroom.
- **`with_listener_concurrency(n)`** (default: 32) — caps in-flight server-initiated dispatches (`roots/list`, `elicitation/create`, `sampling/createMessage`). Excess requests are dropped (with `tracing::warn!`) rather than queued — server is expected to retry on its own cadence. Drop-over-queue avoids creating a second OOM vector.

Both bounds default to be invisible to legitimate traffic and decisive against a hostile peer. Operators raise the caps explicitly when running large structured-output `sampling/createMessage` payloads or sustained server-initiated batches. See ADR-0067.

## Brain passes hand — sub-agent semantics

Anthropic: *"brains can pass hands to one another"*

entelix:

```rust
use entelix::Subagent;

// Strict-name sub-agent — typo in restrict_to surfaces as
// Error::Config when build() runs. Identity is set at builder
// construction (ADR-0093) so the Subagent is inspectable
// (`metadata()`, `name()`, `description()`) before the
// `into_tool()` conversion.
let sub = Subagent::builder(model, &parent_registry, "research_assistant", "Search the web for citations.")
    .restrict_to(&["search", "fetch_url"])
    .with_skills(&parent_skills, &["citation-format"])?
    .with_sink(parent_sink)
    .build()?
    .into_tool()?;

// Predicate sub-agent — frozen view, evaluated once per parent
// tool at build() time.
let sub = Subagent::builder(model, &parent_registry, "read_only", "Read-only assistant.")
    .filter(|t| t.metadata().name.starts_with("read_"))
    .build()?
    .into_tool()?;
```

- `parent_registry.restricted_to(...)` / `parent_registry.filter(...)`
  produce a **narrowed view** that shares the parent's layer
  factory by `Arc` — `PolicyLayer`, `OtelLayer`, retry middleware
  apply transparently to sub-agent dispatches. Constructing a
  fresh `ToolRegistry::new()` would silently drop the layer
  stack — `scripts/check-managed-shape.sh` enforces against this
  regression (ADR-0035, invariant 7).
- Parent's `CredentialProvider` lives exclusively on the
  `Transport` (invariant 10) — sub-agents that share the parent's
  `ChatModel` inherit credentials transparently; tools never see
  them.
- Sub-agent dispatch emits `record_sub_agent_invoked` on the
  parent's `AuditSink` (invariant 18, ADR-0037). Replays of the
  parent's session reconstruct the child invocation without
  re-running the sub-agent.
- Cost/usage rolls up to the parent's `entelix-policy::CostMeter`
  through the inherited layer stack.
- `crates/entelix-agents/tests/subagent_capture_pattern.rs`
  pins the strict / graceful asymmetry: `restrict_to` rejects
  typos at `build()`, `filter` accepts empty results
  (pure-orchestration shape), the predicate is evaluated once
  per parent tool at `build()` (frozen view).

This is the **one-line sub-agent pattern** that LangGraph requires ~50 lines to express. We get it from honoring the brain/hand decoupling.

## Human-in-the-loop (HITL) approval

Two complementary flows for human-gated tool dispatch:

### In-band wait — `ChannelApprover`

For approvals that resolve in seconds (operator on a Slack channel responds quickly), `entelix::ChannelApprover` blocks `Approver::decide` on a `oneshot` channel until the decision arrives. The agent task stays alive; no checkpoint is taken. Fastest UX when the human is "in the loop" synchronously.

### Out-of-band pause-and-resume — `ApprovalLayer` + `AwaitExternal`

For long-running review (operator approves over hours/days through a separate UI / job queue), `Approver::decide` returns `ApprovalDecision::AwaitExternal`. `ApprovalLayer` (a `tower::Layer<S>` over `Service<ToolInvocation>`) raises `Error::Interrupted { payload: { kind: "approval_pending", tool_use_id, tool, input, run_id } }`. The graph dispatch loop catches it, persists a checkpoint with pre-node state, and surfaces the typed error to the caller. The agent run releases inflight resources cleanly.

When the operator's decision lands, resume threads it through the typed `Command::ApproveTool { tool_use_id, decision }` resume primitive on `CompiledGraph::resume_with`. The graph's resume path attaches the decision to `ExecutionContext` internally; the layer's override-lookup runs *before* the approver and short-circuits — the approver isn't re-asked, the pending dispatch completes with the operator's decision.

`ReActAgentBuilder::with_approver(approver)` and `SubagentBuilder::with_approver(approver)` both auto-wire `ApprovalLayer` into the tool registry; sub-agent narrowed views inherit the approval layer through the `Arc`-shared layer factory (ADR-0035 + ADR-0070). The agent's `AgentEventSink<S>` observes `AgentEvent::ToolCallApproved` / `ToolCallDenied` for both flows. See `crates/entelix/examples/18_tool_approval.rs` for the full pause-and-resume pattern.

ADRs 0070 (HITL approval Layer) + 0071 (`AwaitExternal` graph-interrupt integration). The two flows compose: an approver may return `Approve` for some tool calls, `AwaitExternal` for others — the layer handles each path correctly.

## Per-call overrides — `RunOverrides`

`entelix::RunOverrides` rides on `ExecutionContext::extension::<RunOverrides>()` so per-call knobs (`with_model`, `with_system_prompt`, `with_max_iterations`) reach `ChatModel::complete_full` / `stream_deltas` and `CompiledGraph::execute_loop_inner` without requiring a fresh `ChatModel` per variant. The agent surfaces a convenience entry `Agent::execute_with(input, overrides, ctx)` that attaches the overrides for the duration of the call. Compile-time recursion limits stay authoritative (operators can lower per-call but never raise — F6 mitigation preserved). ADR-0069.

## Tool-dispatch ambient state — `ToolDispatchScope`

Some tools need ambient state (tokio task-locals, RLS `SET LOCAL` settings, tracing scopes) active while their future polls — state the SDK can't supply through `ExecutionContext` directly because it lives in a thread-local rather than a typed field. `ToolDispatchScope` is a trait with one method `wrap(ctx, fut)`; operators implement it with their enter/exit machinery (`tokio::task_local!::scope(value, fut)`, etc.) and attach via `ScopedToolLayer::new(scope)` to the registry. The wrap fires for every tool dispatch — parent agent, sub-agent narrowed view, recipe-driven direct dispatch — uniformly, because all dispatch routes through the registry's layer stack. ADR-0068.

## Token isolation — concrete enforcement

Anthropic: *"tokens are never reachable from the sandbox where Claude's code runs"*

entelix:

1. `CredentialProvider::resolve()` returns a `secrecy::Secret<String>` only inside `Transport::authorize()`. The token is added to `http::HeaderMap` and immediately scoped out.
2. `Tool::execute(input: Value, ctx: &AgentContext<D>)` — `AgentContext<D>` wraps `ExecutionContext` (tenant_id, span context, registry handles) and an operator-supplied typed `D` (defaults to `()`); neither carries the token. Compile-checked: `ExecutionContext` does not embed `CredentialProvider`. ADR-0084.
3. `entelix-policy::PiiRedactor` runs in a `pre_request` hook to scrub user-provided tokens before they reach `Codec::encode_request`.
4. CI grep gate: `grep -rE 'CredentialProvider|Secret<String>' crates/entelix-tools/` must return zero hits — tools cannot see credentials.

## Why this is hard to retrofit

Most Rust agent libraries (rig, swiftide) bind state to the agent struct or use ad-hoc message lists. Refactoring to managed-agent shape later requires:
- redesigning the API (where does session_id flow?)
- splitting state from compute (whole crate restructure)
- introducing event log semantics (every state change becomes an event)

entelix starts here. It's not a 1.5 add-on; it's the v0.1 spine.

## Resolved design questions

These were open at design time; 1.0 RC closes them:

1. **Sub-agent isolation depth** — *closed by ADR-0035*. Default
   is **filtered inherit** through `parent_registry.restricted_to(...)` /
   `parent_registry.filter(...)`; the layer stack rides over by
   `Arc`. `scripts/check-managed-shape.sh` enforces against the
   regression to a fresh `ToolRegistry::new()`.
2. **Cross-pod hand sharing** — *closed by `entelix-persistence::with_session_lock`*.
   Distributed advisory lock keyed on `(tenant_id, thread_id)`
   serialises mutations across pods; `PostgresPersistence` uses
   `pg_advisory_xact_lock(hash(tenant, thread))`,
   `RedisPersistence` uses `SET NX PX` with a Lua release script.
3. **Brain hot-swap mid-session** — *closed by IR provider-neutrality*.
   The session log carries `GraphEvent`s referencing only the IR
   shape; swap `ChatModel<C, T>` between turns and the next turn
   sees the full prior transcript through the new codec without
   recomputation.

## Cross-references

- ADR-0035 — managed-agent shape enforcement (`Subagent` MUST
  narrow the parent registry; `check-managed-shape.sh`).
- ADR-0037 — `AuditSink` typed channel for managed-agent
  lifecycle events (invariant 18).
- ADR-0064 — 1.0 release charter (semver promise + companion
  cadence policy).
- `crates/entelix-agents/tests/subagent_layer_inheritance.rs` —
  layer-stack inheritance regression suite.
- `crates/entelix-agents/tests/subagent_capture_pattern.rs` —
  strict-vs-graceful capture pattern regression suite.
