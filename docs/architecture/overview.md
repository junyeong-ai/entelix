# entelix Architecture — Overview (FINAL)

## The shape — 4 horizontal layers

```
┌──────────────────────────────────────────────────────────────────────┐
│  Layer 4 — Recipes & deployment                                      │
│  entelix-agents          (ReAct, Supervisor, Hierarchical, Chat)     │
│  entelix-server          (axum HTTP + 5-mode SSE)                    │
│  entelix-policy          (multi-tenant, rate-limit, PII, cost)       │
│  entelix-otel            (GenAI semconv observability)               │
└──────────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────────┐
│  Layer 3 — Persistence & integrations                                │
│  entelix-persistence     (Postgres + Redis backends)                 │
│  entelix-mcp             (native JSON-RPC; MCP 1.5 streamable-http)  │
│  entelix-cloud           (Bedrock + Vertex + Foundry transports)     │
│  entelix-tools           (http_fetch, search adapters)               │
└──────────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────────┐
│  Layer 2 — State & control flow                                      │
│  entelix-graph           (StateGraph + Reducer + Checkpointer trait) │
│  entelix-session         (SessionGraph event log)                    │
│  entelix-memory          (Store + Embedder/Retriever traits)         │
└──────────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────────┐
│  Layer 1 — Composition & primitives                                  │
│  entelix-runnable        (Runnable trait + LCEL pipe)                │
│  entelix-prompt          (PromptTemplate + ChatPromptTemplate)       │
│  entelix-core            (IR, Codec, Transport, Tool, Auth, Hooks)   │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                       entelix (facade)
                  re-exports with feature flags
```

`entelix-core` is the DAG root — depends on no other entelix crate. Every other crate adds a vertical capability.

## Three coordinate systems

entelix's surface is shaped by **three independent design axes**:

### Axis 1 — Anthropic *managed-agent* shape (operational)

| Anthropic | entelix |
|---|---|
| Session (event log) | `entelix-session::SessionGraph` |
| Harness (stateless brain) | `entelix-core::Agent` + Codec + Transport |
| Hand (Tool execution) | `entelix-core::Tool` + `entelix-mcp` |
| Cattle, not pets | every component replaceable; only events persist |
| Lazy provisioning | MCP/cloud connections opened on demand |

Detail: `docs/architecture/managed-agents.md`.

### Axis 2 — LangChain LCEL (composition)

| LangChain | entelix |
|---|---|
| Runnable | `entelix-runnable::Runnable<I, O>` |
| LCEL `\|` pipe | `.pipe()` method (or `pipe!` macro) |
| ChatModel | `entelix-core::ChatModel` (Codec + Transport, Runnable<Vec<Message>, Message>) |
| Prompt | `entelix-prompt::PromptTemplate`, `ChatPromptTemplate` |
| OutputParser | `entelix-runnable::JsonOutputParser` (validation retries route through `entelix-core::OutputValidator` + `ChatModelConfig::validation_retries`, invariant 20) |
| Tool | `entelix-core::Tool` + `ToolToRunnableAdapter` |

Detail: `docs/architecture/runnable-and-lcel.md`.

### Axis 3 — LangGraph StateGraph (control flow)

| LangGraph | entelix |
|---|---|
| StateGraph | `entelix-graph::StateGraph<S>` |
| Reducer | `entelix-graph::Reducer<T>` + `Annotated<T, R>` |
| Conditional edges | `StateGraph::add_conditional_edges` |
| Send API | `entelix-graph::Send<T>` (fan-out) |
| Checkpointer | `entelix-graph::Checkpointer` trait |
| HITL `interrupt()` | `entelix-graph::interrupt()` + `Command::{Resume, Update, GoTo}` |
| HITL approval (in-band) | `entelix-agents::ChannelApprover` (synchronous wait on `oneshot`) |
| HITL approval (out-of-band) | `entelix::ApprovalLayer` + `Command::ApproveTool` typed resume (pause via `Error::Interrupted`, resume via `CompiledGraph::resume_with(Command::ApproveTool { tool_use_id, decision })`) |
| Per-call model / system override | `entelix-core::RunOverrides` (rides `ExecutionContext::extension`); `Agent::execute_with` convenience |
| Tool ambient state (task-locals, RLS `SET LOCAL`) | `entelix-core::tools::ToolDispatchScope` + `ScopedToolLayer` |
| Subgraphs | `StateGraph::compile()` 결과를 노드로 |
| Time travel | `Checkpointer::history()` + `Checkpointer::update_state()` |

Detail: `docs/architecture/state-graph.md`.

## Three-tier state model

This is the most important structural decision in entelix. State is **explicitly partitioned** into three tiers (invariant 3):

| Tier | Lifetime | Owner crate | Role |
|---|---|---|---|
| **StateGraph state** | per-thread, working | `entelix-graph` | working memory the graph mutates |
| **SessionGraph events** | per-thread, durable audit | `entelix-session` | append-only log of what happened |
| **Memory Store** | cross-thread, persistent | `entelix-memory` | facts that outlive any thread |

Alongside the three state tiers, the typed **AuditSink** channel
(invariant 18, ADR-0037) emits managed-agent lifecycle events
(`SubAgentInvoked` / `AgentHandoff` / `Resumed` / `MemoryRecall` /
`UsageLimitExceeded`) one-way into Tier 2's event log via
`SessionAuditSink`. The audit channel is *not* a fourth state tier
— it's a typed input into Tier 2 — but `session-and-memory.md`
groups it as a separate concern because it crosses the
managed-agent boundary that ordinary state mutations don't.

Detail: `docs/architecture/session-and-memory.md`.

## Data flow — single request

```
1. Client → POST /v1/threads/{thread_id}/runs   (sync) or
            GET  /v1/threads/{thread_id}/stream  (5-mode SSE)
                          │
2. entelix-server          extract tenant_id from header, attach to
                          ExecutionContext::tenant_id (mandatory,
                          invariant 11)
                          │
3. entelix-policy          rate-limit check (RateLimiter), PII redact
                          input (PiiRedactor), cost quota preflight
                          (QuotaLimiter)
                          │
4. entelix-session         load events for thread (or create)
                          replay → build initial StateGraph state
                          │
5. entelix-graph           CompiledGraph::execute(state, ctx)
                          - run node functions in order
                          - fold via per-field StateMerge reducers
                          - dispatch tools at tool-use steps
                          - persist Checkpoint per step
                          - record GraphEvent per step
                          │
6. entelix-core            for each model call:
                          - Codec::encode_request(IR) → wire JSON
                          - Transport::authorize() → POST → stream
                          - StreamAggregator: deltas → coherent turn
                          │
7. entelix-tools / entelix-mcp   for each tool call:
                          - lookup in ToolRegistry<D> (narrowed view
                            for sub-agents, ADR-0089)
                          - dispatch via Tool<D>::execute (built-ins,
                            #[tool] macro outputs) or McpToolAdapter
                            (MCP-published tools)
                          │
8. entelix-policy          on response: charge CostMeter (transactional,
                          inside Ok branch only — invariant 12),
                          PII redact output
                          │
9. entelix-otel            throughout: emit gen_ai.* spans + cache
                          token attributes; entelix.agent.run root
                          span scopes the whole tree (ADR-0057)
                          │
10. entelix-core           AuditSink emits sub-agent / handoff /
                          resume / memory-recall lifecycle events
                          (invariant 18, ADR-0037)
                          │
11. entelix-server         stream SSE frames to client (5-mode)
                          final state persisted to Checkpointer
```

## Crash recovery (cattle-not-pets)

Pod dies between step 7 and 8:
- Events 1..N already in `SessionGraph` Postgres backend (write-through).
- Latest Checkpoint already in `Checkpointer` Postgres backend (write-through per node).
- Client re-issues with same `thread_id` and `cursor`.
- New pod: `Agent::wake(thread_id)` →
  - `Persistence::load_events(thread_id)` → SessionGraph reconstructed
  - `Checkpointer::get_latest(thread_id)` → StateGraph state restored (avoid full event replay)
  - resume from event N+1
- Client sees no break; SSE stream re-attaches.

This combines Anthropic's `wake(sessionId) → getSession(id) → continue` with LangGraph's `Checkpointer`.

## Multi-agent — brain passes hand

Parent agent registers a sub-agent as a tool. The brain hands a
narrowed `ToolRegistry` to the sub-agent; the sub-agent runs as a
ReAct loop and reports a final string result back through the
parent's tool-call surface.

```rust
use entelix::{Subagent, create_react_agent};

// Sub-agent: narrowed view of the parent registry, exposed to
// the parent as a single tool.
let code_review = Subagent::builder(
        model.clone(),
        &parent_tools,
        "code_review",
        "Review the supplied PR and return a short summary.",
    )
    .filter(|t| t.metadata().name.starts_with("github_"))   // F7 — restrict hands
    .build()?
    .into_tool()?;

// Parent registry includes the sub-agent alongside its other tools.
let parent_tools = parent_tools.register(Arc::new(code_review))?;
let agent = create_react_agent(model, parent_tools)?;
// Sub-agent dispatch emits AuditSink::record_sub_agent_invoked
// (invariant 18, ADR-0037); replays reconstruct the lifecycle.
// Costs roll up through the inherited tower::Layer stack
// (PolicyLayer + CostMeter), no per-sub-agent rewiring.
```

## Why a workspace, not a single crate

ADR-0001 captures this decision in detail. Summary: single-crate + many features creates public-type bloat (every operator sees every type, even ones their feature set never touches). entelix splits along the only boundary that matters: **deployment surface**.

A user wanting Anthropic + Postgres + axum depends on `entelix` with `["postgres", "server"]` — pulls 5 crates. A user embedding entelix in a custom runtime depends on `entelix-core` directly — pulls 1.

## Reading order

1. `CLAUDE.md` — invariants + naming taxonomy + commands
2. `CHANGELOG.md` — `[1.0.0-rc.1]` catalogues every shipped
   surface
3. `docs/adr/0064-1.0-release-charter.md` — semver promise +
   public-API enforcement contract
4. `docs/architecture/managed-agents.md` — Axis 1 (Anthropic
   shape)
5. `docs/architecture/runnable-and-lcel.md` — Axis 2 (LCEL
   composition)
6. `docs/architecture/state-graph.md` — Axis 3 (StateGraph +
   StateMerge derive + add_contributing_node + add_send_edges)
7. `docs/architecture/session-and-memory.md` — three-tier state
   model + audit channel (invariant 18)
8. `docs/adr/` — 72 architecture decision records
9. `docs/public-api/<crate>.txt` — per-crate frozen surface
   (facade excluded by design)
