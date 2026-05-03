# Migration Guide — LangGraph (Python) → entelix

> Audience: existing LangGraph users porting an agent to Rust.
> Goal: cover the 80% of patterns end-to-end. The remaining 20%
> (custom `Annotated[...]` reducers in user-defined classes,
> store API specifics, etc.) follow the same translation rules.

## At a glance

| Concept | LangGraph (Python) | entelix (Rust) |
|---|---|---|
| State type | `TypedDict` | `#[derive(Clone, ...)] struct` |
| Graph builder | `StateGraph(StateClass)` | `StateGraph::<StateClass>::new()` |
| Add a node | `graph.add_node("plan", planner_fn)` | `graph.add_node("plan", planner_runnable)` |
| Add a delta-style node | `Annotated[List, operator.add]` | `graph.add_node_with("plan", planner, merger)` |
| Edge | `graph.add_edge("plan", "execute")` | `graph.add_edge("plan", "execute")` |
| Conditional edge | `graph.add_conditional_edges("review", route_fn, mapping)` | `graph.add_conditional_edges("review", \|s\| ..., mapping)` |
| Compile | `graph.compile()` | `graph.compile()?` |
| Invoke | `graph.invoke(state, config={"thread_id": "x"})` | `graph.invoke(state, &ctx_with_thread_id)` |
| Stream | `for chunk in graph.stream(state, stream_mode="values")` | `let mut s = graph.stream(state, StreamMode::Values, &ctx)?;` |
| Checkpointer | `MemorySaver()` / `PostgresSaver` | `InMemoryCheckpointer::new()` / `PostgresCheckpointer::connect(url)?` |
| Resume | `graph.invoke(None, config)` | `graph.resume(thread_id, &ctx).await?` |
| Interrupt | `interrupt(payload)` | `interrupt(payload)?` |
| Tools | `@tool` decorator | `impl Tool for X { ... }` |
| ReAct recipe | `create_react_agent(model, tools)` | `create_react_agent(model, tools)?` |
| HTTP server | `LangServe` | `entelix::AgentRouterBuilder` |

## State

LangGraph state is a `TypedDict` (open dict). entelix state is a
typed `struct` you `#[derive(Clone, Default, StateMerge)]` on. The
bound is `Clone + Send + Sync + 'static` — practically every
business struct meets this. `StateMerge` is the entelix
counterpart to LangGraph's `Annotated[T, reducer]` per-field
auto-merge.

```python
# LangGraph
class AgentState(TypedDict):
    messages: Annotated[list[Message], operator.add]
    iteration: int
    last_phase: str
```

```rust
// entelix
use entelix::{Annotated, Append, Max, Replace, StateMerge};

#[derive(Clone, Default, StateMerge)]
struct AgentState {
    messages: Annotated<Vec<Message>, Append<Message>>,
    iteration: Annotated<u32, Max<u32>>,
    last_phase: String,                        // plain field = Replace
}
```

The derive macro emits a companion struct `AgentStateContribution`
with one `Option<T>` per field plus `with_<field>` builder methods
that auto-wrap raw `T` into `Annotated::new(value, R::default())`.
Nodes return `AgentStateContribution` and the framework folds it
in through `S::merge_contribution` — the same partial-return shape
as LangGraph's "return only the keys you wrote" idiom.

## Adding nodes

LangGraph nodes are functions that return a partial state dict:

```python
def planner(state: AgentState) -> dict:
    return {"messages": [...], "iteration": state["iteration"] + 1}
    # `last_phase` not returned → preserved
```

entelix has three equivalent shapes (pick the one that fits the node):

### Contribution-style (`add_contributing_node`) — recommended LangGraph parity

The runnable returns `S::Contribution`. Slots set to `Some` merge
through the per-field reducer; slots left as `None` keep the
current value. Direct equivalent of LangGraph's partial-dict return.

```rust
use entelix::RunnableLambda;

let planner = RunnableLambda::new(|s: AgentState, _ctx| async move {
    // Touched: messages + iteration. last_phase left as None → kept.
    Ok(AgentStateContribution::default()
        .with_messages(vec![/* ... */])
        .with_iteration(s.iteration.value + 1))
});
graph.add_contributing_node("plan", planner);
```

### Full-state replace (`add_node`)

The closure returns the *full* next state. Lower ceremony when the
node owns most of the state. No `StateMerge` requirement.

```rust
let planner = RunnableLambda::new(|mut s: AgentState, _ctx| async move {
    s.messages.value.push(/* ... */);
    s.iteration.value += 1;
    Ok::<_, _>(s)
});
graph.add_node("plan", planner);
```

### Delta + bespoke merger (`add_node_with`)

The runnable returns a delta of arbitrary type `U`; a merger
closure combines `(state, delta) → state`. Use when the merge
logic is graph-specific or has cross-field invariants the
declarative `StateMerge` shape can't express.

```rust
graph.add_node_with("plan", planner_delta_runnable,
    |mut state: AgentState, delta: PlannerDelta| {
        // Custom merge — e.g. enforce iteration cap, validate
        // message order, etc.
        state.messages.value.extend(delta.new_messages);
        state.iteration.value += delta.increment;
        Ok(state)
    });
```

## Parallel fan-out (Send) — `add_send_edges`

LangGraph's `Send(...)` distributes a single state to N parallel
branches and folds the results. entelix's `add_send_edges` does
the same and uses `S::merge` automatically — no per-call reducer
parameter:

```python
# LangGraph
def route(state):
    return [Send("worker", {"task": t}) for t in state["tasks"]]
graph.add_conditional_edges("plan", route)
```

```rust
// entelix — S::merge wires the join automatically because of derive(StateMerge)
graph.add_send_edges(
    "plan",
    ["worker"],
    |s: &AgentState| s.tasks.value.iter()
        .map(|t| ("worker".into(), AgentState { task: t.clone(), ..s.clone() }))
        .collect(),
    "join",                          // join target after the fold
);
```

## Reducers

| LangGraph | entelix |
|---|---|
| `Annotated[list, operator.add]` | `Annotated<Vec<T>, Append<T>>` |
| `Annotated[dict, lambda a, b: {**a, **b}]` | `Annotated<HashMap<K, V>, MergeMap<K, V>>` |
| (no built-in) | `Annotated<T, Max<T>>` (any `T: Ord`) |
| Default replace | plain field type (no `Annotated` wrapper) |

The reducer's `Default` impl is what the contribution builder
auto-supplies, so `with_messages(vec![...])` is enough — no
manual `Annotated::new(...)` boilerplate at the call site.

For stateful reducers (configuration-carrying), implement
`StateMerge` manually: the trait has two methods — `merge(self, update: Self) -> Self`
(parallel-branch join) and `merge_contribution(self, c: Self::Contribution) -> Self`
(per-node fold). The companion `Contribution` struct is whatever
shape the operator finds clearest.

## Conditional edges

```python
# LangGraph
graph.add_conditional_edges(
    "review",
    lambda s: "loop" if s["iteration"] < 3 else "done",
    {"loop": "plan", "done": "answer"},
)
```

```rust
// entelix
graph.add_conditional_edges(
    "review",
    |s: &AgentState| {
        if s.iteration < 3 { "loop".to_owned() } else { "done".to_owned() }
    },
    [("loop", "plan"), ("done", "answer")],
);
```

Selector returns `String` (entelix uses owned strings; allocation
is cheap and avoids lifetime gymnastics on the `&S` borrow).

## Streaming

LangGraph's `stream_mode={"values","updates","messages","debug","events"}`
maps 1:1 to `entelix::StreamMode::{Values,Updates,Messages,Debug,Events}`.

```python
# LangGraph
for chunk in graph.stream(state, config, stream_mode="updates"):
    print(chunk)
```

```rust
// entelix
let mut stream = graph.stream(state, StreamMode::Updates, &ctx).await?;
while let Some(chunk) = stream.next().await {
    println!("{:?}", chunk?);
}
```

Chunks come typed (`StreamChunk<S>` enum) — pattern-match
on `Value` / `Update { node, value }` / `Message(StreamDelta)` /
`Debug(DebugEvent)` / `Event(RunnableEvent)`.

## Checkpointer + resume

```python
# LangGraph
saver = MemorySaver()  # or PostgresSaver.from_conn_string(url)
graph = builder.compile(checkpointer=saver)
graph.invoke(state, {"configurable": {"thread_id": "x"}})
# … later, on a different process …
graph.invoke(None, {"configurable": {"thread_id": "x"}})  # resumes
```

```rust
// entelix
let cp: Arc<dyn Checkpointer<S>> = Arc::new(InMemoryCheckpointer::<S>::new());
// or: Arc::new(PostgresCheckpointer::connect(url).await?)
let graph = builder.with_checkpointer(cp.clone()).compile()?;
let ctx = ExecutionContext::new().with_thread_id("x");
graph.invoke(state, &ctx).await?;
// … fresh process, same checkpointer …
let graph = builder.with_checkpointer(cp).compile()?;  // identical builder
graph.resume("x", &ctx).await?;
```

The harness is stateless — the `CompiledGraph` instance is
disposable; only the `Checkpointer` carries durable state across
processes (entelix invariant 2). See `examples/11_durable_session.rs`.

## Interrupt / Human-in-the-loop

```python
# LangGraph
def review(state):
    if state["approved"] is None:
        return interrupt({"options": ["approve", "reject"]})
    return state

# resume:
graph.invoke(Command(resume={"approved": True}), config)
```

```rust
// entelix
let review = RunnableLambda::new(|s: State, _ctx| async move {
    if s.approved.is_none() {
        return interrupt(serde_json::json!({"options": ["approve", "reject"]}));
    }
    Ok(s)
});

// resume:
graph.resume_with(thread_id, Command::Update(approved_state), &ctx).await?;
```

`Command::{Resume, Update(S), GoTo(node)}` mirrors LangGraph's
`Command(resume=...)` family. See `examples/04_hitl.rs`.

### Tool-dispatch approval (LangGraph has no direct equivalent)

LangGraph operators wanting *tool-call-level* HITL (approve / deny
specific tool dispatches before they fire) typically inline an
interrupt inside the tool node body:

```python
# LangGraph (operator-built pattern)
def tool_node(state):
    for call in state.pending_tool_calls:
        if not call.approved:
            return interrupt({"tool": call.name, "input": call.input})
        # … dispatch
    return state
```

entelix surfaces tool-dispatch approval as a typed primitive
(`Approver` trait + `ApprovalLayer` Tower middleware), wired
once on `ReActAgentBuilder::with_approver`:

```rust
// entelix
let agent = ReActAgentBuilder::new(model, tools)
    .with_sink(parent_sink)
    .with_approver(my_approver)  // auto-wires ApprovalLayer
    .build()?;
```

Two flows compose cleanly: `Approver::decide` returns
`Approve` / `Reject` for instant decisions (in-band), or
`AwaitExternal` to pause via `Error::Interrupted` and resume
through `Command::ApproveTool { tool_use_id, decision }` on
`CompiledGraph::resume_with` (out-of-band). See
`examples/18_tool_approval.rs` and ADRs 0070 (layer) +
0071 (pause-and-resume mechanism) + 0072 (typed resume primitive).

## Per-call overrides — `RunOverrides`

LangGraph passes per-invocation config through the `config`
argument; entelix uses `ExecutionContext::extension::<RunOverrides>()`
which reaches `ChatModel::complete_full` / `stream_deltas` and
`CompiledGraph` recursion-limit clamping uniformly:

```python
# LangGraph
graph.invoke(state, config={"configurable": {"model": "haiku"}})
```

```rust
// entelix
let overrides = RunOverrides::new()
    .with_model("claude-3-5-haiku-latest")
    .with_max_iterations(8);
agent.execute_with(state, overrides, &ctx).await?;
```

The compile-time recursion limit stays authoritative — operators
can lower per-call but never raise (F6 mitigation preserved).
ADR-0069.

## Tools

LangGraph: `@tool` decorator on a function. entelix: implement the
[`Tool`](../../crates/entelix-core/src/tools/tool.rs) trait.

```python
@tool
def lookup(record_id: str) -> str:
    """Look up a record by id."""
    return fetch(record_id)
```

```rust
struct LookupTool;

#[async_trait]
impl Tool for LookupTool {
    fn name(&self) -> &str { "lookup" }
    fn description(&self) -> &str { "Look up a record by id." }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "required": ["record_id"],
            "properties": { "record_id": { "type": "string" } }
        })
    }
    async fn execute(&self, input: serde_json::Value, ctx: &ExecutionContext)
        -> Result<serde_json::Value>
    {
        let id = input["record_id"].as_str().unwrap_or("");
        Ok(serde_json::Value::String(fetch(id, ctx).await?))
    }
}
```

The `Tool` trait is intentionally *minimal* — `name`,
`description`, `input_schema`, `execute`. No `prepare`, no
`cleanup`, no back-channels (entelix invariant 4).

## Pre-built agent recipes

| LangGraph | entelix |
|---|---|
| `create_react_agent(model, tools)` | `create_react_agent(model, tools)?` |
| `create_supervisor(...)` | `create_supervisor_agent(router, agents)?` |
| (no exact equivalent) | `create_hierarchical_agent(router, teams)?` |

## HTTP server

LangServe → [`entelix::AgentRouterBuilder`]. Mount any
`CompiledGraph<S>`:

```rust
use entelix::AgentRouterBuilder;

let agent = builder.compile()?;
let service = AgentRouterBuilder::new(agent)
    .checkpointer(cp)
    .tenant_header("X-Tenant-Id")?
    .build();
let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
axum::serve(listener, service.into_router()).await?;
```

Endpoints:

- `POST /v1/threads/{thread_id}/runs` — synchronous invoke.
- `GET /v1/threads/{thread_id}/stream?mode=values|updates|messages|debug|events` — SSE.
- `POST /v1/threads/{thread_id}/wake` — resume from checkpoint.
- `GET /v1/health` — liveness.

See `examples/14_serve_agent.rs` for an end-to-end demo.

## What entelix has that LangGraph doesn't

- **Compile-time state typing** — your state struct is checked at
  compile time across every node and edge. Node return-type
  mismatches don't reach production.
- **Codec × transport sparse matrix** — unify Anthropic /
  OpenAI / Gemini / Bedrock / Vertex / Foundry behind one IR
  (`ModelRequest`). Switching providers is a one-line
  `ChatModel::new(codec, transport, ...)` change.
- **MCP first-class with all server-initiated channels** —
  `Roots` + `Elicitation` + `Sampling`. Wire your `ChatModel` as
  the sampling backend in 5 lines via `entelix-mcp-chatmodel`'s
  `ChatModelSamplingProvider`. LangGraph requires external
  adapters for each.
- **Multi-tenant primitives** — `entelix::{RateLimiter,
  PiiRedactor, CostMeter, QuotaLimiter, TenantPolicy,
  PolicyRegistry}`. LangGraph leaves these to the host.
- **Postgres row-level security on every persistence backend** —
  `PostgresStore` / `PostgresCheckpointer` / `PostgresSessionLog`
  + `PgVectorStore` + `PgGraphMemory` all enforce
  `tenant_isolation` policy via `current_setting('entelix.tenant_id', true)`.
  Defense-in-depth on top of the application-level `Namespace`
  scoping.
- **OpenTelemetry GenAI semconv 0.32** — `entelix::OtelLayer`
  ships full `gen_ai.*` attribute coverage including cache token
  telemetry (`gen_ai.usage.cached_input_tokens` /
  `cache_creation_input_tokens` / `reasoning_tokens`). Tool I/O
  capture mode (`Off` / `Truncated{4096}` / `Full`) gates payload
  size. `Agent::execute` opens `entelix.agent.run` root span so
  trace UI shows agent → model → tool as one tree.
- **Typed audit channel** — `entelix::AuditSink` with 4
  `record_*` verbs (`sub_agent_invoked` / `agent_handoff` /
  `resumed` / `memory_recall`). `entelix::SessionAuditSink`
  maps onto `GraphEvent` so replays reconstruct managed-agent
  lifecycle without re-running the dispatch path.
- **Five first-class memory patterns**: `BufferMemory` /
  `SummaryMemory` / `EntityMemory` / `SemanticMemory<E, V>` /
  `EpisodicMemory<E, V>` (time-ordered episode log). Each composes
  over the `Store<V>` trait so swapping the persistence layer
  doesn't touch the recipe.
- **`GraphMemory<N, E>`** — typed nodes + timestamped edges with
  18 trait methods (CRUD, traversal, list, count, prune,
  bulk insert via `add_edges_batch`). `PgGraphMemory` folds BFS
  traversal into one `WITH RECURSIVE` round-trip and bulk insert
  into one `INSERT … SELECT FROM UNNEST(…)` call.
- **Distributed session lock** — `entelix::with_session_lock`
  prevents concurrent writes to the same `thread_id` across pods.
  LangGraph's `PostgresSaver` doesn't enforce this.

## Things to watch out for

- **Async everywhere** — Python LangGraph mixes sync + async. Every
  entelix entry point is async; you'll need a Tokio runtime.
- **State must be `Clone`** — checkpointing snapshots state by
  cloning. Wrap large internal data in `Arc` if cloning is costly.
- **`#[non_exhaustive]` enums** — `Error`, `StopReason`, etc. add a
  trailing wildcard arm in your match if you want exhaustive
  handling.
- **Recursion limit** — default 25 (F6 mitigation). Override per
  graph with `.with_recursion_limit(n)` if your loop legitimately
  exceeds this.
- **Naming taxonomy** — `*Codec`, `*Transport`, `*Provider`,
  `*Manager`, `*Limiter`, `*Meter`, `*Redactor`. See
  `docs/adr/0010-naming-taxonomy.md`.

## See also

- `docs/architecture/managed-agents.md` — the Anthropic shape
  entelix mirrors.
- `docs/architecture/state-graph.md` — the StateGraph internals.
- `docs/migrations/rig.md` — for users porting from `rig`.
- `examples/` — 17 working examples; 03/04/05/11 cover the
  StateGraph patterns above; **16_state_merge_pipeline** shows
  `derive(StateMerge)` + `add_contributing_node` +
  `add_send_edges` end-to-end; **17_mcp_sampling_provider**
  wires `ChatModelSamplingProvider` against a stub Codec/Transport.
