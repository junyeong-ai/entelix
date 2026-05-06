# StateGraph — control flow layer

## What this layer provides

LangGraph parity in Rust: a typed state machine with nodes, edges,
conditional routing, parallel fan-out, durable checkpoints,
human-in-the-loop interrupts, time travel, and subgraph
composition.

This is the most architecturally novel component of entelix.
**Take it seriously**: missteps propagate through the entire SDK.

## Anatomy

```
StateGraph<S>
    │
    ├─ State schema: S — Clone + Send + Sync + 'static
    │                    + StateMerge for declarative per-field merge
    ├─ Nodes: Runnable<S, S>  | Runnable<S, U> + merger
    │                          | Runnable<S, S::Contribution>
    ├─ Edges: static | conditional | send (parallel fan-out)
    └─ compile() → CompiledGraph<S>

CompiledGraph<S>
    │
    ├─ invoke(initial_state, ctx) → final S
    ├─ stream(initial_state, ctx, mode) → BoxStream of RunnableEvent
    ├─ resume(ctx) / resume_with(command, ctx) / resume_from(id, command, ctx)
    └─ Runnable<S, S>   ← so it can nest inside other graphs (subgraphs)
```

## Typed state schema

State is a typed Rust struct. The `#[derive(StateMerge)]` proc-macro
emits a per-field merge story plus a `<Name>Contribution` companion
struct for partial-update semantics:

```rust
use entelix::{Annotated, Append, Max, StateMerge};

#[derive(Clone, Debug, Default, StateMerge)]
pub struct ChatState {
    log: Annotated<Vec<String>, Append<String>>,   // accumulated
    score: Annotated<i32, Max<i32>>,                // best-of
    last_phase: String,                             // last-write-wins
}
```

The derive emits:

- `ChatStateContribution` — companion struct with `Option<T>` per field
  + `with_<field>(T)` builder methods that auto-wrap raw `T` into
  `Annotated::new(value, R::default())`.
- `impl StateMerge for ChatState`:
  - `type Contribution = ChatStateContribution`
  - `fn merge(self, update: Self) -> Self` — folds two full states
    via per-field reducers (used by `add_send_edges` parallel join).
  - `fn merge_contribution(self, c: Self::Contribution) -> Self` —
    folds a partial update; slots set to `None` keep the current
    value, slots set to `Some` merge through the per-field reducer.

Fields without `Annotated<T, R>` are treated as `Replace`
(last-write-wins).

## Reducers

Built-in `Reducer<T>` impls — the building block for `Annotated<T, R>`:

```rust
pub trait Reducer<T>: Send + Sync + 'static {
    fn reduce(&self, current: T, update: T) -> T;
}

pub struct Replace;                   // last-write-wins
pub struct Append<U>(PhantomData<U>); // Vec<U> concatenation
pub struct MergeMap<K, V>(...);       // HashMap right-bias union
pub struct Max<T>(PhantomData<T>);    // Ord::max
```

Custom reducers are user types. Conventional names follow the
naming taxonomy: `*Reducer` suffix.

## Building a graph — three node-registration shapes

```rust
use entelix::{StateGraph, RunnableLambda};

let graph = StateGraph::<ChatState>::new()
    // 1. Full-state replace — the runnable returns the next S.
    //    Lowest ceremony; no StateMerge requirement.
    .add_node("retrieve", retrieve_runnable)

    // 2. Contribution-style — the runnable returns S::Contribution.
    //    Slots set to None keep current; Some slots merge per reducer.
    //    Recommended for LangGraph TypedDict parity.
    .add_contributing_node("model", model_runnable)

    // 3. Delta + bespoke merger — the runnable returns U; the
    //    closure folds (S, U) → S. Best for cross-field invariants
    //    the declarative StateMerge can't express.
    .add_node_with("tool_dispatch", tool_dispatch_runnable,
        |mut state: ChatState, delta: ToolDelta| {
            // ... custom merge with invariant enforcement
            Ok(state)
        })

    .set_entry_point("retrieve")
    .add_edge("retrieve", "model")
    .add_conditional_edges(
        "model",
        |s: &ChatState| (if s.has_pending_tool() { "tool_dispatch" } else { "done" }).to_owned(),
        [("tool_dispatch", "tool_dispatch"), ("done", entelix::END)],
    )
    .add_edge("tool_dispatch", "model")    // loop back
    .add_finish_point("model")
    .compile()?;
```

## Send API — parallel fan-out

`add_send_edges` distributes one state to N parallel branches and
folds the results via `<S as StateMerge>::merge`. No per-call
reducer parameter — the state's `StateMerge` impl is the source of
truth.

```rust
graph.add_send_edges(
    "split",                                      // source node
    ["worker_a", "worker_b", "worker_c"],         // declared targets
    |s: &ChatState| {
        // selector returns (target_node, branch_state) pairs
        s.tasks.iter().map(|t| ("worker_a".into(), s.with_task(t))).collect()
    },
    "join",                                       // join target
);
```

Each branch runs concurrently via `try_join_all`. Branch results
merge into the pre-fan-out state through `S::merge`, then control
flows to the join node.

## Conditional edges + recursion limit (F6 mitigation)

```rust
let compiled = StateGraph::<ChatState>::new()
    /* ... */
    .with_recursion_limit(25)              // default 25
    .compile()?;

let result = compiled.invoke(state, &ctx).await;
match result {
    Err(Error::Provider { .. }) => /* upstream error */,
    Err(e) if matches!(e, Error::InvalidRequest(ref m) if m.contains("recursion limit")) => {
        // F6 hit — log + surface partial state from the latest checkpoint
    }
    Ok(s) => /* normal */,
}
```

The compile-time cap is **authoritative** — it pins F6 mitigation at the operator's design-time choice. Per-call `entelix::RunOverrides::with_max_iterations(n)` can *lower* the effective limit (`min(compile_time_cap, n)`) for one specific dispatch, but never raise it (ADR-0069):

```rust
let ctx = ExecutionContext::new()
    .add_extension(RunOverrides::new().with_max_iterations(10));
compiled.invoke(state, &ctx).await?;  // effective cap = min(25, 10) = 10
```

## Checkpointer — durable state

```rust
#[async_trait]
pub trait Checkpointer<S>: Send + Sync + 'static
where
    S: Clone + Send + Sync + 'static,
{
    async fn put(&self, checkpoint: Checkpoint<S>) -> Result<()>;
    async fn latest(&self, key: &ThreadKey) -> Result<Option<Checkpoint<S>>>;
    async fn by_id(&self, key: &ThreadKey, id: &CheckpointId) -> Result<Option<Checkpoint<S>>>;
    async fn history(&self, key: &ThreadKey, limit: usize) -> Result<Vec<Checkpoint<S>>>;
    async fn update_state(
        &self,
        key: &ThreadKey,
        parent_id: &CheckpointId,
        new_state: S,
    ) -> Result<CheckpointId>;
}

#[non_exhaustive]
pub struct Checkpoint<S>
where
    S: Clone + Send + Sync + 'static,
{
    pub id: CheckpointId,
    pub tenant_id: TenantId,               // invariant 11 — mandatory
    pub thread_id: String,
    pub parent_id: Option<CheckpointId>,   // time-travel parent
    pub step: usize,                       // monotonic counter within thread
    pub state: S,
    pub next_node: Option<String>,         // None when the graph terminated
    pub timestamp: DateTime<Utc>,
}
```

The checkpoint carries `tenant_id` + `thread_id` directly (the
`ThreadKey` newtype is a borrowed view, not a stored field) so
the persistence-side row layout indexes naturally. `put` takes
the checkpoint by value — the addressing comes from the
checkpoint's own fields, not a separate key argument.

Implementations:
- `InMemoryCheckpointer<S>` — in-process default, in `entelix-graph`
- `PostgresCheckpointer<S>` — production, in `entelix-persistence`,
  enforces row-level security via `current_setting('entelix.tenant_id', true)`
  (ADR-0041)
- `RedisCheckpointer<S>` — fast resume, in `entelix-persistence`

## Frequency policy (F8 mitigation)

By default, the runner writes a checkpoint **after each successful
node**. Override at compile time:

```rust
use entelix::CheckpointGranularity;

let compiled = StateGraph::<ChatState>::new()
    .with_checkpointer(Arc::new(pg_checkpointer))
    .with_checkpoint_granularity(CheckpointGranularity::PerNode)  // default
    /* or */                          // .with_checkpoint_granularity(CheckpointGranularity::Off)
    .compile()?;
```

`Off` skips writes entirely (ephemeral graphs); `PerNode` matches
the F8 mitigation. The `Checkpointer` is still wired (a downstream
API may require it) — the granularity knob just gates the write.

## Human-in-the-loop — `interrupt()` and `Command`

```rust
use entelix::{interrupt, Command};

let approval_node = RunnableLambda::new(|state: ChatState, _ctx| async move {
    if state.requires_approval() {
        // Pauses execution; the executor catches the Interrupted
        // error, persists a checkpoint pointing at this node, and
        // returns control to the host.
        return Err(interrupt(serde_json::json!({
            "kind": "approval_required",
            "context": state.last_message(),
        })));
    }
    Ok(state.with_approved(true))
});
```

Resuming:

```rust
let compiled = graph.with_checkpointer(Arc::new(pg_checkpointer)).compile()?;

// First run — interrupts, returns Error::Interrupted { payload }
let outcome = compiled.invoke(initial, &ctx).await;

// Operator approves out-of-band, then resumes:
compiled.resume_with(Command::Resume, &ctx).await?;
// or: Command::Update(new_state)  — re-run from the same checkpoint with patched state
// or: Command::GoTo(node_name)    — jump to a specific node
```

`interrupt_before(["node"])` and `interrupt_after(["node"])`
schedule pauses without code-level `interrupt()` calls (ADR-0028).

## Time travel

```rust
// History is a list of checkpoint ids in mint-time order.
let history = pg_checkpointer.history(&key, 50).await?;

// Pick a past checkpoint and patch state.
let target_id = &history[3];
let new_id = pg_checkpointer
    .update_state(&key, target_id, ChatState { user_input: "rephrased question".into(), ..base })
    .await?;

// Re-execute from that checkpoint.
compiled.resume_from(&new_id, Command::Resume, &ctx).await?;
```

## Subgraphs

A `CompiledGraph<S>` implements `Runnable<S, S>` — so it nests
inside another graph as a node:

```rust
let translator: CompiledGraph<TranslateState> = StateGraph::new()
    .add_node("translate", translate_runnable)
    /* ... */
    .compile()?;

let outer = StateGraph::<OuterState>::new()
    .add_node("translator_subgraph", translator)   // subgraph as a node
    .add_node("review", review_runnable)
    /* ... */;
```

State-shape mismatch? Wrap the subgraph in a `RunnableLambda` that
projects between the two state types.

## Cycles — what's allowed

- ReAct loop (model ↔ tool_dispatch ↔ model) — yes, via conditional edges
- Direct self-edge — yes
- A → B → A — yes
- Unbounded recursion — caught by `with_recursion_limit` (F6 mitigation)

## Streaming

`compiled.stream(initial, mode, &ctx)` returns a stream of
`RunnableEvent`. `mode` is `StreamMode`:

| Mode | What it yields | Use case |
|---|---|---|
| `Values` | full state after each step | UI showing "current state" |
| `Updates` | each per-node update | log of changes |
| `Messages` | only message tokens | chat UI |
| `Debug` | internal trace events | dev tooling |
| `Events` | named lifecycle events | observability |

## DAG validation

At `compile()` time:
- Entry point exists.
- Every conditional edge maps to declared nodes.
- Every send edge target is a declared node.
- A node may not have both a static edge and a conditional edge
  (and vice-versa across all edge kinds).
- Join target of a send edge is registered or `END`.

Errors surface as `Error::InvalidRequest` with diagnostic detail.

## Performance considerations

- DAG analysis is built on `HashMap` / `HashSet` — no graph
  library dependency.
- Per-node overhead: 1 atomic counter increment (recursion limit) +
  1 checkpoint write (default per-node — `CheckpointGranularity::Off`
  skips it for ephemeral graphs).
- Checkpoint writes are awaited sequentially after each node so the
  next node sees a consistent baseline; the `PostgresCheckpointer`
  uses one tenant-scoped tx per write.

## Cross-references

- ADR-0006 — Runnable + StateGraph 1.0 spine.
- ADR-0010 — naming taxonomy (`*Reducer`, `*Checkpointer`).
- ADR-0028 — `interrupt_before` / `interrupt_after` deadlock avoidance on resume.
- ADR-0041 — Postgres row-level security.
- ADR-0059 — `StateMerge` trait + `#[derive(StateMerge)]` proc-macro.
- ADR-0061 — `StateMerge::Contribution` + `add_contributing_node`
  (supersedes the `add_reducing_node` portion of ADR-0059).
- ADR-0062 — `add_send_edges` uses `StateMerge::merge`.
- `crates/entelix/examples/16_state_merge_pipeline.rs` — full
  Phase-9 surface end-to-end.
