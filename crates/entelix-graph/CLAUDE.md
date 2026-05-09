# entelix-graph

Control-flow contract (invariant 8). LangGraph parity surface.

## Surface

- **`StateGraph<S>`** — typed state, builder methods: `add_node` (full state) / `add_node_with(name, runnable, merger)` (delta + bespoke merger) / `add_contributing_node` (declarative per-field merge via `S: StateMerge`) / `add_edge` / `add_conditional_edges` / `add_send_edges` (parallel fan-out, joins via `S::merge`) / `set_entry_point` / `set_finish_point`. Verb prefixes follow the naming exception (`add_*` for collection inserts, `set_*` for named role).
- **`CompiledGraph<S>`** — `Runnable<S, S>` impl (`invoke` / `stream`) + `resume(ctx)` / `resume_with(command, ctx)` / `resume_from(checkpoint_id, ctx)`. Enforces `recursion_limit` (default 25) and raises `Error::InvalidRequest` on overflow.
- **`Reducer<T>` trait** — pure state-merge function. Built-ins: `Replace`, `Append`, `MergeMap`, `Max`. `add_node_with` lets a node return a typed delta merged through the reducer.
- **`Checkpointer<S>` trait** + `InMemoryCheckpointer<S>` — write-after-each-node persistence. Verb-family read methods (`get_latest` / `get_by_id` / `list_history` per `.claude/rules/naming.md`). `CheckpointGranularity::{PerNode (default), Off}`. Postgres / Redis impls live in `entelix-persistence`.
- **HITL** — pause primitive lives in `entelix-core::interruption` (`interrupt(payload)` / `interrupt_with(kind, payload)` raise `Error::Interrupted { kind: InterruptionKind, payload }`). This crate's `interrupt_before(nodes)` / `interrupt_after(nodes)` schedule pauses with `kind: InterruptionKind::ScheduledPause { phase, node }`. Resume via `CompiledGraph::resume_with(Command, &ctx)` where `Command<S>` is `Resume` / `Update(S)` / `GoTo(node)` / `ApproveTool { tool_use_id, decision }` (typed approval primitive).

## Crate-local rules

- **`#[derive(StateMerge)]` is the canonical way to declare per-field reducers.** The `Reducer<T>` trait + `Append` / `Max` / `MergeMap` / `Replace` built-ins exist for advanced cases (custom dynamic reducer composition, runtime-selected merge strategies). New operators reach for the derive first; the bare trait is for the 5% case where the per-field reducer is computed at runtime rather than declared at compile time.
- Reducers must be pure (no IO, no `Send` futures). The reduce function runs synchronously in the dispatch loop.
- Conditional edges return `String` keys; the routing table is `[(key, target_node)]`. Keep keys `snake_case`.
- New checkpointer impls: include namespace-collision tests at the persistence layer (invariant 13). Reference suite under `entelix-memory/tests/namespace_collision.rs`.
- Nodes returning `Result<S, Error>` (full state) and nodes returning `Result<Delta, Error>` (delta + reducer) coexist; pick by whether the node owns the whole state shape.

## Forbidden

- Ad-hoc `loop { … if done { break } }` user code in agent recipes — model it as `StateGraph` with conditional edge to `END` (invariant 8).
- Holding any lock across `.await` inside a node (lock ordering rule).
- Skipping `recursion_limit` because "this graph won't loop" — the cap is the only safety against cyclic dispatch.

