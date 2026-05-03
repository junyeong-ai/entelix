# ADR 0062 — `add_send_edges` uses `StateMerge::merge` (parallel-join coherence with the per-field reducer story)

**Status**: Accepted (closes the parallel-join positioning hole that ADR-0061 introduced)
**Date**: 2026-05-01
**Decision**: Phase 9 of the post-7-차원-audit roadmap (fifth sub-slice — `StateMerge::merge` actually wired into its documented use site)

## Context

ADR-0061 declared two distinct merge axes on `StateMerge`:

> - `merge(Self, Self) -> Self` (used by parallel-branch joins
>   via `add_send_edges`, where two branches each produce a
>   complete `S` and the dispatcher needs to combine them).
> - `merge_contribution(Self, Self::Contribution) -> Self` (used
>   by single-node `add_contributing_node`).

The `merge_contribution` half landed correctly in slice 53. The
`merge` half — claimed to back `add_send_edges` parallel-join —
*did not actually wire into the dispatcher*. `add_send_edges`
still took an explicit `R: Reducer<S>` parameter and the
`execute_send_edge` body called `send.reducer.reduce(folded, branch)`.
The `merge(Self, Self)` trait method existed only as
documentation: every operator using `add_send_edges` had to write
their own `Reducer<S>` impl by hand, even when their state
already advertised its merge story via
`#[derive(StateMerge)]`. Two surfaces for the same job, one
documented and one real.

This slice closes the gap: `add_send_edges` consumes `S::merge`
directly and the explicit-reducer parameter goes away.

## Decision

`StateGraph::add_send_edges` becomes:

```rust
pub fn add_send_edges<F, I, T>(
    self,
    from: impl Into<String>,
    targets: I,
    selector: F,
    join: impl Into<String>,
) -> Self
where
    F: Fn(&S) -> Vec<(String, S)> + Send + Sync + 'static,
    I: IntoIterator<Item = T>,
    T: Into<String>,
    S: StateMerge,
```

The dispatch loop in `execute_send_edge` folds branch results
through `<S as StateMerge>::merge`. The `Reducer<S>` parameter is
removed; the `Reducer<S>` trait itself stays (it remains the
per-slot building block inside `Annotated<T, R>` and the merge-
building blocks the derive macro emits) but no longer participates
in send-edge wiring.

### How dispatch reaches `S::merge` without invading `CompiledGraph`'s bounds

A naive "add `S: StateMerge` to `CompiledGraph<S>`" change forces
*every* graph (including ones that never use send edges) onto the
trait bound. Using a trait-object merger keeps the bound localised
to the `add_send_edges` builder method:

```rust
pub type SendMerger<S> = Arc<dyn Fn(S, S) -> S + Send + Sync>;

pub struct SendEdge<S> {
    targets: Vec<String>,
    targets_set: HashSet<String>,
    pub selector: SendSelector<S>,
    pub merger: SendMerger<S>,
    pub join: String,
}
```

`add_send_edges` constructs the merger as
`Arc::new(<S as StateMerge>::merge) as SendMerger<S>`. The bound
is checked at the builder call site (where the operator adds the
edge); `SendEdge<S>` and `CompiledGraph<S>` carry no `StateMerge`
bound at all. Existing graphs that never call `add_send_edges`
keep working unchanged — including state types that don't
implement `StateMerge`.

### Why no fallback "explicit reducer" overload

A sibling `add_send_edges_with(reducer)` keeping the explicit
shape would make the API choice "which surface should I reach for
when I have a state that derives StateMerge?" require a per-call
decision. Two surfaces, two ways to make the same edge —
duplicate flexibility costs ergonomic clarity. The standing
project instruction ("처음부터 이렇게 설계된 것처럼 흔적 0") says
delete the old and ship the right shape outright; there's no
legitimate use-case for the explicit-reducer form once
`StateMerge` is the canonical merge surface.

State types with cross-field invariants enforced at merge time
already have the escape hatch: implement `StateMerge` manually
(documented in ADR-0061) so `<S as StateMerge>::merge` does
exactly what the operator wants.

### Why `<S as StateMerge>::merge` instead of `S::merge`

Disambiguation: `S` may have other inherent or trait `merge`
methods. The fully-qualified path forces compiler resolution to
the `StateMerge` impl regardless. Reads as a one-character
nuance that prevents a class of confusing
"why is the wrong merge running?" debugging sessions.

### Tests

- 4 existing send-edge integration tests
  (`crates/entelix-graph/tests/send_edges.rs`) updated to use
  `derive(StateMerge)` on the test state struct (`State` now has
  `log: Annotated<Vec<String>, Append<String>>`) with the explicit
  reducer parameter removed from every `add_send_edges` call. The
  test assertions remain identical (`log` accumulates branch
  contributions); the change exercises that the framework wires
  the right reducer automatically.

## Consequences

✅ The parallel-join story is now coherent end-to-end. Operators
who derive `StateMerge` get send-edge folding for free; adding a
new state field automatically participates in the join shape
without any send-edge call-site edits.
✅ `merge(Self, Self)` is no longer a documentation-only trait
method — it's the dispatch surface for every parallel-fan-out
join.
✅ One canonical merge surface across the whole `StateGraph`
shape: `add_contributing_node` calls `merge_contribution`,
`add_send_edges` calls `merge`. Both flow from the same
`#[derive(StateMerge)]`.
✅ `Reducer<S>` trait stays as the per-slot building block
(`Append`, `Max`, `MergeMap`, `Replace`); operators don't lose
the per-slot composition primitive.
✅ Localised trait-object pattern: `SendEdge<S>` has no
`StateMerge` bound — `CompiledGraph<S>` and the wider dispatch
machinery stay generic over any `S: Clone + Send + Sync + 'static`,
and only the builder call site validates `S: StateMerge`. No
build-graph cascade.
❌ Public-API baseline drift on `entelix-graph`
(`add_send_edges` signature change, `SendEdge::new` signature
change, new `SendMerger<S>` type alias) and `entelix` (facade
re-export). Refrozen.
❌ The explicit `Reducer<S>` overload is gone. State types with
custom join semantics implement `StateMerge` manually instead.

## Alternatives considered

1. **Keep the explicit-reducer signature, document `StateMerge::merge`
   as informational** — codifies the documented-but-unused trait
   method; preserves the ADR-0061 inconsistency. Rejected.
2. **Add `S: StateMerge` to `CompiledGraph<S>` and call
   `S::merge` directly in `execute_send_edge`** — forces every
   graph (including ones that don't use send edges) onto the
   `StateMerge` bound, costing a derive macro on every state
   type. Rejected.
3. **Sibling `add_send_edges_with(reducer)` overload alongside
   the new shape** — two surfaces, two ways to make the same
   edge; duplicate flexibility costs ergonomic clarity. Rejected
   per the standing project instruction.
4. **Synchronous-merge typing — `S::merge` returns `Result<S>`** —
   parallel join might want to bail on unmergeable branches. The
   trait stays infallible to match `Reducer::reduce` and to keep
   the dispatcher's fold loop straight; merge logic that needs
   to fail surfaces it as a manual impl that panics on the
   unmergeable case (or operators handle it pre-fan-out via the
   selector closure rejecting branches early). Rejected for now;
   reserved if real demand surfaces.
5. **`Box<dyn FnMut(S, S) -> S>` instead of
   `Arc<dyn Fn(S, S) -> S>`** — a `FnMut` would let stateful
   mergers update across calls. The dispatcher already folds in
   a single sequential loop, so stateful mergers would be
   surprising; `Fn` keeps the merger pure (mirrors
   `Reducer::reduce`'s `&self` shape). Rejected.

## Operator usage patterns

**Default LangGraph-Send shape** (state derives `StateMerge`,
the merge story is auto-wired):

```rust
use entelix_graph::{Annotated, Append, Max, StateGraph, StateMerge};

#[derive(Clone, Default, StateMerge)]
struct AgentState {
    log: Annotated<Vec<String>, Append<String>>,
    score: Annotated<i32, Max<i32>>,
    last_phase: String,
}

let graph = StateGraph::<AgentState>::new()
    .add_node("plan", planner)
    .add_node("draft", draft_runnable)
    .add_node("critique", critique_runnable)
    .add_node("finalize", finalize_runnable)
    .set_entry_point("plan")
    .add_send_edges(
        "plan",
        ["draft", "critique"],
        |s| vec![("draft".into(), s.clone()), ("critique".into(), s.clone())],
        "finalize",
    )
    .add_finish_point("finalize")
    .compile()?;
```

**Cross-field invariants enforced at merge** (manual `StateMerge`
impl — operator-controlled merge logic still wires through
`add_send_edges`):

```rust
impl StateMerge for AgentState {
    type Contribution = AgentStateContribution;
    fn merge(self, update: Self) -> Self {
        let merged = /* per-field combine, with invariant enforcement */;
        merged
    }
    fn merge_contribution(self, c: Self::Contribution) -> Self {
        /* contribution path, same invariant enforcement */
    }
}
// add_send_edges still works — it just calls the manual `merge`.
```

## References

- ADR-0061 — `StateMerge::Contribution` + `add_contributing_node`
  (parent — declared the `merge(Self, Self)` use site that this
  slice actually wires).
- ADR-0059 — `StateMerge` trait + derive (grandparent).
- ADR-0006 — Runnable + StateGraph 1.0 spine.
- 7-차원 roadmap §S10 — Phase 9 (companion-perf hardening /
  parity completion), fifth sub-slice — closes the
  parallel-join positioning hole left by slice 53.
- `crates/entelix-graph/src/state_graph.rs` — `add_send_edges`
  new signature.
- `crates/entelix-graph/src/compiled.rs` — `SendMerger<S>`
  type alias + updated `SendEdge::new` + updated
  `execute_send_edge`.
- `crates/entelix-graph/tests/send_edges.rs` — 4 tests updated
  to derive `StateMerge` on the test state.
