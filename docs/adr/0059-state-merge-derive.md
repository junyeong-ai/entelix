# ADR 0059 — `StateMerge` trait + `#[derive(StateMerge)]` proc-macro for declarative per-field reducer composition

**Status**: Superseded by ADR-0061 (`add_reducing_node` replaced by `add_contributing_node` + `StateMerge::Contribution` companion shape)
**Date**: 2026-05-01
**Decision**: Phase 9 of the post-7-차원-audit roadmap (second sub-slice — LangGraph parity completion)

## Context

ADR-0006 shipped the `StateGraph<S>` spine; the `Reducer<T>` trait
and `Annotated<T, R>` wrapper landed alongside (`Replace`,
`Append`, `MergeMap`, `Max` impls). `Annotated<T, R>` was a
*standalone helper*: users had to wrap their state fields manually
and either (a) thread the wrapped state through a full-state
`add_node` returning `Result<S>`, or (b) write a per-graph merger
closure inside `add_node_with` that called each reducer by hand.

The LangGraph parity story was incomplete. In Python LangGraph,
the user writes:

```python
class AgentState(TypedDict):
    log: Annotated[list[str], operator.add]   # appended
    score: Annotated[int, max]                # max-reduced
    last_phase: str                           # replaced

graph.add_node("plan", planner)  # returns {"log": [...], "score": 50}
graph.add_node("score", scorer)
```

The runtime *automatically* applies the right reducer per slot
when a node returns a partial dict. The framework owns merge
ergonomics — adding a new state field never edits the graph
builder.

Rust's type system gives us a stronger version: per-field reducer
choice is declared *on the type*, the compiler verifies the reducer
signature matches the field shape, and the merge code is generated
once per state struct via a derive macro.

The pre-existing `add_node_with(name, runnable, merger)` shipped
the *flexible* end (per-graph merger closure). What was missing:
the *declarative* end — state-type-driven merge that the framework
applies automatically.

## Decision

Add three things in concert:

### 1. `StateMerge` trait in `entelix-graph`

```rust
pub trait StateMerge: Sized {
    fn merge(self, update: Self) -> Self;
}
```

The dispatch-loop counterpart to `Reducer<T>`, one level up:
implementors describe how an incoming update folds into the
current state. Manual impls are supported when field-by-field
shape doesn't fit (cross-field invariants).

### 2. `Annotated::merge(self, other: Self) -> Self`

Building block the derive emits per `Annotated` field. Both sides
share `R` by construction; the result keeps `self`'s reducer
instance (matters for stateful reducers; moot for the unit-struct
stock impls).

`Annotated<T, R>` also gains a `Default` impl gated on `T: Default
+ R: Default` so deriving `Default` on a state struct that uses
`Annotated` fields works without further plumbing.

### 3. `entelix-graph-derive` proc-macro crate (19th workspace member)

Ships `#[derive(StateMerge)]`. The macro inspects struct fields:

- Field type ends in `Annotated<...>` → emit
  `self.field.merge(update.field)`
- Any other type → emit `update.field` (replace)

Tuple and unit structs are rejected at compile time — per-field
reducer composition requires named fields.

The derive lives in its own proc-macro crate (standard Rust
pattern; required by `[lib] proc-macro = true`). `entelix-graph`
re-exports it as `pub use entelix_graph_derive::StateMerge` —
same name as the trait. Macros and traits live in different
namespaces, so `#[derive(StateMerge)]` and `impl StateMerge for X`
coexist without ambiguity. Serde's `Serialize` / `Deserialize`
follow the identical shape.

### 4. `StateGraph::add_reducing_node(name, runnable)`

Third node-registration entry point. Takes `R: Runnable<S, S>` +
`S: StateMerge`; wraps the runnable in `ReducingNodeAdapter<S>`
which snapshots the inbound state, runs the inner runnable to
produce a *contribution* (also of shape `S` — typically built
from `S::default()` plus the slots the node wrote), and returns
`snapshot.merge(contribution)`.

The three node-registration shapes now coexist along the
ergonomic spectrum:

| Shape | When |
|---|---|
| `add_node(name, R)` where `R: Runnable<S, S>` | Full-state replace. Node owns the entire shape. |
| `add_node_with(name, R, merger)` where `R: Runnable<S, U>` | Bespoke per-graph merge closure. Best when merge is graph-specific. |
| `add_reducing_node(name, R)` where `R: Runnable<S, S>` + `S: StateMerge` | Declarative merge through the state type's `StateMerge`. Best when state advertises its merge story (typical LangGraph parity case). |

Adding a new state field is a one-line struct edit — graph
builders and node bodies don't change.

### Why three shapes, not one

Squashing everything onto a single shape — for example "always
require `S: StateMerge` and have `add_node` apply merge" — would
break the `Runnable<S, S>` full-state-replace semantic that
`add_node` currently advertises. It would also force *every*
state type to implement `StateMerge` (or write a no-op impl)
before being usable in a graph. The split keeps existing graphs
working unchanged and opts users into the merge story explicitly.

### Why no derive attributes

The macro takes no `#[merge_with(...)]` or `#[reducer(...)]`
attributes. The reducer is declared *in the type* via
`Annotated<T, R>` — putting the same information in an attribute
would fragment the SSoT. If a user wants per-field control without
the `Annotated` wrapper, they implement `StateMerge` manually;
that path is documented and tested.

### Why detect by last path segment

The macro syntactically matches `Annotated` as the last path
segment of a field type. This handles `Annotated<...>`,
`entelix_graph::Annotated<...>`, and `crate::state::Annotated<...>`
uniformly. The only case it misses is a *user-defined* type
named `Annotated` from a different module — by convention,
shadowing the entelix `Annotated` is forbidden and would surface
as a type-check error during compilation of the generated code
anyway (the generated `.merge(...)` call would fail to resolve).

### Tests

- 5 derive integration tests in `entelix-graph-derive/tests/derive_state_merge.rs`:
  mixed Annotated + plain fields, all-Annotated, all-plain,
  explicit-Replace through `Annotated<T, Replace>`, generic
  struct.
- 2 graph integration tests in `entelix-graph/tests/reducing_node.rs`:
  multi-node sequential graph chains per-field reducers
  correctly; `add_reducing_node` coexists with `add_node`
  (full-replace) in the same graph.
- 4 unit tests added to `entelix-graph/src/reducer.rs`:
  `Annotated::merge` for `Append` and `Max`, manual `StateMerge`
  impl with cross-field invariants, derive companion smoke test.
- 2 unit tests in `entelix-graph/src/reducing_node.rs`:
  log-append + score-max + tag-replace through the adapter,
  inner-error propagation.

## Consequences

✅ Operators write a state struct once with `Annotated<T, R>`
fields and `#[derive(StateMerge)]`; the graph builder never
mentions reducers. Adding a new field is a one-line edit on the
struct, not on every node site.
✅ `Replace` / `Append` / `MergeMap` / `Max` all plug in
declaratively. Custom `Reducer<T>` impls also work — the macro
doesn't care about which reducer is bundled.
✅ Generic state structs are supported (the derive forwards
`split_for_impl` generics).
✅ Backward-compatible by construction: existing `add_node` and
`add_node_with` shapes work unchanged. The new entry point is
opt-in per node.
✅ `Annotated<T, R>: Default` lets state structs derive
`Default` without manual impls — a frequent stumbling block in
the prior design.
✅ Workspace grows to 19 crates; derive lives in its own
proc-macro crate (standard Rust pattern), avoiding bringing
syn/quote into `entelix-graph`'s build graph.
❌ Public-API baseline grew (`StateMerge` trait, derive,
`ReducingNodeAdapter`, `Annotated::merge`, `Annotated::default`,
`StateGraph::add_reducing_node`, facade re-exports). Refrozen.
❌ Macro detects `Annotated` syntactically by last path segment.
A user re-naming a different type to `Annotated` and using it in
a `StateMerge`-derived struct would surface a confusing
"`merge` not found" compile error. Convention-not-mechanism;
documented in the derive crate doc.
❌ Three node-registration entry points (vs. one in LangGraph)
cost a marginal complexity tax on the surface. The benefit:
each entry point is the right tool for its job, and
mid-graph mixing is supported.

## Alternatives considered

1. **Single `add_node` that conditionally applies `S::merge` when
   `S: StateMerge`** — Rust has no specialisation on stable. The
   only way to branch is at the trait bound, which would force
   `S: StateMerge` on every state type. Rejected.
2. **Reducer attributes (`#[merge(append)]`, `#[merge(max)]`)
   instead of `Annotated<T, R>` field type** — splits the
   declaration across two surfaces (the field type *and* the
   attribute). The `Annotated<T, R>` wrapper is a single
   self-describing type. Rejected.
3. **Companion `Update<S>` struct with per-field `Option<T>`
   generated by the derive** — closer to LangGraph's TypedDict
   semantic (literal "did the node return this slot or not?")
   but invasive: every node returns `Update<S>`, not `S`, forcing
   call-site changes everywhere. Reserved for a possible follow-up
   if operators ask for finer "no contribution" semantics.
4. **Inline derive in `entelix-graph` itself** — Rust forbids
   shipping proc-macros from a non-proc-macro crate
   (`[lib] proc-macro = true` is exclusive of normal lib code).
   The 19th-crate split is mechanically required. Rejected.
5. **Hand-written `StateMerge` impls only, no derive** — keeps
   the dependency footprint smaller (no syn/quote/proc-macro2)
   but reintroduces the boilerplate the slice exists to remove.
   Five-line struct → twenty-line impl is a regression. Rejected.

## Operator usage patterns

**Default LangGraph parity**:

```rust
use entelix_graph::{Annotated, Append, Max, StateGraph, StateMerge};

#[derive(Clone, Default, StateMerge)]
struct AgentState {
    log: Annotated<Vec<String>, Append<String>>,
    score: Annotated<i32, Max<i32>>,
    last_phase: String,
}

let graph = StateGraph::<AgentState>::new()
    .add_reducing_node("plan", planner)
    .add_reducing_node("score", scorer)
    .add_edge("plan", "score")
    .set_entry_point("plan")
    .add_finish_point("score")
    .compile()?;
```

**Mixed full-replace + reducing in the same graph**:

```rust
let graph = StateGraph::<AgentState>::new()
    .add_reducing_node("contribute", reducing_node)   // merges
    .add_node("stamp", full_state_node)               // replaces
    ...
```

**Manual `StateMerge` impl for cross-field invariants**:

```rust
impl StateMerge for AgentState {
    fn merge(self, update: Self) -> Self {
        let merged = Self {
            log: self.log.merge(update.log),
            score: self.score.merge(update.score),
            last_phase: update.last_phase,
        };
        // Enforce: score must be non-negative even if a contributing
        // node disagreed (defensive — operator owns the rule).
        debug_assert!(merged.score.value >= 0, "score must be non-negative");
        merged
    }
}
```

## References

- ADR-0006 — Runnable + StateGraph 1.0 spine (parent — `Reducer<T>` and `Annotated<T, R>` shipped here).
- 7-차원 roadmap §S10 — Phase 9 (companion-perf hardening / parity completion), second sub-slice.
- `crates/entelix-graph-derive/` — new 19th workspace member; proc-macro crate.
- `crates/entelix-graph/src/reducer.rs` — `StateMerge` trait, `Annotated::merge` + `Default` impl.
- `crates/entelix-graph/src/reducing_node.rs` — `ReducingNodeAdapter<S>`.
- `crates/entelix-graph/src/state_graph.rs` — `add_reducing_node` builder method.
- `crates/entelix-graph/src/lib.rs` — re-exports trait + derive (same name, different namespace; serde pattern).
