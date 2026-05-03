# ADR 0061 — `StateMerge::Contribution` companion type + `add_contributing_node` (LangGraph TypedDict parity completion)

**Status**: Accepted (supersedes the `add_reducing_node` portion of ADR-0059)
**Date**: 2026-05-01
**Decision**: Phase 9 of the post-7-차원-audit roadmap (fourth sub-slice — final close of the per-field reducer story)

## Context

ADR-0059 introduced `StateMerge` + `#[derive(StateMerge)]` +
`StateGraph::add_reducing_node`. The reducing-node entry point
took a `Runnable<S, S>` whose output was treated as a *contribution*
and folded into the inbound state via `S::merge(snapshot, contribution)`.

That shape was simple but had a sharp corner the slice's tests
didn't catch. When a node didn't write a particular slot, the
`Default::default()` value for that field still flowed into the
merge — and reducers like `Max<i32>` happily treat the default `0`
as a candidate. A current value of `-100` would collapse to `0`
under `Max(-100, 0)` even though the contributing node never
intended to touch the score. The same hazard exists for any
reducer where the *default value* of `T` is a meaningful candidate
in the reducer's domain (every numeric `Max`, every `Append<U>`
on a state field that should keep its current `Vec` rather than
getting a duplicate-tail merge from a default-empty contribution
that was misordered, …).

LangGraph's Python TypedDict surface dodges this by relying on the
fact that nodes return a *partial dict*: the runtime can see
exactly which keys were set and only merges those. Rust has no
such "field-was-set-vs-defaulted" distinction at the type level —
we have to encode the intent.

## Decision

Encode "did the node touch this slot?" as `Option<T>` on a
companion `Contribution` struct. Every state type implementing
`StateMerge` declares an associated `Contribution` type whose
fields are `Option`-wrapped versions of the state fields. Nodes
return the `Contribution`; `merge_contribution` unfolds it,
applying the per-field reducer only to slots the node actually
populated.

### `StateMerge` trait shape (extended from ADR-0059)

```rust
pub trait StateMerge: Sized {
    type Contribution: Default + Send + Sync + 'static;
    fn merge(self, update: Self) -> Self;
    fn merge_contribution(self, contribution: Self::Contribution) -> Self;
}
```

`merge(Self, Self) -> Self` stays — it's the right shape for
parallel-branch joins (`add_send_edges`) where two branches each
produce a complete `S`. `merge_contribution(Self, Contribution)`
is the new per-node entry point.

### Derive macro emits both companion struct + impl

For

```rust
#[derive(Clone, Default, StateMerge)]
struct AgentState {
    log: Annotated<Vec<String>, Append<String>>,
    score: Annotated<i32, Max<i32>>,
    last_message: String,
}
```

the macro generates `AgentStateContribution` with:

- One `pub field: Option<T>` per input field.
- One `with_<field>(self, value: ...) -> Self` builder method per
  field. For `Annotated<T, R>` fields the builder takes raw `T`
  and wraps it via `Annotated::new(value, R::default())` — node
  bodies stay readable.
- `Default` derive on the companion struct, so
  `AgentStateContribution::default()` is the zero-everything-set
  starting point.

And the `StateMerge` impl with both methods.

### `add_contributing_node` replaces `add_reducing_node`

The `add_reducing_node` builder method introduced by ADR-0059 is
*removed* in this slice — its semantics had the default-overrides
bug above and there's no legitimate use case for it once the
contribution shape exists. The new builder:

```rust
StateGraph::<S>::add_contributing_node(name, runnable: Runnable<S, S::Contribution>)
```

wraps the runnable in `ContributingNodeAdapter<S>`, which
snapshots the inbound state, runs the node to get its
contribution, and returns `snapshot.merge_contribution(contribution)`.

Per the project standing instructions ("처음부터 이렇게 설계된
것처럼 흔적도 없이 클린하게"): no compatibility shim from
`add_reducing_node`, no deprecation comment, no fallback. The
`reducing_node` module + tests file are deleted. The intent of
slice 51 was always per-field reducer composition — this slice is
the correct shape that intent has from "the start".

### Why builder methods take raw `T` for `Annotated` fields

A node author writes:

```rust
AgentStateContribution::default()
    .with_log(vec!["entry".into()])
    .with_score(50)
```

instead of

```rust
AgentStateContribution::default()
    .with_log(Annotated::new(vec!["entry".into()], Append::new()))
    .with_score(Annotated::new(50, Max::new()))
```

The builder reaches for `R::default()` to construct the
`Annotated`. This requires the reducer to implement `Default` —
all four stock reducers (`Replace`, `Append`, `MergeMap`, `Max`)
do, and any unit-struct user reducer trivially does. Stateful
reducers needing configuration are out of scope for the derive;
operators using them implement `StateMerge` manually.

### Why `Contribution: Default + Send + Sync + 'static`

- `Default` — every node body starts from `Contribution::default()`
  and chains `with_*` setters. Without `Default` the entry
  ergonomic collapses.
- `Send + Sync + 'static` — required by the `Runnable<S, S::Contribution>`
  trait bounds (output type flows through async tasks across
  thread boundaries).

### Why `merge` stays alongside `merge_contribution`

Two distinct callers:

- `add_send_edges` parallel-branch join — two branches each
  produce a complete `S`; the dispatcher folds them via
  `S::merge(left, right)`. There's no contribution shape here:
  both sides are full states.
- `add_contributing_node` — single node returns a partial
  `Contribution`; folded via `merge_contribution`.

Collapsing onto one method would force every call site through
the `Option`-wrapped surface, even where both inputs are
complete. Two methods, two clear shapes.

### Why detect `Annotated` syntactically (unchanged from ADR-0059)

The macro inspects only the *last* path segment of a field type;
shadowing `Annotated` with a different type is forbidden by
convention and would surface as a type-check error during
compilation of the generated code.

### Tests

- 7 derive integration tests (`tests/derive_state_merge.rs`):
  the original 5 from slice 51 (still pass — `merge` semantic
  unchanged) + 2 new tests for `merge_contribution` and
  the builder methods (`contribution_companion_with_builder_methods`,
  `contribution_with_no_slots_keeps_every_current_value`).
- 3 graph integration tests
  (`crates/entelix-graph/tests/contributing_node.rs`):
  multi-node sequential graph chains correctly,
  unwritten-slot-keeps-current-value regression (`-100` score
  through an empty contribution stays `-100`),
  `add_contributing_node` coexists with `add_node` in the same
  graph.
- 4 unit tests in `entelix-graph/src/reducer.rs` (manual
  `StateMerge` impl exercises both `merge` and
  `merge_contribution` paths).
- 3 unit tests in `entelix-graph/src/contributing_node.rs`
  (adapter-level: writes-only-named-slots; unwritten-keeps-current;
  inner-error propagation).

## Consequences

✅ The default-overrides bug in `add_reducing_node` is structurally
fixed — a node that doesn't write a slot leaves it `None` in the
contribution, and `merge_contribution` keeps the current value
verbatim.
✅ Operator ergonomic matches LangGraph TypedDict: nodes write
only the slots they touched, framework auto-merges through per-
field reducers. Adding a new state field is still a one-line
edit on the struct (the derive regenerates the companion +
builders).
✅ `merge` and `merge_contribution` cover the two distinct shapes
(parallel join vs. single-node contribution) without forcing one
shape onto the other's call sites.
✅ `<Name>Contribution` builder methods take raw `T` for
`Annotated` fields — node bodies don't carry boilerplate
`Annotated::new(value, R::default())` chains.
✅ `add_reducing_node` removed cleanly — no shim, no deprecation,
no fallback. The codebase looks like contribution-style was
the design from the start.
❌ Public-API baseline drift on `entelix-graph`,
`entelix-graph-derive`, and `entelix` (facade). Refrozen.
❌ Reducers used in `Annotated<T, R>` fields must now implement
`Default` (was implicit before — no slot needed to construct the
reducer at derive time). All stock reducers already qualify;
operators with stateful reducers go through manual
`StateMerge` impls.

## Alternatives considered

1. **Keep `add_reducing_node` as-is, document the
   default-overrides hazard** — tells the user "watch out, this
   API has a sharp edge." Right answer for a quick fix; wrong
   answer for a long-term API. Rejected.
2. **Make every state field `Option<T>` directly** — invades
   the user's state struct and breaks ergonomic on the read
   side ("did I forget to set this, or is it intentionally
   empty?" ambiguity). The companion struct keeps the
   `Option`-ness on the *contribution* path only. Rejected.
3. **`Update<S>` companion built via a separate trait
   (`StateUpdate`)** — splits the per-field metadata across
   two derives. The single `StateMerge` derive owns both shapes
   and they stay coherent. Rejected.
4. **Builder method takes `Annotated<T, R>` directly (no raw
   `T` ergonomics)** — keeps the reducer instance fully
   user-controlled but forces every call site through the
   wrapper boilerplate. The `R::default()` shortcut is the
   right ergonomic for stock reducers; operators with stateful
   reducers fall back to manual impl. Rejected.
5. **Keep both `add_reducing_node` and `add_contributing_node`
   as parallel entry points** — three node-registration shapes
   fight for attention without adding ergonomic clarity. The
   contribution shape strictly dominates the same-shape merge
   for per-node use. Rejected.

## References

- ADR-0006 — Runnable + StateGraph 1.0 spine.
- ADR-0059 — `StateMerge` trait + derive (parent — this slice
  supersedes its `add_reducing_node` portion; the trait shape and
  derive remain, with `Contribution` + `merge_contribution`
  added).
- 7-차원 roadmap §S10 — Phase 9 (companion-perf hardening /
  parity completion), fourth sub-slice — completes the
  per-field reducer story.
- `crates/entelix-graph-derive/src/lib.rs` — derive macro emits
  companion struct + builder methods + both `merge` impls.
- `crates/entelix-graph/src/reducer.rs` — `StateMerge` trait
  with associated `Contribution` type.
- `crates/entelix-graph/src/contributing_node.rs` —
  `ContributingNodeAdapter<S>` (replaces `ReducingNodeAdapter`).
- `crates/entelix-graph/src/state_graph.rs` —
  `add_contributing_node` builder method.
- `crates/entelix-graph/tests/contributing_node.rs` — 3
  integration tests including the negative-current regression.
- `crates/entelix-graph-derive/tests/derive_state_merge.rs` —
  5 original + 2 new contribution-path tests.
