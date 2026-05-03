# entelix-graph-derive

Proc-macro for `#[derive(StateMerge)]` (ADR-0061). Emits a `<Name>Contribution` companion struct (per-field `Option<T>`) + `with_<field>` builder methods + the `StateMerge` impl that auto-wraps raw `T` into `Annotated::new(value, R::default())` for annotated fields.

## Surface

- **`#[derive(StateMerge)]`** — applied to a state struct `S`. Generates:
  - `<S>Contribution { field_a: Option<T>, ... }` — companion partial-shape used by `add_contributing_node`.
  - `S::merge(self, other: Self) -> Self` and `S::merge_contribution(self, c: <S>Contribution) -> Self` per the `StateMerge` trait.
  - `<S>Contribution::with_<field>(self, value)` builder method per field — typed deltas without the `Option::Some` ceremony.
- Re-export: `use entelix_graph::StateMerge` brings both the trait and the derive macro (rust namespacing keeps them disambiguated).

## Crate-local rules

- **Per-field `Annotated<T, R>` semantics** — fields whose type is `Annotated<T, R>` get `Annotated::merge` composition; plain `T` fields default to `Replace` (right-bias). Mixing the two on one struct is the canonical pattern.
- **Generated identifiers are deterministic** — `<S>Contribution` (concat suffix) so the companion is always discoverable. No randomized suffixes.
- **Default-overrides bug structurally avoided** — the `Contribution` variant distinguishes "wrote the default" from "didn't write at all" via `Option<T>`. The derive does NOT collapse `None` into the field's `Default` value at merge time. Slice 53 (ADR-0061) supersedes ADR-0059's `add_reducing_node` shape that had this bug.
- **proc-macro hygiene** — every emitted item is `pub`-visible to callers but uses `::entelix_graph::` absolute paths internally so a state struct in any crate can derive without re-exporting graph internals.

## Forbidden

- A field type the derive cannot reason about (e.g., a generic `T` without a `StateMerge` bound). The macro emits a clear compile error pointing at the offending field.
- Re-introducing `add_reducing_node` style merge (silent default-collapse). Use `add_contributing_node` + the `Contribution` shape.

## References

- ADR-0061 — `StateMerge::Contribution` + `add_contributing_node` (supersedes ADR-0059's reducer-node API).
- ADR-0062 — `add_send_edges` mandatory `S: StateMerge` (parallel-branch join uses `<S as StateMerge>::merge` automatically).
- `crates/entelix-graph-derive/tests/derive_state_merge.rs` — emit-shape regression suite.
