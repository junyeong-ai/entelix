# ADR 0089 — `ToolRegistry<D>` generic + `SubagentBuilder` selection verbs

**Status**: Accepted
**Date**: 2026-05-06
**Decision**: `ToolRegistry` becomes generic over `D` with default `()`. The registry holds a `D` clone that the leaf [`InnerToolService<D>`] threads into every `Tool<D>::execute` via [`AgentContext::new(invocation.ctx, deps.clone())`]. Layers and the tower service spine stay D-free — they consume `ToolInvocation` (carries `ExecutionContext`, no `D`), so the layer ecosystem (`PolicyLayer`, `OtelLayer`, `RetryService`, `ApprovalLayer`, `ScopedToolLayer`, `ToolEventLayer`) compiles unchanged. `Subagent::from_whitelist` / `from_filter` factory functions are removed; the sole construction surface is `Subagent::builder(model, &parent_registry)` returning `SubagentBuilder` with the `restrict_to` / `filter` selection verbs and `with_sink` / `with_approver` / `with_skills` configuration verbs. Selection verbs (`restrict_to`, `filter`) are added as named exceptions to the naming taxonomy's verb-prefix rule.

## Context

Slice 100 (ADR-0085) made `Tool<D>` generic but left `ToolRegistry` deps-less. Tools registered through that registry were always `Tool<()>`; operator-typed deps had no path to dispatch. Slice 103 closes the loop: registry holds `D`, threads it through to every leaf `Tool::execute`, and lets sub-agents inherit the parent's deps automatically through the existing layer-factory share-by-Arc invariant.

The Subagent factory functions (`from_whitelist`, `from_filter`) plus post-construction `with_*` setters had grown into 5 entry points with overlapping configuration surface. A single builder unifies authority bounds (selection) with optional wiring (sink / approver / skills) into one fluent path — the operator stops choosing between "did I declare this in `from_whitelist` or `with_skills`" and just composes.

## Decision

### `ToolRegistry<D = ()>`

```rust
pub struct ToolRegistry<D = ()> {
    by_name: HashMap<String, Arc<dyn Tool<D>>>,
    deps: D,
    factory: Option<LayerFactory<D>>,
}

impl ToolRegistry<()> {
    pub fn new() -> Self { Self::with_deps(()) }
}

impl<D: Clone + Send + Sync + 'static> ToolRegistry<D> {
    pub fn with_deps(deps: D) -> Self { ... }
    pub fn register(self, tool: Arc<dyn Tool<D>>) -> Result<Self> { ... }
    pub fn layer<L>(self, layer: L) -> Self where ... { ... }
    pub fn deps(&self) -> &D { ... }
    pub fn filter<F>(&self, predicate: F) -> Self where F: Fn(&dyn Tool<D>) -> bool { ... }
    pub fn restricted_to(&self, allowed: &[&str]) -> Result<Self> { ... }
    pub fn dispatch(&self, ...) -> Result<Value> { ... }
    // ... is_empty / len / names / get / service stay generic over D
}
```

The default `<D = ()>` keeps every existing call site (`ToolRegistry::new()`, `let registry: ToolRegistry = ...`) working unchanged. Operator-typed registries opt in via `ToolRegistry::with_deps(my_deps)`.

### Layer factory stays D-free at the type level

```rust
type LayerFactory<D> = Arc<dyn Fn(InnerToolService<D>) -> BoxedToolService + Send + Sync>;
```

The factory is generic over `D` but its *output* is `BoxedToolService` — a `BoxCloneService<ToolInvocation, Value, Error>`. Layers operate on that boxed shape; they never see `D`. Concretely, `PolicyLayer` and `OtelLayer` and the rest stay defined exactly as they were in slice 100. invariant 4's parent-layer-factory share-by-Arc shape is preserved — the factory clones into the narrowed view (`filter` / `restricted_to`) without copy.

### `InnerToolService<D>` materialises `AgentContext<D>`

```rust
impl<D: Clone + Send + Sync + 'static> Service<ToolInvocation> for InnerToolService<D> {
    fn call(&mut self, invocation: ToolInvocation) -> Self::Future {
        let tool = Arc::clone(&self.tool);
        let deps = self.deps.clone();
        Box::pin(async move {
            // ... validate input ...
            let agent_ctx = AgentContext::new(invocation.ctx, deps);
            tool.execute(invocation.input, &agent_ctx).await
        })
    }
}
```

`D` surfaces solely at the leaf-service boundary; tower layers upstream see only `ToolInvocation`. This is the structural fix from ADR-0084 §"Why split?" — layer ecosystem D-free, leaf ecosystem D-aware.

### `SubagentBuilder` replaces factory functions

```rust
let sub = Subagent::builder(model, &parent_registry)
    .restrict_to(&["alpha", "beta"])      // strict whitelist (typo errors)
    // OR .filter(|tool| tool.metadata().name.starts_with("calc_"))  // graceful predicate
    .with_sink(parent_sink)               // forward lifecycle events upstream
    .with_approver(approver)              // inherit parent's HITL gate
    .with_skills(&parent_skills, &["echo"])  // narrowed skill subset
    .build()?;                            // validates selection + skills, returns Subagent
```

Removed entry points (no shims, invariant 14):
- `Subagent::from_whitelist(model, parent, allowed)` →
  `Subagent::builder(model, parent).restrict_to(allowed).build()`
- `Subagent::from_filter(model, parent, predicate)` →
  `Subagent::builder(model, parent).filter(predicate).build()`
- `Subagent::with_sink / with_approver / with_skills` (post-construction setters) →
  same names but on `SubagentBuilder`, before `.build()`

The `Selection` enum (`All` / `Restrict(Vec<String>)` / `Filter(Box<dyn Fn>)`) is internal — operators interact only via the verb methods. Strict / graceful asymmetry matches the registry-level pattern (`restricted_to` errors on missing names, `filter` accepts empty result).

`build()` is the validation point — selection apply (`restricted_to` may error) and skills lookup (missing names error) both run there. Using the builder for non-validating selection (`All`, `filter`) still goes through `build()` for type uniformity.

### Naming-taxonomy exception: selection verbs

`SubagentBuilder::restrict_to` and `SubagentBuilder::filter` violate the verb-prefix rule (`with_` / `add_` / `set_` / `register`). Adding them to `is_builder_verb_prefix` in `xtask/src/invariants/naming.rs` and documenting the carve-out in `.claude/rules/naming.md` codifies the exception:

> Selection verbs (`restrict_to` / `filter`) are reserved for builders that produce *narrowed views* of an existing parent set.

Analogous to `Iterator::filter` — the verb is canonical English for "select a subset", and `with_restriction` / `with_predicate` would read worse. The exception is bounded: only these two names, only on builders, only when narrowing a parent set.

## Consequences

- `ToolRegistry::new()` stays the deps-less default — no migration churn for the 99% of code that doesn't need typed deps.
- Operator typed-deps tools land via `ToolRegistry::with_deps(...).register(Arc::new(MyTypedTool))` once they have a `Tool<MyDeps>` impl. The macro extension to support `Tool<D>` lands when `Agent<S, D>` (slice 104) closes the carrier loop.
- 23 `Subagent::from_whitelist` + `Subagent::from_filter` call sites across 6 test files migrated to the builder pattern. Mechanical — `Subagent::from_whitelist(M, P, A)?` becomes `Subagent::builder(M, P).restrict_to(A).build()?`, `from_filter(M, P, F)` becomes `.filter(F).build()`.
- 4 public-API baselines refreshed: `entelix-core` (ToolRegistry generic), `entelix-agents` (SubagentBuilder + removed factories), `entelix-tools` and `entelix-mcp` from slice 100/102 carry-over.
- `cargo xtask naming` extended with two named exceptions (`restrict_to`, `filter`) — bounded, documented in naming.md.
- All workspace tests pass; clippy `--all-targets` clean.

## Why `Subagent<M>` does not yet take `D`

`Subagent<M>` holds a `ToolRegistry<()>` for now — sub-agents inherit the parent's tool selection but not yet the parent's typed deps. Slice 104 (`Agent<S, D>`) generalises `Subagent<M, D = ()>` end-to-end. The intermediate state in slice 103 keeps the migration tractable: `ToolRegistry<D>` is the foundation, `Subagent<M, D>` and `Agent<S, D>` build on it once the carrier loop closes.

## References

- ADR-0084 — `AgentContext<D>` typed-deps carrier; `InnerToolService<D>` materialises this at dispatch time.
- ADR-0085 — `Tool<D>` trait migration; `ToolRegistry<D>` is the registry-side counterpart.
- ADR-0010 — naming taxonomy; this ADR adds the `restrict_to` / `filter` selection-verb exception.
- Invariant 4 — Hand contract / managed-agent shape (parent-layer-factory share-by-Arc preserved).
- Invariant 7 — F7 mitigation (sub-agent must explicitly declare authority bounds — builder requires `restrict_to` or `filter` for non-trivial narrowing).
- v3 plan slice 103.
