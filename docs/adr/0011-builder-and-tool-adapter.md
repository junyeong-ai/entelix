# ADR 0011 — Builder macro choice + Tool-as-Runnable boundary + Typed-I/O ergonomics

**Status**: Accepted
**Date**: 2026-04-26 (O1, O2) · 2026-04-30 (O3)
**Closes**: O1 (bon vs typed-builder), O2 (Tool extends Runnable vs Adapter), O3 (typed-I/O ergonomics on top of erased Tool)

## Context

Two open decisions blocked the start of Phase 1 (composition primitive
implementation in `entelix-core` + `entelix-runnable`):

1. **O1** — Which derive-style builder crate underlies our `*Builder`
   convention? (`bon` 3.x vs `typed-builder` 0.20 vs hand-rolled.)
2. **O2** — Does `Tool` extend `Runnable<Value, Value>`, sit alongside it,
   or get exposed via an adapter?

Both decisions touch every public type in Phase 1 onward, so they must be
closed before code lands.

## Decision

### O1 — `bon` 3.9 is the single supported builder macro

`bon` is already pinned at the workspace root (`workspace.dependencies.bon =
"3.9"`). This ADR confirms that as load-bearing — no `typed-builder`, no
hand-rolled builders unless the type genuinely cannot fit `bon`'s shape
(rare; document the exception in the type's rustdoc).

Reasons:

- **Method-derived builders** — `#[bon::builder]` attaches to a free function
  or `impl` method, not just a struct. This matches our patterns
  (`Agent::builder()`, `StateGraph::node().build()`) without forcing
  artificial structs.
- **Compile-time output** — `bon` expands to plain `pub fn` chains, so
  `cargo public-api` diff sees the real surface, not opaque proc-macro state
  machines (a chronic typed-builder pain point).
- **Ergonomic optionals** — `Option<T>` fields drop the `.with_xxx(Some(...))`
  wart; users write `.xxx(...)` and skip when not needed.
- **Async builders** — Phase 1 `Agent::builder()...build().await?` shape
  works directly. `typed-builder` requires custom `build_async` plumbing.
- **Compile-cost spike** — informal measurement on a 50-field builder
  shape: `bon` ≈ 0.7s, `typed-builder` ≈ 0.9s, both acceptable. No
  tie-breaker; preference goes to ergonomic features.

Convention reaffirmed (see ADR-0010): every builder type ends in `*Builder`
and the finalizer is `fn build(self) -> Result<T, Error>` (always `Result`,
even when current impl is infallible — leaves room for validation later
without breaking callers).

### O2 — `Tool` does **not** extend `Runnable`; an adapter bridges them

`Tool` and `Runnable` are sibling capability traits, not a hierarchy. The
crate that owns `Runnable` (`entelix-runnable`) provides
`ToolToRunnableAdapter` so any `Tool` composes via `.pipe()`.

```rust
// entelix-core::tools (DAG root)
#[async_trait::async_trait]
pub trait Tool: Send + Sync + 'static {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn input_schema(&self) -> &schemars::Schema;
    async fn execute(
        &self,
        input: serde_json::Value,
        ctx: &ExecutionContext,
    ) -> Result<serde_json::Value, Error>;
}

// entelix-runnable::adapters (depends on entelix-core)
pub struct ToolToRunnableAdapter<T: Tool> { inner: Arc<T> }

#[async_trait::async_trait]
impl<T: Tool> Runnable<serde_json::Value, serde_json::Value>
    for ToolToRunnableAdapter<T> { /* delegates to T::execute */ }
```

Reasons:

- **Crate DAG hygiene** — `Tool` lives in `entelix-core` (DAG root,
  zero entelix deps). `Runnable` lives in `entelix-runnable`
  (depends on core). Trait inheritance would force a cycle or move
  `Runnable` into core, bloating the root.
- **Asymmetric surface** — `Tool` carries metadata (`name`,
  `description`, `input_schema`) and a single concrete signature
  (`(Value, &Ctx) → Value`). `Runnable<I, O>` is generic over I/O.
  Forcing `Tool` to subtype a typed `Runnable<Value, Value>` either
  loses the generics on the `Runnable` side or duplicates them on
  the `Tool` side. The adapter keeps each contract pure.
- **Composition still works** — `let chain =
  prompt.pipe(model).pipe(tool.into_runnable());` is one extra method
  call in user code. `IntoRunnable` blanket impl makes it ergonomic.
- **Sub-agent semantics align** — Anthropic's "brain passes hand"
  (managed-agents.md) treats tools as a registry, not a chain link.
  Keeping `Tool` orthogonal to `Runnable` matches the operational shape.

The reverse direction (Runnable → Tool) is **not** auto-derived. Wrapping
an arbitrary `Runnable<Value, Value>` as a `Tool` requires explicit metadata
(name + description + schema), so we expose
`Tool::from_runnable(name, desc, schema, runnable)` rather than blanket
impl. Prevents accidental tool exposure.

### O3 — Typed-I/O ergonomics through `SchemaTool` + `SchemaToolAdapter`

The base `Tool` trait carries `serde_json::Value` on both sides so the
registry, dispatcher, and metadata machinery can stay erased. Most tool
authors prefer compile-time typed `Input`/`Output` plus an
auto-generated JSON Schema. `entelix-tools::SchemaTool` is the typed
sibling; `SchemaToolAdapter<T: SchemaTool>` is the adapter that bridges
back to the erased `Tool` contract. The mainstream typed-tool
ergonomics (LangChain `StructuredTool`, OpenAI `tool` decorators)
become a one-line `.into_adapter()` call without touching the
dispatcher.

```rust
// entelix-tools::schema_tool
#[async_trait::async_trait]
pub trait SchemaTool: Send + Sync + 'static {
    type Input: DeserializeOwned + JsonSchema + Send + 'static;
    type Output: Serialize + Send + 'static;
    const NAME: &'static str;
    fn description(&self) -> &str;
    fn effect(&self) -> ToolEffect { ToolEffect::default() }
    fn version(&self) -> Option<&str> { None }
    fn retry_hint(&self) -> Option<RetryHint> { None }
    fn output_schema(&self) -> Option<serde_json::Value> { None }
    async fn execute(&self, input: Self::Input, ctx: &ExecutionContext)
        -> Result<Self::Output>;
}

pub trait SchemaToolExt: SchemaTool + Sized {
    fn into_adapter(self) -> SchemaToolAdapter<Self> { /* ... */ }
}

// Implements `Tool` — input schema cached in Arc<ToolMetadata> at
// construction; runtime hot path is one pointer dereference.
pub struct SchemaToolAdapter<T: SchemaTool> { /* ... */ }
```

Reasons:

- **Mainstream parity** — typed-input ergonomics is the dominant
  pattern in LangChain, Anthropic SDK, OpenAI Python SDK. The erased
  `Tool` is right for the dispatcher; the typed `SchemaTool` is right
  for the author.
- **Single source of truth for schemas** — `schemars::schema_for!(T::Input)`
  generates the JSON Schema once at adapter construction. Hand-rolled
  schemas drift; `Deserialize` + `JsonSchema` derives stay in lockstep.
- **Invariant 4 alignment** — `SchemaTool` is *not* `Tool`. The adapter
  is what implements `Tool`, so the dispatcher / registry / metadata
  surface is unchanged. No second method on the `Tool` contract.
- **Effect / version / retry-hint passthrough** — the typed author
  overrides provided trait methods; the adapter funnels them into
  `ToolMetadata` so OTel attributes, retry middleware, and
  `Approver` defaults all behave identically to a hand-rolled `Tool`.
- **No back-channel** — the adapter only consumes `&ExecutionContext`;
  Invariant 10 (no tokens in tools) carries through unchanged.

Lives in `entelix-tools` (not `entelix-core`) because `schemars` is a
schema-derivation dep, not a DAG-root concern. Operators who don't
need typed ergonomics never pull `schemars` transitively.

## Invariant impact

This ADR is consistent with all 18 invariants. Notably:

- **Invariant 4** (Hand contract) — Tool's only method remains
  `execute(input, ctx) → output`. Adapter does not add methods to `Tool`.
- **Invariant 7** (Runnable composition contract) — Tool composes via
  adapter, satisfying invariant 7's spirit (every composable thing reachable
  through `Runnable<I, O>`) without trait inheritance.
- **Invariant 12** (No backwards-compat shims) — first ADR to close an open
  decision. If a follow-up wants `Tool: Runnable<Value, Value>`, that's a
  new ADR + cleanup migration; no `pub use` aliases now.

## Consequences

✅ DAG cycle-free: `entelix-core` → `entelix-runnable` only one direction.
✅ `Tool` stays minimal — easy mental model, easy to implement.
✅ `bon` is the one builder macro; reviewer doesn't ask "which builder?"
✅ Adapter is small (~30 lines), reviewable end-to-end.
❌ Users must remember `.into_runnable()` once when composing a tool. Tradeoff
   accepted for crate-DAG hygiene.
❌ If we later want `Tool: Runnable`, an ADR + breaking change is required.
   Pre-1.0 we accept that risk; the architecture docs already document the
   adapter pattern as the chosen path.

## Alternatives considered

1. **`Tool: Runnable<Value, Value>` (trait inheritance)** — forces
   `Runnable` into `entelix-core` or a cycle. Rejected.
2. **Runnable in `entelix-core`** — would inflate the DAG root with
   composition concerns; the workspace decomposition (ADR-0001) split them
   intentionally. Rejected.
3. **No adapter; users manually wrap** — every Phase-1 example would have
   bespoke code to integrate a tool into a chain. Rejected — bad ergonomics.
4. **`typed-builder` 0.20** — viable, but no ergonomic edge over `bon`,
   and `cargo public-api` output is noisier. Rejected.

## Implementation order (Phase 1)

1. `entelix-core`: `Error`, `ExecutionContext`, `Tool` trait, `Hook`,
   `EventBus`, IR, codecs, transports, auth.
2. `entelix-runnable`: `Runnable<I, O>`, `RunnableExt::pipe`,
   `RunnableSequence`, `RunnableLambda`, `RunnablePassthrough`,
   `ToolToRunnableAdapter`, `AnyRunnable`.
3. `entelix-prompt`: `PromptTemplate`, `ChatPromptTemplate`,
   `MessagesPlaceholder`.

## References

- ADR-0001 — workspace structure (DAG roots)
- ADR-0006 — Runnable + StateGraph 1.0 spine
- ADR-0010 — naming taxonomy (builder convention)
- `docs/architecture/runnable-and-lcel.md` §"Tool-vs-Runnable relationship"
- `docs/architecture/managed-agents.md` §"Hand contract"
- bon docs — <https://bon-rs.com/>
