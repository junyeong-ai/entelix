# ADR 0091 — `OutputValidator<O>` trait + `complete_typed_validated`

**Status**: Accepted
**Date**: 2026-05-06
**Decision**: A new `OutputValidator<O>: Send + Sync + 'static` trait — with a blanket impl for `Fn(&O) -> Result<()>` so closures fit — covers post-decode semantic validation that the JSON schema cannot express. `ChatModel::complete_typed_validated<O, V>(messages, validator, ctx)` mirrors `complete_typed<O>` and adds a validator step after parse: when the validator returns `Err(Error::ModelRetry { hint, .. })` the same retry loop reflects the hint to the model and re-invokes within the existing `validation_retries` budget. Other error variants from the validator bubble unchanged. `String` and `&str` gain blanket `LlmRenderable<String>` impls (identity rendering) so validators raising `Error::ModelRetry` write `"hint text".to_owned().for_llm()` ergonomically.

## Context

Slice 106 (ADR-0090) added `Error::ModelRetry` and the schema-mismatch retry loop. The variant existed but was unreachable from operator code at the chat-model layer — only the `complete_typed<O>` parse path raised it (via `Error::Serde`). Operators that wanted to enforce semantic constraints ("score must be 0-100", "answer must reference one of these document IDs") had to write their own retry loop.

pydantic-ai's `@agent.output_validator(...)` decorator is the canonical pattern: the validator runs after parse, raising `ModelRetry(message)` triggers the same retry the model gets for schema mismatches. Slice 106b ports the shape to the chat-model layer.

## Decision

### Trait surface

```rust
pub trait OutputValidator<O>: Send + Sync + 'static
where
    O: Send + 'static,
{
    fn validate(&self, output: &O) -> Result<()>;
}

impl<O, F> OutputValidator<O> for F
where
    O: Send + 'static,
    F: Fn(&O) -> Result<()> + Send + Sync + 'static,
{
    fn validate(&self, output: &O) -> Result<()> {
        self(output)
    }
}
```

Sync (not async) — operators wanting to `.await` inside validation compose around the `complete_typed_validated` call boundary instead. The 95% case (range checks, regex matches, business-rule branches) is sync; pulling `async-trait` into the validator surface for the async 5% would impose ceremony on every call site.

The blanket impl over closures means most operators write a one-liner; an explicit impl is the path for stateful validators that share state across calls (counters, caches).

### `complete_typed_validated`

```rust
pub async fn complete_typed_validated<O, V>(
    &self,
    messages: Vec<Message>,
    validator: V,
    ctx: &ExecutionContext,
) -> Result<O>
where
    O: schemars::JsonSchema + serde::de::DeserializeOwned + Send + 'static,
    V: OutputValidator<O>,
{
    // Same loop as complete_typed; after parse:
    //   match validator.validate(&value) {
    //       Ok(())                                 => return Ok(value),
    //       Err(Error::ModelRetry { hint, .. })    => retry within budget,
    //       Err(other)                              => return Err(other),
    //   }
}
```

Schema-mismatch retries (`Error::Serde`) and validator-driven retries (`Error::ModelRetry`) share the `validation_retries` budget. Both signal "the model emitted output we can't accept"; distinguishing them at the budget level adds knobs without buying behaviour operators commonly want to vary independently.

When the budget exhausts, the validator's last `Error::ModelRetry` surfaces unchanged — operators see the typed variant and can branch on it without parsing strings.

### `LlmRenderable<String>` for `String` / `&str`

```rust
impl LlmRenderable<String> for String { … }   // identity
impl LlmRenderable<String> for &str { … }     // identity (allocating)
```

Required so validators write `"hint text".to_owned().for_llm()` to produce the `RenderedForLlm<String>` that `Error::model_retry` expects. The seal is preserved — `for_llm`'s default impl (the only `RenderedForLlm::new` caller) routes every emit through the trait, even when the operator's hint is already a plain string.

## Why share the budget with schema-mismatch retries

Two practical reasons:

1. **Operator mental model**: "the model gets N attempts to produce output I can use" is one knob, not two. Splitting `validation_retries` into `schema_retries` + `validator_retries` doubles the surface for negligible gain.
2. **Correctness budget**: each retry costs tokens and wall-clock latency. A run that burns all retries on schema mismatches still owes the operator a correct output; doubling the budget for the validator path silently doubles the worst-case spend.

Operators that want differentiated budgets implement two `complete_typed_validated` calls — first round-trip handles parse + business-rule check, second handles a separate validator with its own budget. The tooling expression is composable.

## Why not `OutputValidator<O, D>`

`D` (operator-side typed deps) would let validators reach a DB pool / HTTP client. But:

- `complete_typed_validated` is on `ChatModel`, which is D-free — adding `D` to the validator surface forces ChatModel to either be generic on D (a major refactor) or accept `&AgentContext<D>` parameters (changes the dispatch signature).
- The 95% case for validators is pure (range / regex / cross-field). Stateful validators that need deps thread state through closure capture; the operator owns the lifetime.
- A `D`-aware validator surface lands at the *agent* level (slice 104+ `Agent<S, D>`) where deps already flow through. ChatModel-level validation stays sync + D-free.

## Consequences

- 1 new public trait (`OutputValidator<O>`) + 2 blanket impls (`String`, `&str` for `LlmRenderable<String>`).
- 1 new method (`ChatModel::complete_typed_validated<O, V>`).
- 4 new regression tests (accept-first, retry-recovers, exhausted-surfaces-ModelRetry, non-retry-error-bubbles).
- `entelix-core` public-API baseline refreshed.
- Closes the slice 106 follow-up identified in the slice memory entry.

## References

- ADR-0079 — `OutputStrategy` + `complete_typed`.
- ADR-0090 — `Error::ModelRetry` + `complete_typed` validation loop (the schema-mismatch path that this ADR extends).
- ADR-0033 / ADR-0076 — `LlmRenderable` funnel; the `String` blanket impl preserves the seal.
- pydantic-ai `@agent.output_validator` — reference shape.
