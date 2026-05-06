# ADR 0096 — `ChatModel::stream_typed<O>` typed structured-output streaming

**Status**: Accepted
**Date**: 2026-05-06
**Decision**: Add `ChatModel::stream_typed<O>(messages, ctx) -> Result<TypedModelStream<O>>`. The new `TypedModelStream<O>` carries `stream: BoxDeltaStream<'static>` for raw deltas (text fragments operators echo to a UI as the model produces tokens) and `completion: BoxFuture<'static, Result<O>>` for the typed payload parsed once the stream drains. `stream_typed` does **not** retry on parse failure — by the time `completion` resolves the deltas have already been surfaced to the consumer, so re-invoking with a corrective hint would emit a divergent second stream. Operators that want the unified retry budget call `complete_typed` / `complete_typed_validated` instead.

## Context

Slice 106 + 106b shipped `complete_typed<O>` and `complete_typed_validated<O, V>` — the one-shot typed surface with the validation_retries loop (ADR-0090 + ADR-0091, invariant 20). The streaming counterpart was missing: operators that wanted to display intermediate text deltas to a user *and* parse the final response into a typed `O` had to:

- Manually call `stream_deltas`, accumulate text fragments themselves, and then call `serde_json::from_str` against a JSON-schema response that the wire didn't carry typed metadata for.
- OR call `complete_typed` (no streaming) and miss the incremental UX.

That gap meant the typed surface was effectively a one-shot-only contract. Phase A's typed-output story stayed half-shipped without the streaming sibling.

## Decision

### Surface

```rust
impl<C: Codec + 'static, T: Transport + 'static> ChatModel<C, T> {
    pub async fn stream_typed<O>(
        &self,
        messages: Vec<Message>,
        ctx: &ExecutionContext,
    ) -> Result<TypedModelStream<O>>
    where
        O: JsonSchema + DeserializeOwned + Send + 'static;
}

pub struct TypedModelStream<O> {
    pub stream: BoxDeltaStream<'static>,
    pub completion: BoxFuture<'static, Result<O>>,
}
```

### How it works

- `response_format = ResponseFormat::strict(JsonSchemaSpec::for::<O>())` is set on the request, identical to `complete_typed`. The model emits the typed JSON via either Native strategy (text deltas the consumer accumulates) or Tool strategy (a single `tool_use` block).
- The streaming dispatch path runs through `streaming_service().oneshot(invocation)` — same `tower::Service` spine the `OtelLayer` and `PolicyLayer` already wrap.
- The returned `ModelStream { stream, completion }` is mapped: `stream` passes through unchanged (consumer sees the raw deltas); `completion` is wrapped with `parse_typed_response::<O>(response)` so the caller awaits typed `O` instead of a raw `ModelResponse`.
- Run-budget observation lands inside the `Ok` branch of the wrapped completion — invariant 12 (cost is computed transactionally) holds end-to-end: a stream that errors mid-flight surfaces the error on `completion` and never charges.

### Why no retry

`complete_typed` retries on schema mismatch by re-invoking the model and reflecting the parse diagnostic to it as a corrective `User` message. That works because the previous attempt produced no consumer-visible output — the failed `ModelResponse` is internal. With streaming, the deltas have *already* been emitted to the UI by the time the parse runs. Re-invoking would either:

- Produce a second divergent stream the operator now has to splice over the first (UX disaster), or
- Buffer the entire stream until `completion` succeeds (defeating the streaming benefit).

Neither is acceptable. The clean choice is "no retry on the streaming path" — surface the parse failure on `completion`, document the tradeoff, and tell operators wanting retry to call `complete_typed`. The `validation_retries` config continues to govern `complete_typed` only; passing it on a `ChatModel` used for `stream_typed` is silently ignored (the doc-comment names this).

### Why no `stream_typed_validated`

A semantic validator runs *after* the parsed `O` is available — it can be applied at the call site:

```rust
let typed_stream = model.stream_typed::<MyOutput>(msgs, &ctx).await?;
// drain `stream`, surface deltas to UI ...
let value: MyOutput = typed_stream.completion.await?;
my_validator.validate(&value)?;  // operator handles validator failure
```

A built-in `stream_typed_validated` would either re-invoke (same retry problem above) or just call the validator after `completion` (which the operator can do themselves in one line). The added surface doesn't earn its weight.

## Test coverage

`crates/entelix-core/tests/stream_typed.rs` (3 regression tests):

1. `stream_typed_resolves_completion_to_typed_value` — happy path: stream drains, completion resolves to `Ok(O)`.
2. `stream_typed_schema_mismatch_surfaces_serde_error` — schema mismatch surfaces on `completion`; `validation_retries` is **not** honoured (proves the no-retry contract).
3. `stream_typed_emits_deltas_through_stream_field` — `stream` carries `Start` + `TextDelta` + `Stop` to the consumer; the typed completion's `O` arrives independently.

## Public-API impact

`entelix-core` baseline grows by 26 lines:
- `ChatModel::stream_typed<O>` method.
- `pub struct TypedModelStream<O>` + its `stream` / `completion` fields + `Debug` impl.

Facade `entelix` re-exports `TypedModelStream`. Bump captured in the `entelix-core.txt` baseline.

## Consequences

- The typed-output surface is now feature-complete: one-shot (`complete_typed`), one-shot-with-validator (`complete_typed_validated`), streaming (`stream_typed`).
- Operators wanting validation retry stay on `complete_typed`; operators wanting incremental display use `stream_typed`. The split is honest about the tradeoff instead of pretending streaming + retry compose.
- ADR-0090 + ADR-0091 are not affected — they cover the one-shot retry channel, which `stream_typed` deliberately doesn't enter.

## References

- ADR-0079 — `OutputStrategy` + `complete_typed<O>` (initial typed-output cut).
- ADR-0090 — `Error::ModelRetry` + `complete_typed` validation_retries loop.
- ADR-0091 — `OutputValidator<O>` + `complete_typed_validated`.
- v3 plan slice 119 (Phase B-5).
