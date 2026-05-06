# ADR 0090 — `Error::ModelRetry` + `complete_typed<O>` validation loop

**Status**: Accepted
**Date**: 2026-05-06
**Decision**: A new `Error::ModelRetry { hint: RenderedForLlm<String>, attempt: u32 }` variant carries model-driven retry signals — distinct from `Error::Provider` (transport retries) and `Error::InvalidRequest` (operator misuse) so retry classifiers, OTel dashboards, and budget meters branch on a typed channel. `ChatModelConfig::validation_retries: u32` (default `0`) configures the budget; `ChatModel::with_validation_retries(n)` builds it. `complete_typed<O>` catches `Error::Serde` parse failures, reflects the diagnostic to the model as a corrective user message (assistant's failed turn echoed back, then a `User` message with the schema-mismatch text), and re-invokes up to `validation_retries` times before bubbling the final error. Closes the slice 190 (lost in the 175-195 incident, F-19) work.

## Context

Modern agent SDKs all surface model-driven retry as a first-class feature:

- **pydantic-ai**: `ModelRetry(message)` exception inside an output validator triggers automatic re-prompting.
- **Vercel AI SDK 5**: `generateObject({ schema, retries: 2 })` retries on validation failure.
- **Instructor (Python)**: `max_retries` argument with reflection-style re-prompting.
- **Mirascope**: `@validate` decorator with retry budget.

In every case the same shape: typed retry signal, budget on the call, automatic conversation augmentation (failed reply + correction prompt) for the next attempt. The slice 190 work that landed this for entelix was lost in the 175-195 incident; ADR-0090 re-implements it cleanly under the slice 99/100/103 carrier groundwork (`AgentContext<D>`, `Tool<D>`, `ToolRegistry<D>`).

## Decision

### `Error::ModelRetry { hint, attempt }`

```rust
pub enum Error {
    // ...
    ModelRetry {
        hint: crate::llm_facing::RenderedForLlm<String>,
        attempt: u32,
    },
    Serde(#[from] serde_json::Error),
    // ...
}

impl Error {
    pub const fn model_retry(
        hint: crate::llm_facing::RenderedForLlm<String>,
        attempt: u32,
    ) -> Self {
        Self::ModelRetry { hint, attempt }
    }
}
```

`hint: RenderedForLlm<String>` enforces invariant 16 at the type level — operators raising this variant must route the message through the LLM-rendering funnel, not copy a vendor-side error string verbatim. `attempt: u32` is self-describing so callers don't track retry state externally.

`Display` / `LlmRenderable` mirror the variant: `Display` shows `"model retry requested (attempt N)"` for operator-side logs; `render_for_llm` surfaces the `hint` text (used when the variant leaks past the retry loop, which should be rare).

### `validation_retries` config

```rust
#[non_exhaustive]
pub struct ChatModelConfig {
    // ...
    validation_retries: u32, // default 0
}

impl ChatModel<C, T> {
    pub const fn with_validation_retries(mut self, n: u32) -> Self {
        self.config.validation_retries = n;
        self
    }
}
```

Default `0` — explicit opt-in matches invariant 15's "no silent fallback" stance. Operators set `1`–`3` for typical LLM correction loops; higher counts inflate token spend without meaningful recovery probability.

### `complete_typed<O>` retry loop

```rust
pub async fn complete_typed<O>(
    &self,
    messages: Vec<Message>,
    ctx: &ExecutionContext,
) -> Result<O>
where
    O: schemars::JsonSchema + serde::de::DeserializeOwned + Send + 'static,
{
    let mut conversation = messages;
    let max_retries = self.config.validation_retries;
    let mut attempt: u32 = 0;
    loop {
        // ... build request, dispatch, observe usage ...
        let assistant_text = response_text_for_retry(&response);
        match parse_typed_response::<O>(response) {
            Ok(value) => return Ok(value),
            Err(err) if matches!(err, Error::Serde(_)) && attempt < max_retries => {
                attempt += 1;
                let parse_diagnostic = err.to_string();
                conversation.push(Message::new(Role::Assistant, vec![ContentPart::Text { text: assistant_text.unwrap_or_default(), .. }]));
                conversation.push(Message::new(Role::User, vec![ContentPart::Text {
                    text: format!(
                        "Your previous response did not match the required JSON schema for `{short_name}`. \
                         Parser diagnostic: {parse_diagnostic}\n\
                         Re-emit the response as a single valid JSON object that conforms to the schema."
                    ),
                    ..
                }]));
            }
            Err(err) => return Err(err),
        }
    }
}
```

Three behaviours surfaced by the regression test suite:

1. **`validation_retries = 0`** — first parse failure surfaces unchanged. Single transport hit.
2. **`validation_retries = N`, second attempt valid** — first call fails parse, conversation appends `(assistant: bad reply, user: corrective prompt)`, second call succeeds. Two transport hits.
3. **`validation_retries = N`, all attempts fail** — `N+1` transport hits, final `Error::Serde` surfaces.

The retry loop only catches `Error::Serde`. Transport errors (`Error::Provider`), budget breaches (`Error::UsageLimitExceeded`), and credential failures (`Error::Auth`) bubble unchanged — those go through their own classifiers (slice 100 `RetryService`, ADR-0034).

### Why echo the assistant's failed turn

The LLM correction loop benefits from the model seeing its own bad output. Without the echo, the corrective user message has no context — "fix this JSON" but no JSON in scope. With the echo, the model has: its own bad attempt + the parser diagnostic + a clear instruction to re-emit. This matches Instructor's reflection pattern and Vercel AI SDK 5's `experimental_repairText` shape.

## Why `Error::ModelRetry` is not yet wired into `complete_typed`

Slice 106's primary deliverable is the schema-level retry on `Error::Serde`. The `Error::ModelRetry` variant exists in the type system for tools / hooks / future `OutputValidator` impls to raise; the retry loop in `complete_typed` does not yet catch it (that's the typed-validator extension in a later slice). Wiring `OutputValidator<O, D>` to raise `ModelRetry` and have the same loop catch it is mechanical once the trait lands.

## Consequences

- 9th `Error` variant (`InvalidRequest`, `Config`, `Provider`, `Cancelled`, `DeadlineExceeded`, `Interrupted`, `Serde`, `Auth`, `UsageLimitExceeded`, **`ModelRetry`**). Two pattern matches updated (`llm_facing.rs::render_for_llm`, `stream.rs::clone_error`).
- `ChatModelConfig::validation_retries` adds 4 bytes; the config stays `Clone + Debug`.
- 3 regression tests in `crates/entelix-core/tests/complete_typed_retry.rs` covering the three behavioural paths.
- 1 public-API baseline refreshed (`entelix-core`).

## References

- ADR-0079 — `OutputStrategy` + `complete_typed` typed structured output.
- ADR-0033 / ADR-0076 — `LlmRenderable` funnel; `Error::ModelRetry::hint` requires the typed `RenderedForLlm<String>` carrier.
- v3 plan slice 106; lost slice 190 (F-19) re-implementation.
