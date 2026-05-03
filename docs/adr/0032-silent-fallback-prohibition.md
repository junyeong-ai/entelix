# ADR 0032 — Silent fallback prohibition (invariant #15)

**Status**: Accepted
**Date**: 2026-04-30
**Decision**: Phase 1 of the post-7-차원-audit roadmap

## Context

A 7-차원 audit of the workspace surfaced a recurring class of defects: codecs and the cost meter coercing missing or unexpected vendor signals into a benign-looking IR value. Examples uncovered:

- `Anthropic` `max_tokens` defaulted to `4096` when the IR carried `None` — silent truncation surfaces as `stop_reason: max_tokens` and the caller cannot tell the SDK, not the vendor, set the cap.
- `Bedrock` / `Gemini` / `OpenAI Responses` decoders mapped `stopReason: None` (or `finishReason: None`) to `StopReason::EndTurn` — a truncated stream looks identical to a clean completion.
- `Bedrock` `stop_sequence` decoded with `sequence: String::new()` — the matched string was dropped without warning.
- `OpenAI Responses` `tool_use_seen` precedence over `status: "incomplete"` — a partial tool_use truncated by the token cap surfaced as `ToolUse` and ReAct loops re-entered indefinitely.
- `ModelPricing.cache_read_per_1k = None` fell back to `input_per_1k / 10` — `gen_ai.usage.cost` reported a wrong number for every vendor whose cache-read rate is not exactly 10% of input.
- `CacheTtl::OneHour` rendered as `{type: "1h"}` (vendor expects `{type: "ephemeral", ttl: "1h"}`) — the wire was wrong but no warning fired.

Each individual case was small. Aggregated, they violated invariant 6 ("Lossy encoding emits warnings") *in spirit but not in code*: the codecs technically did not "drop" information, they replaced it with a plausible default. Operators still lost the truth.

## Decision

Add invariant **#15 — Silent fallback prohibition** to CLAUDE.md and enforce it across codecs, transports, and the cost meter.

> **15. No silent fallback** — codecs and transports must surface every information-loss event through one of two channels: `ModelWarning::LossyEncode { field, detail }` for coerced values, or `StopReason::Other { raw }` for unknown vendor reasons. Default-injecting a value (e.g. `max_tokens.unwrap_or(4096)`, `cache_rate.unwrap_or(input/10)`, `stop_reason.unwrap_or(EndTurn)`) is a bug regardless of how reasonable the default looks. Vendor-mandatory IR fields are rejected at encode time with `Error::invalid_request`; missing decode signals surface as `Other{raw:"missing"}` plus a `LossyEncode` warning.

### Concrete contracts

| Site | Old behaviour | New behaviour |
|---|---|---|
| `AnthropicMessagesCodec::encode` (`max_tokens`) | injected `4096` default | `Error::invalid_request("max_tokens required")` |
| `Bedrock` / `Gemini` / `OpenAI Responses` decode missing `finish_reason` | `EndTurn` | `Other{raw:"missing"}` + `LossyEncode { field: "finish_reason" }` |
| `Bedrock` decode `stop_sequence` | `StopSequence{sequence:""}` | preserve from `additionalModelResponseFields.stop_sequence` when present, else `Other{raw:"stop_sequence"}` + `LossyEncode` |
| `OpenAI Responses` `incomplete` + partial `tool_use` | `ToolUse` (status dropped) | `Other{raw:"tool_use_truncated"}` + `LossyEncode { field: "stop_reason" }` |
| `StopReason::Refusal` | bare variant | `Refusal { reason: RefusalReason }` — Safety / Recitation / Guardrail / ProviderFailure / Other |
| `ModelPricing` cache rates | `Option<Decimal>` with fallback math | mandatory `Decimal` arguments to `ModelPricing::new(input, output, cache_read, cache_write)`; vendors that don't charge pass `Decimal::ZERO` |
| `Anthropic` `cache_control` wire | `{type: "1h"}` (wrong) | `{type: "ephemeral", ttl: "1h"}` per vendor spec; 5m default omits `ttl` sibling |

### Enforcement

- **Code-level**: every coerced value path must `warnings.push(LossyEncode {...})`. Mandatory-vendor-field omissions raise `Error::invalid_request` at encode time.
- **Test-level**: `tests/codec_consistency_matrix.rs` adds three regression checks:
  1. every codec maps unknown finish_reason to `Other` + `UnknownStopReason`;
  2. every codec maps missing finish_reason to `Other{raw:"missing"}` + `LossyEncode`;
  3. every codec preserves the matched stop_sequence string (or warns when the wire format hides it).
- **Script-level**: `scripts/check-silent-fallback.sh` greps `crates/entelix-core/src/codecs/`, `crates/entelix-core/src/transports/`, and `crates/entelix-policy/src/cost.rs` for `unwrap_or` / `unwrap_or_default` / `unwrap_or_else` patterns and rejects any not on the explicit allow-list (e.g. `field.unwrap_or("")` for an empty-string default that is itself a vendor wire convention). New silent-fallback sites cannot ship without an audit.

## Consequences

✅ Operators see the truth — `gen_ai.client.cost`, `stop_reason`, and `cache_read_input_tokens` all reflect what the vendor actually returned, not what the SDK guessed.
✅ Truncated streams stop being indistinguishable from clean completions; ReAct loops can no longer re-enter on a partial tool_use.
✅ `Refusal` is structured — observability dashboards can split safety vs. recitation vs. guardrail blocks.
❌ Callers must now supply `max_tokens` to Anthropic explicitly (one-liner). Previously-passing test fixtures fail with `InvalidRequest("max_tokens required")` until updated.
❌ `ModelPricing` requires four `Decimal` arguments instead of two; existing pricing tables need migration. The new `ModelPricing::new` constructor makes the rates explicit, which is the point.
❌ Codecs adding new vendor signals must opt every missing-or-unknown branch through `Other` + `LossyEncode`; the script gate enforces this.

## Alternatives considered

1. **Loosen invariant 6 to allow "reasonable defaults"** — undermines the audit signal entirely; rejected.
2. **Keep silent fallbacks but emit `tracing::warn!`** — invisible to consumers building on the IR; rejected. The warning channel must be the typed `ModelWarning`.
3. **Per-codec opt-in fallback policy** — multiplies surface area and still hides the loss from observability backends; rejected.

## References

- 7-차원 audit (Phase 1 finding sets H1–H6 in `audit-heuristic-risks` fork report).
- ADR-0006 — invariant 6 ("Lossy encoding emits warnings").
- ADR-0028 — `RetryClassifier` policy externalization (the same "external policy beats hidden default" principle, Phase 3).
