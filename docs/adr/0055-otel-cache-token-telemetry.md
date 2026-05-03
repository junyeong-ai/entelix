# ADR 0055 — OpenTelemetry cache + reasoning token telemetry

**Status**: Accepted
**Date**: 2026-05-01
**Decision**: Phase 8 of the post-7-차원-audit roadmap (third sub-slice)

## Context

`Usage` has carried `cached_input_tokens`,
`cache_creation_input_tokens`, and `reasoning_tokens` since
the IR landed — every shipping codec populates them
(Anthropic `cache_read_input_tokens` /
`cache_creation_input_tokens`, Bedrock `cacheReadInputTokens`
/ `cacheWriteInputTokens`, OpenAI `cached_tokens`, Anthropic
thinking budgets, OpenAI o-series reasoning).

The OTel layer ignored them. `GenAiMetrics::record_call`
emitted samples only for `input_tokens` and `output_tokens`;
`gen_ai.response` tracing events surfaced the same two fields
plus `gen_ai.usage.cost`. That's a ~3-4× under-reporting of
billable activity for any deployment using prompt caching or
reasoning models — and worse, it makes cache-hit-rate
optimisation invisible to operators.

Anthropic's prompt caching charges cache writes at ~25% above
input rate and serves cache reads at ~10% of input rate.
Operators tuning prompt structure for cache locality need to
*see* the read/write split per call to know whether their
prompts are getting amortised. Without telemetry, they're
flying blind.

The roadmap §S9 (Phase 8) called this out explicitly:
"sampling/elicitation/JSON-RPC batch + agent OTel span +
**cache token telemetry** + tool I/O event body". This slice
ships the cache-token half.

## Decision

Three new token-type values on the existing
`gen_ai.client.token.usage` histogram, three new attribute
names for the `gen_ai.response` tracing event, three new
semconv constants. No new histograms — the standard semconv
already specifies a single histogram tagged by
`gen_ai.token.type`, and the dashboard convention is to slice
that histogram by token type.

### New semconv constants

```rust
// Attribute key (pre-existing tag, now constant-named):
pub const TOKEN_TYPE: &str = "gen_ai.token.type";

// Standard values:
pub const TOKEN_TYPE_INPUT: &str = "input";
pub const TOKEN_TYPE_OUTPUT: &str = "output";
pub const TOKEN_TYPE_CACHED: &str = "cached";  // semconv 0.32

// Entelix-specific extension values:
pub const TOKEN_TYPE_CACHE_CREATION: &str = "cache_creation";
pub const TOKEN_TYPE_REASONING: &str = "reasoning";

// Tracing-event attribute names:
pub const USAGE_CACHED_INPUT_TOKENS: &str = "gen_ai.usage.cached_input_tokens";
pub const USAGE_CACHE_CREATION_INPUT_TOKENS: &str = "gen_ai.usage.cache_creation_input_tokens";
pub const USAGE_REASONING_TOKENS: &str = "gen_ai.usage.reasoning_tokens";
```

`TOKEN_TYPE_CACHED` aligns with semconv 0.32 (released after
our 0.31 baseline). `TOKEN_TYPE_CACHE_CREATION` and
`TOKEN_TYPE_REASONING` are entelix-specific extensions —
semconv has no standard value for cache writes or reasoning
budgets yet, so operators filter on the literal string at
query time. The constant module is the single source of truth
so any future semconv standardisation is a one-line rename
(invariant: "no string literals like `gen_ai.system` inline
in layer code").

### `GenAiMetrics::record_call` extension

`TokenKind` enum gains three variants (`Cached`,
`CacheCreation`, `Reasoning`); `record_call` emits one sample
per non-zero counter. Zero-token branches still skip emission
(the original "don't pollute dashboards with constant 0
series" rule applies uniformly).

```rust
if usage.cached_input_tokens > 0 {
    self.record_tokens(&attrs, TokenKind::Cached, ...);
}
if usage.cache_creation_input_tokens > 0 {
    self.record_tokens(&attrs, TokenKind::CacheCreation, ...);
}
if usage.reasoning_tokens > 0 {
    self.record_tokens(&attrs, TokenKind::Reasoning, ...);
}
```

### `gen_ai.response` tracing event extension

The `OtelLayer` model-side `tracing::event!` call gains three
fields:

```rust
gen_ai.usage.cached_input_tokens = response.usage.cached_input_tokens,
gen_ai.usage.cache_creation_input_tokens = response.usage.cache_creation_input_tokens,
gen_ai.usage.reasoning_tokens = response.usage.reasoning_tokens,
```

Spans now carry the full token breakdown — operators tracing
a request through their backend see input / output / cached /
cache-write / reasoning side-by-side, not just input + output.

### Why one histogram, not three

The semconv ships exactly one `gen_ai.client.token.usage`
histogram tagged by `gen_ai.token.type`. Splitting into three
histograms would:

- Break the standard query convention (operators sum across
  the tag at dashboard time; three histograms force a
  hand-rolled join).
- Force an OTel SDK change for every operator dashboard the
  moment they wanted "total tokens" across kinds.
- Diverge from the upstream semconv direction where additional
  token types land as new tag values, not new instruments.

Single histogram + per-kind tag values is the right shape.

### Why "off when zero" emission

Two reasons:

1. Constant-zero metric series pollute dashboards (they look
   like "I always have zero cache hits!" when in reality the
   deployment never opted into caching). The existing
   `input_tokens > 0` / `output_tokens > 0` guards already
   honour this.
2. Histograms with constant-zero samples slightly distort
   percentile calculations — small effect, but real.

The tracing-event fields (`gen_ai.usage.cached_input_tokens =
0` etc.) emit unconditionally because span attributes are
*always* present per call — operators see "0 cache reads on
this call" as informative, where the metric histogram should
not.

### Tests

- 1 new unit test: `cache_and_reasoning_token_kinds_emit_when_non_zero`
  — guards the dispatch path through cache + reasoning kinds.
- 1 new unit test: `token_kind_strings_align_with_semconv_constants`
  — pins each enum variant's string output to its semconv
  constant (catches future drift).
- Existing `metrics_construct_against_global_meter` and
  `zero_tokens_emit_no_token_sample` continue to pass —
  zero-counter behaviour is preserved.

## Consequences

✅ Operators can finally measure cache hit rate per
(model, namespace, etc.) by querying
`sum(gen_ai.client.token.usage{gen_ai.token.type="cached"}) /
sum(gen_ai.client.token.usage{gen_ai.token.type="input"})`.
Cost optimisation (prompt restructuring for locality)
becomes data-driven instead of guesswork.
✅ Reasoning tokens are isolated as their own series — easy
to spot when a reasoning model burns more on internal
deliberation than on visible output.
✅ Tracing spans surface the full token breakdown — operators
debugging a single slow / expensive call see input / output
/ cached / cache-write / reasoning side by side without
cross-referencing metrics.
✅ Single histogram + tag values aligns with semconv direction
— future standardisation drops or renames the entelix-specific
extensions without breaking the histogram contract.
✅ Constant-named token types prevent the "drift on attribute
strings" silent bug class (entelix-otel CLAUDE.md rule).
❌ `cache_creation` and `reasoning` token type values aren't
in the upstream semconv yet — operators federating across
multiple SDKs will see only entelix's emissions for these
buckets. Documented as the trade-off; unblocks the operator
visibility today rather than waiting for spec convergence.
❌ Public-API baseline for `entelix-otel` grew (5 new
constants in `semconv`). Refrozen.

## Alternatives considered

1. **Three separate histograms**
   (`gen_ai.client.cached_token.usage`, etc.) — breaks the
   standard semconv shape, forces dashboard rewrites, denies
   operators the existing "sum across `gen_ai.token.type`"
   dashboard convention. Rejected.
2. **Single token bucket aggregated server-side, tagged
   per-kind on the response event only** — half-measure that
   gives operators tracing visibility but no metric
   aggregation. Rejected; metric aggregation is the harder
   problem.
3. **Wait for semconv 0.32+ adoption to land
   `cache_creation` / `reasoning` standard values** —
   operator visibility is needed today; the entelix extension
   strings can rename trivially when standards land.
   Rejected.
4. **Sub-histograms via OpenTelemetry views (per-tag
   filtering)** — operators with that level of OTel
   sophistication can already do it; the SDK-shipped instrument
   stays simple. Rejected as default.

## Operator usage patterns

**Cache hit rate dashboard query** (Prometheus-style):
```
sum by (gen_ai_request_model) (
  rate(gen_ai_client_token_usage_total{gen_ai_token_type="cached"}[5m])
) /
sum by (gen_ai_request_model) (
  rate(gen_ai_client_token_usage_total{gen_ai_token_type="input"}[5m])
)
```

**Cache write amortisation** (writes vs reads, expecting
reads >> writes for well-cached prompts):
```
sum(rate(gen_ai_client_token_usage_total{gen_ai_token_type="cached"}[1h])) /
sum(rate(gen_ai_client_token_usage_total{gen_ai_token_type="cache_creation"}[1h]))
```

**Reasoning vs visible output** (reasoning-heavy models):
```
sum(rate(gen_ai_client_token_usage_total{gen_ai_token_type="reasoning"}[5m])) /
sum(rate(gen_ai_client_token_usage_total{gen_ai_token_type="output"}[5m]))
```

**Per-call investigation** (tracing): the `gen_ai.response`
event carries `gen_ai.usage.cached_input_tokens`,
`gen_ai.usage.cache_creation_input_tokens`,
`gen_ai.usage.reasoning_tokens` — surface them in the span
viewer alongside `gen_ai.usage.cost`.

## References

- ADR-0009 — OpenTelemetry GenAI semconv adoption (parent).
- 7-차원 roadmap §S9 — Phase 8 (MCP + OTel completion), third
  sub-slice.
- OpenTelemetry GenAI semconv 0.31 / 0.32 — the upstream
  spec the layer tracks.
- `crates/entelix-core/src/ir/usage.rs` — `Usage` struct
  carrying the cache + reasoning fields populated by every
  codec.
- `crates/entelix-otel/src/semconv.rs` — 5 new constants.
- `crates/entelix-otel/src/metrics.rs` — `TokenKind`
  extension + per-kind sampling + 2 new tests.
- `crates/entelix-otel/src/layer.rs` — `gen_ai.response`
  event extended with 3 cache/reasoning fields.
