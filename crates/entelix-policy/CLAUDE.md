# entelix-policy

Multi-tenant operational primitives. Composes through `tower::Layer<S>` over `entelix_core` service spine.

## Surface

- **`RateLimiter` trait** + `TokenBucketLimiter` — async, per-key, time-injectable. Time abstraction (`Clock` trait + `SystemClock`) lives in `entelix-core::time` so any sub-crate can take a time source without depending on `entelix-policy`.
- **`PiiRedactor` trait** + `RegexRedactor` — bidirectional. Runs `pre_request` AND `post_response`.
- **`CostMeter`** + `PricingTable` / `ModelPricing` — `rust_decimal` arithmetic (no float). `charge(tenant, model, usage)` runs only after the response decoder succeeds (transactional, invariant 12).
- **`QuotaLimiter`** — composite gate: rate (RPS) + budget ceiling (per-tenant cumulative spend cap). Runs *before* the request.
- **`TenantPolicy`** — per-tenant aggregate of optional handles (`rate_limiter`, `redactor`, `cost_meter`, `quota`).
- **`PolicyRegistry`** — `DashMap<tenant_id, Arc<TenantPolicy>>` with fallback default policy.
- **`PolicyLayer<S>`** — `tower::Layer` that wraps both `Service<ModelInvocation>` (request scrub + response scrub + cost) AND `Service<ToolInvocation>` (input scrub + output scrub). One layer instance, two service shapes.

## Crate-local rules

- **Cost emission is transactional — `Ok` branch only.** `gen_ai.usage.cost` / `gen_ai.tool.cost` / `gen_ai.embedding.cost` never appear on the error branch (invariant 12). Mirror coverage exists in `entelix-otel`'s `OtelLayer` and `entelix-memory`'s `MeteredEmbedder`.
- **Redaction is bidirectional.** A new `PiiRedactor` impl that only scrubs requests (or only responses) leaks PII on the unscrubbed side. Test both directions.
- `rust_decimal` everywhere for currency — never `f64`. Float drift in cumulative cost is silent and unforgivable.
- New gate (e.g. concurrency limit): add to `QuotaLimiter` as another optional component, not a parallel pre-request gate. One pre-request decision per layer.

## Forbidden

- Charging cost on a failed call (any non-`Ok` branch). Tests assert no `gen_ai.*.cost` attribute is recorded on the error branch.
- One-directional `PiiRedactor` impls.
- `f64` arithmetic on currency.
- Bypassing `PolicyLayer` for the tool invocation side ("tools don't need redaction") — tool input/output is also user-visible PII surface.

