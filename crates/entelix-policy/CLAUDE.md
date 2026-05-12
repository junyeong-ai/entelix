# entelix-policy

Multi-tenant operational primitives. Composes through `tower::Layer<S>` over `entelix_core` service spine.

## Surface

- **`RateLimiter` trait** + `TokenBucketLimiter` — async, per-key, time-injectable. Time abstraction (`Clock` trait + `SystemClock`) lives in `entelix-core::time` so any sub-crate can take a time source without depending on `entelix-policy`.
- **`PiiRedactor` trait** + `RegexRedactor` — bidirectional. Runs `pre_request` AND `post_response`.
- **`CostMeter`** + `PricingTable` / `ModelPricing` — `rust_decimal` arithmetic (no float). `charge(tenant, model, usage)` runs only after the response decoder succeeds (transactional, invariant 12). `UnknownModelPolicy::{Reject, WarnOnce}` controls behaviour when `model` is absent from the pricing table; `UnknownModelSink` trait (sync `&self`, `record_unknown_model(&TenantId, &str)` — `AuditSink`-style per invariant 18) is wired via `CostMeter::with_unknown_model_sink(Arc<dyn UnknownModelSink>)` and fires on every attempt regardless of the policy's log-dedup state, so production dashboards see raw catalog-drift counts. `replace_pricing(table)` (full-table swap), `replace_model_pricing(model, pricing)` (single-row insert-or-replace), and `pricing_snapshot()` (owned clone for admin diff) cover the hot-reload paths; every mutation serialises through the same `RwLock<PricingTable>` so concurrent admin writers cannot tear the table.
- **`QuotaLimiter`** — composite gate: rate (RPS) + budget ceiling (per-tenant cumulative spend cap). Runs *before* the request.
- **`TenantPolicy`** — per-tenant aggregate of optional handles. Fluent construction: `TenantPolicy::new().with_redactor(r).with_quota(q).with_cost_meter(m)` — every primitive is `Option<Arc<...>>`, absence means "disabled" (pass-through).
- **`PolicyRegistry`** — `DashMap<TenantId, Arc<TenantPolicy>>` + `Arc<RwLock<Arc<TenantPolicy>>>` fallback. `register(tenant, policy)` / `replace_fallback(policy)` for whole-policy swaps; `mutate_fallback(|p| -> TenantPolicy)` / `mutate_tenant(tid, |p| -> TenantPolicy)` for atomic partial updates — the closure receives the current policy by reference and returns the next one, the registry installs it in a single shard-locked step so concurrent mutations on the same tenant cannot lose updates. Closures run under the slot's write lock; perform any I/O *before* calling.
- **`PolicyLayer`** — `tower::Layer` (`NAME = "policy"`, impls `NamedLayer`) that wraps both `Service<ModelInvocation>` (request scrub + response scrub + cost) AND `Service<ToolInvocation>` (input scrub + output scrub). One layer instance, two service shapes.

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

