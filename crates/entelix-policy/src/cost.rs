//! `CostMeter` + [`PricingTable`] / [`ModelPricing`] — `rust_decimal`-
//! backed transactional charge accumulator. F4 mitigation: a charge
//! is recorded **only after** the response decoder succeeds — there is
//! no API path that lets an in-flight failure produce a partial
//! charge.
//!
//! Pricing is per-model, per-1000-tokens. Vendors publish
//! cents-per-1k figures; using `rust_decimal::Decimal` keeps the
//! per-call cost an exact rational with no float-rounding drift
//! across millions of charges.

// Read-lock guards on `pricing` are scoped inside non-async blocks and
// dropped before the ledger update / tracing call. clippy's
// `significant_drop_tightening` flags the binding pattern even when
// the block scope already drops correctly.
#![allow(clippy::significant_drop_tightening)]

use std::collections::HashMap;
use std::sync::Arc;

use dashmap::DashMap;
use parking_lot::RwLock;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use entelix_core::ir::Usage;

use crate::error::{PolicyError, PolicyResult};

/// Per-model pricing, in cost units per 1000 tokens. The unit is
/// caller-defined (USD cents, GBP pence, internal credits) — the
/// meter is unit-blind and just sums `Decimal`s.
///
/// Every rate is mandatory (invariant #15 — no silent fallback).
/// Vendors that don't charge for a tier (e.g. Bedrock has no cache
/// surface today) pass [`Decimal::ZERO`] explicitly so the operator
/// declares their pricing posture rather than inheriting whatever
/// fallback the SDK happens to ship.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelPricing {
    /// Cost per 1000 prompt tokens.
    pub input_per_1k: Decimal,
    /// Cost per 1000 completion tokens.
    pub output_per_1k: Decimal,
    /// Cost per 1000 cache-read tokens. Vendor-published rate —
    /// e.g. Anthropic Sonnet 4.6 = 10% of input, `OpenAI` gpt-4o =
    /// 50% of input, Bedrock = 0 today.
    pub cache_read_per_1k: Decimal,
    /// Cost per 1000 cache-write tokens. Anthropic charges a
    /// premium (~25% above input) for cache creation; many
    /// vendors charge zero.
    pub cache_write_per_1k: Decimal,
}

impl ModelPricing {
    /// Build a pricing row. All four rates are required — the SDK
    /// never invents a cache rate from the input rate.
    #[must_use]
    pub const fn new(
        input_per_1k: Decimal,
        output_per_1k: Decimal,
        cache_read_per_1k: Decimal,
        cache_write_per_1k: Decimal,
    ) -> Self {
        Self {
            input_per_1k,
            output_per_1k,
            cache_read_per_1k,
            cache_write_per_1k,
        }
    }

    /// Compute the exact cost for one [`Usage`] sample. All
    /// arithmetic is integer-on-`Decimal`; no floats.
    #[must_use]
    pub fn cost_for(&self, usage: &Usage) -> Decimal {
        let input = self.input_per_1k * Decimal::from(usage.input_tokens) / Decimal::from(1000);
        let output = self.output_per_1k * Decimal::from(usage.output_tokens) / Decimal::from(1000);
        let cache_write = self.cache_write_per_1k
            * Decimal::from(usage.cache_creation_input_tokens)
            / Decimal::from(1000);
        let cache_read =
            self.cache_read_per_1k * Decimal::from(usage.cached_input_tokens) / Decimal::from(1000);
        input + output + cache_write + cache_read
    }
}

/// Lookup of model name → [`ModelPricing`]. Keys are the same model
/// strings the codecs send to the wire (e.g. `"claude-opus-4-7"`,
/// `"gpt-4.1"`). Lookup is exact; aliases are the caller's
/// responsibility.
#[derive(Clone, Debug, Default)]
pub struct PricingTable {
    by_model: HashMap<String, ModelPricing>,
}

impl PricingTable {
    /// Empty table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert (or overwrite) one model's pricing.
    pub fn set(&mut self, model: impl Into<String>, pricing: ModelPricing) {
        self.by_model.insert(model.into(), pricing);
    }

    /// Builder-style insert.
    #[must_use]
    pub fn add_model_pricing(mut self, model: impl Into<String>, pricing: ModelPricing) -> Self {
        self.set(model, pricing);
        self
    }

    /// Look up a model's pricing.
    #[must_use]
    pub fn get(&self, model: &str) -> Option<&ModelPricing> {
        self.by_model.get(model)
    }

    /// Number of configured models.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_model.len()
    }

    /// True when the table has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_model.is_empty()
    }
}

/// Behavior when [`CostMeter::charge`] is called with a `model` that
/// has no entry in the [`PricingTable`].
///
/// Default is [`Reject`] — the safe choice for production billing
/// where a missing row is a configuration bug. [`WarnOnce`] is a
/// gentler option for staging environments and incremental vendor
/// rollouts where a new model name reaches traffic before the
/// pricing table catches up.
///
/// [`Reject`]: UnknownModelPolicy::Reject
/// [`WarnOnce`]: UnknownModelPolicy::WarnOnce
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum UnknownModelPolicy {
    /// Return [`PolicyError::UnknownModel`]. The caller must decide
    /// whether to fail the request or swallow it. Default.
    #[default]
    Reject,
    /// Log a `tracing::warn` at most once per distinct model name
    /// and record a zero charge. The model name set is held in the
    /// meter so a single missing entry does not flood telemetry.
    WarnOnce,
}

/// Cap on distinct model names tracked under `WarnOnce`.
///
/// Bounds `warned_models` at roughly `MAX_WARNED_MODELS *
/// avg_name_len` bytes — a couple of KiB at this limit. Past the
/// cap, additional distinct unknowns produce a single saturated
/// warn and stop accumulating, so an adversarial caller spamming
/// distinct names cannot drive the process out of memory.
pub const MAX_WARNED_MODELS: usize = 1024;

/// Default cap on distinct tenant ledger entries.
///
/// Same defense-in-depth shape as [`MAX_WARNED_MODELS`]: an
/// adversarial caller submitting requests with attacker-chosen
/// `tenant_id` strings could otherwise grow the in-memory ledger
/// without bound, exhausting process memory. With this cap, once
/// the ledger has recorded `DEFAULT_MAX_TENANTS` distinct tenants
/// the meter logs a single saturation warn and silently records
/// `Decimal::ZERO` for further unknown-tenant charges. Operators
/// override the cap via [`CostMeter::with_max_tenants`] (deployments
/// with truly large tenant counts size up; deployments draining
/// idle tenants on a schedule keep the default).
///
/// `10_000` is a pragmatic ceiling — a single [`CostMeter`] holds
/// a `String` + `Decimal` per tenant (~64 bytes amortised), so the
/// cap bounds the ledger at roughly 640 KiB.
pub const DEFAULT_MAX_TENANTS: usize = 10_000;

/// Per-tenant cost ledger. Records the cumulative spend for every
/// tenant that has ever been charged.
///
/// Cloning is cheap (`Arc` over the underlying maps) — share one
/// meter across the whole process.
#[derive(Clone)]
pub struct CostMeter {
    pricing: Arc<RwLock<PricingTable>>,
    ledger: Arc<DashMap<entelix_core::TenantId, Decimal>>,
    unknown_policy: UnknownModelPolicy,
    /// Bounded set of model names already warned about under
    /// [`UnknownModelPolicy::WarnOnce`]. Capped at
    /// [`MAX_WARNED_MODELS`] entries to bound memory under
    /// adversarial-input spam.
    warned_models: Arc<DashMap<String, ()>>,
    /// `true` once `warned_models` reached
    /// [`MAX_WARNED_MODELS`] and the saturation warn has been
    /// emitted. Subsequent unknown-model calls return zero charge
    /// silently.
    warned_saturated: Arc<std::sync::atomic::AtomicBool>,
    /// Maximum distinct tenant ledger entries before
    /// [`Self::charge`] starts dropping new tenants on the floor.
    /// See [`DEFAULT_MAX_TENANTS`] for the rationale.
    max_tenants: usize,
    /// `true` once `ledger` reached `max_tenants` and the
    /// saturation warn has been emitted. Subsequent unknown-tenant
    /// calls return `Ok(Decimal::ZERO)` silently.
    tenants_saturated: Arc<std::sync::atomic::AtomicBool>,
}

impl CostMeter {
    /// Build with the supplied pricing table, the default
    /// `UnknownModelPolicy::Reject`, and [`DEFAULT_MAX_TENANTS`].
    #[must_use]
    pub fn new(pricing: PricingTable) -> Self {
        Self {
            pricing: Arc::new(RwLock::new(pricing)),
            ledger: Arc::new(DashMap::new()),
            unknown_policy: UnknownModelPolicy::default(),
            warned_models: Arc::new(DashMap::new()),
            warned_saturated: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            max_tenants: DEFAULT_MAX_TENANTS,
            tenants_saturated: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Builder-style override of the unknown-model policy.
    #[must_use]
    pub const fn with_unknown_model_policy(mut self, policy: UnknownModelPolicy) -> Self {
        self.unknown_policy = policy;
        self
    }

    /// Override the maximum distinct tenant entries the ledger
    /// retains. Past this cap, [`Self::charge`] records
    /// `Decimal::ZERO` for new tenants and emits a single saturation
    /// warn. Operators draining idle tenants on a schedule
    /// (`drain(tenant)` on a periodic job) should leave the
    /// default; deployments with truly large tenant counts size
    /// up. Setting `0` disables charging entirely (every call
    /// returns zero) which is mostly useful for tests.
    #[must_use]
    pub const fn with_max_tenants(mut self, cap: usize) -> Self {
        self.max_tenants = cap;
        self
    }

    /// Effective tenant cap.
    #[must_use]
    pub const fn max_tenants(&self) -> usize {
        self.max_tenants
    }

    /// Number of tenants currently in the ledger.
    #[must_use]
    pub fn tracked_tenant_count(&self) -> usize {
        self.ledger.len()
    }

    /// Hot-swap the pricing table. Used by operators rolling out
    /// new vendor rates without a process restart. The `&self`
    /// receiver is intentional — every clone of the `Arc<CostMeter>`
    /// shares the same pricing slot, so a config-reload thread can
    /// replace rates without coordinating with charge sites.
    pub fn replace_pricing(&self, pricing: PricingTable) {
        *self.pricing.write() = pricing;
    }

    /// Internal: emit a one-shot saturation warn and flip the
    /// `tenants_saturated` flag. Race-tolerant via
    /// `compare_exchange` on the flag — only the first thread
    /// past the cap logs.
    fn warn_tenants_saturated(&self) {
        use std::sync::atomic::Ordering;
        if self
            .tenants_saturated
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
            .is_ok()
        {
            tracing::warn!(
                target: "entelix_policy::cost",
                cap = self.max_tenants,
                "cost meter tenant ledger cap reached — further unknown tenants charged as zero"
            );
        }
    }

    /// Internal: log a `tracing::warn` at most once per distinct
    /// `model` name and bound the warned-set at
    /// [`MAX_WARNED_MODELS`] — past that, emit one saturation warn
    /// and stop accumulating so an adversarial caller cannot drive
    /// memory unbounded with distinct unknown names.
    fn warn_once_for_unknown(&self, model: &str) {
        use std::sync::atomic::Ordering;

        // Fast path: already saturated — silent zero charge.
        if self.warned_saturated.load(Ordering::Relaxed) {
            return;
        }
        // De-dupe gate. DashMap::insert returns the previous value;
        // `None` means this is a fresh model.
        if self.warned_models.contains_key(model) {
            return;
        }
        // Try to claim a slot. Race-tolerant: even if multiple
        // threads pass the contains_key check, the size check after
        // insert handles it.
        if self.warned_models.len() >= MAX_WARNED_MODELS {
            // Saturate exactly once.
            if !self.warned_saturated.swap(true, Ordering::SeqCst) {
                tracing::warn!(
                    target: "entelix_policy::cost",
                    cap = MAX_WARNED_MODELS,
                    "cost meter warned_models cap reached — further unknown models suppressed"
                );
            }
            return;
        }
        if self.warned_models.insert(model.to_owned(), ()).is_none() {
            tracing::warn!(
                target: "entelix_policy::cost",
                model,
                "cost meter has no pricing row for model — recording zero charge"
            );
        }
    }

    /// Record a charge for `tenant` against `model` for `usage`.
    /// Returns the exact charge amount.
    ///
    /// When `model` has no row in the pricing table the behavior
    /// follows [`Self::with_unknown_model_policy`] — by default a
    /// [`PolicyError::UnknownModel`] is returned; under
    /// [`UnknownModelPolicy::WarnOnce`] the meter logs a single
    /// `tracing::warn` per distinct model and returns
    /// `Decimal::ZERO`.
    ///
    /// **Transactional (F4)**: this method is invoked from the
    /// `post_response` hook, which only runs after the codec has
    /// successfully decoded the response. A network failure / parse
    /// error short-circuits before this point and the ledger stays
    /// untouched.
    pub fn charge(
        &self,
        tenant_id: &entelix_core::TenantId,
        model: &str,
        usage: &Usage,
    ) -> PolicyResult<Decimal> {
        let cost = {
            let pricing = self.pricing.read();
            match pricing.get(model) {
                Some(model_pricing) => model_pricing.cost_for(usage),
                None => match self.unknown_policy {
                    UnknownModelPolicy::Reject => {
                        return Err(PolicyError::UnknownModel(model.to_owned()));
                    }
                    UnknownModelPolicy::WarnOnce => {
                        self.warn_once_for_unknown(model);
                        return Ok(Decimal::ZERO);
                    }
                },
            }
        };
        if cost.is_zero() {
            return Ok(cost);
        }
        // Saturation check: only NEW tenants count against the cap;
        // already-tracked tenants accumulate into their existing
        // entry without growing the map. This keeps the cap a
        // memory-bound, not a charging-rate bound. `TenantId`
        // implements `Borrow<str>`, so the lookup uses the existing
        // `Arc<str>` without an extra allocation.
        let already_tracked = self.ledger.contains_key(tenant_id.as_str());
        if !already_tracked && self.ledger.len() >= self.max_tenants {
            self.warn_tenants_saturated();
            return Ok(Decimal::ZERO);
        }
        self.ledger
            .entry(tenant_id.clone())
            .and_modify(|v| *v += cost)
            .or_insert(cost);
        tracing::debug!(
            target: "entelix_policy::cost",
            tenant_id = tenant_id.as_str(),
            model,
            charge = %cost,
            "cost meter charged"
        );
        Ok(cost)
    }

    /// Cumulative spend for `tenant_id`. Returns `Decimal::ZERO` for
    /// an unseen tenant.
    #[must_use]
    pub fn spent_by(&self, tenant_id: &entelix_core::TenantId) -> Decimal {
        self.ledger
            .get(tenant_id.as_str())
            .map_or(Decimal::ZERO, |v| *v)
    }

    /// Reset (and return) the recorded spend for `tenant_id`. Used by
    /// nightly billing to drain the in-memory ledger after
    /// persisting it.
    pub fn drain(&self, tenant_id: &entelix_core::TenantId) -> Decimal {
        self.ledger
            .remove(tenant_id.as_str())
            .map_or(Decimal::ZERO, |(_, v)| v)
    }
}

#[async_trait::async_trait]
impl entelix_core::CostCalculator for CostMeter {
    /// Side-effect-free cost computation for telemetry. Looks up
    /// the pricing row for `model` and returns the computed
    /// per-call cost as `f64` for emission into observability
    /// fields like `gen_ai.usage.cost`.
    ///
    /// `ctx` is accepted for the trait contract — `CostMeter` uses
    /// a global pricing table shared across tenants. Multi-tenant
    /// calculators that need per-tenant pricing tiers wrap a
    /// `CostMeter` per tenant or implement `CostCalculator`
    /// directly with a `(tenant_id, model) → ModelPricing` lookup.
    ///
    /// Returns `None` when the model is not in the pricing table —
    /// telemetry consumers omit the cost attribute rather than
    /// emitting a misleading zero. The calculator path does NOT
    /// mutate the per-tenant ledger; ledger updates flow through
    /// [`Self::charge`] which is invoked by the `PolicyLayer`
    /// service after a successful response.
    async fn compute_cost(
        &self,
        model: &str,
        usage: &Usage,
        _ctx: &entelix_core::ExecutionContext,
    ) -> Option<f64> {
        use rust_decimal::prelude::ToPrimitive;
        let pricing = self.pricing.read();
        let model_pricing = pricing.get(model)?;
        // `Decimal::to_f64` is None only on overflow — at production
        // pricing rates the per-call cost stays well within f64 range.
        model_pricing.cost_for(usage).to_f64()
    }
}

/// Conservative worst-case output budget used by the pre-call
/// estimator when [`entelix_core::ir::ModelRequest::max_tokens`] is
/// unset. Vendor defaults vary (Anthropic = `max_tokens` required by
/// API contract; `OpenAI` = vendor-default ~4096; Gemini = up to
/// 8192). The constant biases toward overestimation so a `RunBudget`
/// pre-call gate fails closed (false-positive rejection is
/// recoverable; silent overrun is not).
const PRE_CALL_UNBOUNDED_OUTPUT_TOKENS: u32 = 8_192;

#[async_trait::async_trait]
impl entelix_core::BudgetCostEstimator for CostMeter {
    /// Pre-call worst-case estimate in `Decimal` precision. Looks up
    /// the pricing row for `request.model`; if absent, returns
    /// `None` so the pre-call gate skips rather than synthesising a
    /// zero (matches `compute_cost`).
    ///
    /// Prompt-token estimation uses [`entelix_core::ByteCountTokenCounter`]
    /// for a conservative count without coupling to a vendor-accurate
    /// tokenizer. Operators with vendor-accurate token counters wired
    /// via [`entelix_core::TokenCounterRegistry`] implement a custom
    /// [`entelix_core::BudgetCostEstimator`] that consults the
    /// registry directly — the trait surface stays vendor-agnostic.
    ///
    /// Output-token estimate is `request.max_tokens` when set, or
    /// `PRE_CALL_UNBOUNDED_OUTPUT_TOKENS` as the worst-case bound.
    /// Cache rates are treated as zero (no cache hit on a yet-to-fire
    /// call), which biases the estimate upward.
    async fn estimate_pre_call(
        &self,
        request: &entelix_core::ir::ModelRequest,
        _ctx: &entelix_core::ExecutionContext,
    ) -> Option<Decimal> {
        use entelix_core::TokenCounter;
        let pricing = self.pricing.read();
        let model_pricing = pricing.get(&request.model)?;
        let counter = entelix_core::ByteCountTokenCounter::new();
        let raw_tokens = counter.count_messages(&request.messages);
        let input_tokens = u32::try_from(raw_tokens).unwrap_or(u32::MAX); // silent-fallback-ok: saturate at u32::MAX so a pathologically long prompt over-estimates rather than wraps; biases the pre-call gate conservatively.
        let output_tokens = request
            .max_tokens
            .unwrap_or(PRE_CALL_UNBOUNDED_OUTPUT_TOKENS); // silent-fallback-ok: PRE_CALL_UNBOUNDED_OUTPUT_TOKENS is the documented worst-case bound for vendors that allow unset max_tokens.
        let projected = Usage::new(input_tokens, output_tokens);
        Some(model_pricing.cost_for(&projected))
    }

    /// Post-call actual charge in `Decimal` precision. Read
    /// directly from the response's [`Usage`]; this is the same
    /// arithmetic [`Self::charge`] feeds into the per-tenant ledger,
    /// surfaced separately so [`entelix_core::RunBudget::observe_cost`]
    /// receives the precision-preserving value before any
    /// `f64`-lossy telemetry conversion.
    async fn calculate_actual(
        &self,
        request: &entelix_core::ir::ModelRequest,
        usage: &Usage,
        _ctx: &entelix_core::ExecutionContext,
    ) -> Option<Decimal> {
        let pricing = self.pricing.read();
        let model_pricing = pricing.get(&request.model)?;
        Some(model_pricing.cost_for(usage))
    }
}

impl std::fmt::Debug for CostMeter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CostMeter")
            .field("models", &self.pricing.read().len())
            .field("tenants", &self.ledger.len())
            .field("unknown_policy", &self.unknown_policy)
            .field("warned_models", &self.warned_models.len())
            .field(
                "warned_saturated",
                &self
                    .warned_saturated
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
            .field("max_tenants", &self.max_tenants)
            .field(
                "tenants_saturated",
                &self
                    .tenants_saturated
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
            .finish()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use entelix_core::TenantId;
    use std::str::FromStr;

    use super::*;

    fn d(s: &str) -> Decimal {
        Decimal::from_str(s).unwrap()
    }

    fn pricing() -> PricingTable {
        PricingTable::new()
            .add_model_pricing(
                "claude-opus-4-7",
                ModelPricing::new(d("15"), d("75"), d("1.5"), d("18.75")),
            )
            .add_model_pricing(
                "gpt-4.1",
                // gpt-4.1 cache-read is 25% of input (vendor-published).
                ModelPricing::new(d("2"), d("8"), d("0.5"), Decimal::ZERO), // magic-ok: test fixture rate
            )
    }

    fn usage(input: u32, output: u32) -> Usage {
        Usage::new(input, output)
    }

    #[test]
    fn cost_for_simple_usage_is_exact() {
        let p = pricing();
        let claude = p.get("claude-opus-4-7").unwrap();
        let cost = claude.cost_for(&usage(1000, 1000));
        // 1000 input * 15/1000 + 1000 output * 75/1000 = 15 + 75 = 90
        assert_eq!(cost, d("90"));
    }

    #[test]
    fn cost_with_cache_writes_and_reads() {
        let p = pricing();
        let claude = p.get("claude-opus-4-7").unwrap();
        let cost = claude.cost_for(
            &Usage::new(500, 200)
                .with_cached_input_tokens(2000)
                .with_cache_creation_input_tokens(800),
        );
        // 500*15/1000 + 200*75/1000 + 800*18.75/1000 + 2000*1.5/1000
        // = 7.5 + 15 + 15 + 3 = 40.5
        assert_eq!(cost, d("40.5"));
    }

    #[test]
    fn cache_read_uses_explicit_rate_no_fallback() {
        // gpt-4.1 has cache_read_per_1k = 0.5 (25% of input). The SDK
        // does not invent a fallback from input_per_1k — the rate is
        // exactly what the operator declared (invariant #15).
        let p = pricing();
        let gpt = p.get("gpt-4.1").unwrap();
        let cost = gpt.cost_for(&Usage::default().with_cached_input_tokens(1000));
        // 1000 cache_read * 0.5 / 1000 = 0.5
        assert_eq!(cost, d("0.5")); // magic-ok: arithmetic check value
    }

    #[test]
    fn cache_write_zero_rate_means_zero_charge() {
        // gpt-4.1 has cache_write_per_1k = ZERO (vendor doesn't
        // charge for cache writes). Cache-write tokens accrue no
        // cost — a regression-test for the "no silent fallback"
        // contract: the SDK does not invent a positive rate from
        // input_per_1k.
        let p = pricing();
        let gpt = p.get("gpt-4.1").unwrap();
        let cost = gpt.cost_for(&Usage::default().with_cache_creation_input_tokens(1_000_000));
        assert_eq!(cost, Decimal::ZERO);
    }

    #[test]
    fn charge_sums_per_tenant_atomically() {
        let meter = CostMeter::new(pricing());
        let u = usage(1000, 1000);
        meter
            .charge(&TenantId::new("alpha"), "claude-opus-4-7", &u)
            .unwrap();
        meter
            .charge(&TenantId::new("alpha"), "claude-opus-4-7", &u)
            .unwrap();
        meter
            .charge(&TenantId::new("bravo"), "claude-opus-4-7", &u)
            .unwrap();
        assert_eq!(meter.spent_by(&TenantId::new("alpha")), d("180"));
        assert_eq!(meter.spent_by(&TenantId::new("bravo")), d("90"));
        assert_eq!(meter.spent_by(&TenantId::new("never-seen")), Decimal::ZERO);
    }

    #[test]
    fn unknown_model_does_not_charge() {
        let meter = CostMeter::new(pricing());
        let err = meter
            .charge(&TenantId::new("alpha"), "unknown-model", &usage(1000, 1000))
            .unwrap_err();
        assert!(matches!(err, PolicyError::UnknownModel(_)));
        assert_eq!(meter.spent_by(&TenantId::new("alpha")), Decimal::ZERO);
    }

    #[test]
    fn zero_usage_is_a_zero_charge_no_ledger_entry() {
        let meter = CostMeter::new(pricing());
        let cost = meter
            .charge(
                &TenantId::new("alpha"),
                "claude-opus-4-7",
                &Usage::default(),
            )
            .unwrap();
        assert_eq!(cost, Decimal::ZERO);
        assert_eq!(meter.spent_by(&TenantId::new("alpha")), Decimal::ZERO);
    }

    #[test]
    fn drain_resets_tenant_ledger() {
        let meter = CostMeter::new(pricing());
        meter
            .charge(
                &TenantId::new("alpha"),
                "claude-opus-4-7",
                &usage(1000, 1000),
            )
            .unwrap();
        assert_eq!(meter.drain(&TenantId::new("alpha")), d("90"));
        assert_eq!(meter.spent_by(&TenantId::new("alpha")), Decimal::ZERO);
    }

    #[test]
    fn warn_once_unknown_model_returns_zero_and_does_not_charge() {
        let meter =
            CostMeter::new(pricing()).with_unknown_model_policy(UnknownModelPolicy::WarnOnce);
        let cost = meter
            .charge(
                &TenantId::new("alpha"),
                "vendor-preview-x",
                &usage(1000, 1000),
            )
            .unwrap();
        assert_eq!(cost, Decimal::ZERO);
        assert_eq!(meter.spent_by(&TenantId::new("alpha")), Decimal::ZERO);
        // Same model again — must not re-warn (state inspected via len).
        meter
            .charge(
                &TenantId::new("alpha"),
                "vendor-preview-x",
                &usage(2000, 2000),
            )
            .unwrap();
        assert_eq!(meter.warned_models.len(), 1);
        // Distinct unknown model — separate warn entry.
        meter
            .charge(&TenantId::new("alpha"), "vendor-preview-y", &usage(1000, 0))
            .unwrap();
        assert_eq!(meter.warned_models.len(), 2);
    }

    #[test]
    fn ledger_caps_at_max_tenants_under_adversarial_spam() {
        // Tiny cap so the test runs fast. Real deployments use
        // DEFAULT_MAX_TENANTS (10000) or override via with_max_tenants.
        let meter = CostMeter::new(pricing()).with_max_tenants(8);
        // First 8 distinct tenants land in the ledger and accumulate.
        for i in 0..8 {
            let charge = meter
                .charge(
                    &TenantId::new(format!("tenant-{i}")),
                    "claude-opus-4-7",
                    &usage(100, 100),
                )
                .unwrap();
            assert!(!charge.is_zero(), "tenant {i} should be charged");
        }
        assert_eq!(meter.tracked_tenant_count(), 8);
        // Past the cap: NEW tenants record Decimal::ZERO and never
        // join the ledger — saturation flag fires once.
        for i in 8..200 {
            let charge = meter
                .charge(
                    &TenantId::new(format!("tenant-{i}")),
                    "claude-opus-4-7",
                    &usage(100, 100),
                )
                .unwrap();
            assert_eq!(
                charge,
                Decimal::ZERO,
                "tenant {i} past cap should be charged zero (silently dropped)"
            );
        }
        assert_eq!(
            meter.tracked_tenant_count(),
            8,
            "ledger size must not grow past max_tenants"
        );
        // Already-tracked tenants continue to accumulate normally —
        // the cap is on distinct entries, not on charging rate.
        let prior = meter.spent_by(&TenantId::new("tenant-0"));
        let _ = meter
            .charge(
                &TenantId::new("tenant-0"),
                "claude-opus-4-7",
                &usage(100, 100),
            )
            .unwrap();
        assert!(meter.spent_by(&TenantId::new("tenant-0")) > prior);
    }

    #[test]
    fn warned_models_caps_at_max_under_adversarial_spam() {
        let meter =
            CostMeter::new(pricing()).with_unknown_model_policy(UnknownModelPolicy::WarnOnce);
        // Spam well past the cap with distinct names.
        for i in 0..(MAX_WARNED_MODELS * 2) {
            let _ = meter.charge(&TenantId::new("alpha"), &format!("model-{i}"), &usage(1, 1));
        }
        assert!(
            meter.warned_models.len() <= MAX_WARNED_MODELS,
            "warned_models exceeded cap: {} > {MAX_WARNED_MODELS}",
            meter.warned_models.len()
        );
        // Ledger remains untouched (zero charges).
        assert_eq!(meter.spent_by(&TenantId::new("alpha")), Decimal::ZERO);
    }

    #[test]
    fn known_model_still_charges_under_warn_once() {
        let meter =
            CostMeter::new(pricing()).with_unknown_model_policy(UnknownModelPolicy::WarnOnce);
        let cost = meter
            .charge(
                &TenantId::new("alpha"),
                "claude-opus-4-7",
                &usage(1000, 1000),
            )
            .unwrap();
        assert_eq!(cost, d("90"));
        assert_eq!(meter.spent_by(&TenantId::new("alpha")), d("90"));
        assert_eq!(meter.warned_models.len(), 0);
    }

    #[test]
    fn pricing_can_be_hot_swapped() {
        let meter = CostMeter::new(pricing());
        let mut new_pricing = pricing();
        new_pricing.set(
            "gpt-4.1",
            ModelPricing::new(d("20"), d("80"), Decimal::ZERO, Decimal::ZERO),
        );
        meter.replace_pricing(new_pricing);
        let cost = meter
            .charge(&TenantId::new("alpha"), "gpt-4.1", &usage(1000, 0))
            .unwrap();
        assert_eq!(cost, d("20"));
    }

    #[test]
    fn pricing_replacement_is_observed_by_cloned_meters() {
        let meter = CostMeter::new(pricing());
        let cloned = meter.clone();

        let mut new_pricing = pricing();
        new_pricing.set(
            "gpt-4.1",
            ModelPricing::new(d("20"), d("80"), Decimal::ZERO, Decimal::ZERO),
        );
        cloned.replace_pricing(new_pricing);

        let cost = meter
            .charge(&TenantId::new("alpha"), "gpt-4.1", &usage(1000, 0))
            .unwrap();
        assert_eq!(
            cost,
            d("20"),
            "the original meter must charge against a pricing table installed via a clone"
        );
    }
}
