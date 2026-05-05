//! `QuotaLimiter` — composite gate that runs *before* a request is
//! sent. Combines:
//!
//! - a [`RateLimiter`] check (RPS-style refusal), and
//! - a [`Budget`] check (cumulative spend cap per tenant).
//!
//! Failing either check returns the appropriate
//! [`PolicyError`] which the hook layer translates to
//! `Error::Provider { status: 429, ... }` (rate) or
//! `Error::Provider { status: 402, ... }` (budget).
//!
//! `QuotaLimiter` does **not** charge — that is `CostMeter`'s job at
//! `post_response`. The pre-flight check is *advisory*: a tenant
//! sitting at 99% of budget can issue one more request even if it
//! ends up overshooting marginally. Hard caps live above this layer
//! (e.g. payment-system pre-authorization).

use std::sync::Arc;

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::cost::CostMeter;
use crate::error::{PolicyError, PolicyResult};
use crate::rate_limit::RateLimiter;

/// Per-tenant cumulative spend ceiling. Compared against
/// `CostMeter::spent_by(tenant)` at pre-request time.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Budget {
    /// Maximum cumulative spend (in the same unit `CostMeter` uses).
    /// `None` means "no ceiling" — the meter still records charges
    /// for visibility but the gate never refuses.
    pub ceiling: Option<Decimal>,
}

impl Budget {
    /// Build with a hard ceiling.
    #[must_use]
    pub const fn capped(ceiling: Decimal) -> Self {
        Self {
            ceiling: Some(ceiling),
        }
    }

    /// Build with no ceiling — recording-only.
    #[must_use]
    pub const fn unlimited() -> Self {
        Self { ceiling: None }
    }
}

/// Composite quota gate.
#[derive(Clone)]
pub struct QuotaLimiter {
    rate_limiter: Option<Arc<dyn RateLimiter>>,
    cost_meter: Option<Arc<CostMeter>>,
    budget: Budget,
}

impl QuotaLimiter {
    /// Build with the rate limiter, cost meter, and budget all
    /// optional. A `QuotaLimiter` with all three `None` is a no-op
    /// — the manager treats absence the same way.
    #[must_use]
    pub fn new(
        rate_limiter: Option<Arc<dyn RateLimiter>>,
        cost_meter: Option<Arc<CostMeter>>,
        budget: Budget,
    ) -> Self {
        Self {
            rate_limiter,
            cost_meter,
            budget,
        }
    }

    /// Pre-request gate. Returns `Ok(())` to admit, or the
    /// appropriate refusal error. `tokens` is the rate-limit cost
    /// of this request (typically 1 for "one request"); custom
    /// counters can be larger if the caller wants to spend extra
    /// against the bucket for expensive endpoints.
    pub async fn check_pre_request(&self, tenant: &str, tokens: u32) -> PolicyResult<()> {
        // Budget gate first — refusing on budget is preferable to
        // spending a rate-limit token only to bounce on the next
        // gate.
        if let (Some(ceiling), Some(meter)) = (self.budget.ceiling, self.cost_meter.as_ref()) {
            let spent = meter.spent_by(tenant);
            if spent >= ceiling {
                return Err(PolicyError::BudgetExhausted {
                    tenant: tenant.to_owned(),
                    spent,
                    ceiling,
                });
            }
        }
        if let Some(rate) = &self.rate_limiter {
            rate.try_acquire(tenant, tokens).await?;
        }
        Ok(())
    }

    /// Borrow the configured budget.
    #[must_use]
    pub const fn budget(&self) -> &Budget {
        &self.budget
    }
}

impl std::fmt::Debug for QuotaLimiter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuotaLimiter")
            .field("rate_limiter", &self.rate_limiter.is_some())
            .field("cost_meter", &self.cost_meter.is_some())
            .field("budget", &self.budget)
            .finish()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::str::FromStr;
    use std::sync::Arc;

    use entelix_core::ir::Usage;

    use super::*;
    use crate::cost::{ModelPricing, PricingTable};
    use crate::rate_limit::TokenBucketLimiter;

    fn d(s: &str) -> Decimal {
        Decimal::from_str(s).unwrap()
    }

    fn meter() -> Arc<CostMeter> {
        Arc::new(CostMeter::new(PricingTable::new().add_model_pricing(
            "x",
            ModelPricing::new(d("10"), d("10"), Decimal::ZERO, Decimal::ZERO),
        )))
    }

    #[tokio::test]
    async fn empty_quota_is_pass_through() {
        let q = QuotaLimiter::new(None, None, Budget::unlimited());
        q.check_pre_request("t", 1).await.unwrap();
    }

    #[tokio::test]
    async fn rate_limit_refusal_propagates() {
        let limiter: Arc<dyn RateLimiter> = Arc::new(TokenBucketLimiter::new(1, 1.0).unwrap());
        let q = QuotaLimiter::new(Some(limiter), None, Budget::unlimited());
        q.check_pre_request("t", 1).await.unwrap();
        let err = q.check_pre_request("t", 1).await.unwrap_err();
        assert!(matches!(err, PolicyError::RateLimited { .. }));
    }

    #[tokio::test]
    async fn budget_refusal_when_spent_meets_ceiling() {
        let cm = meter();
        // Push spend up to the ceiling.
        cm.charge("t", "x", &Usage::new(1000, 1000)).unwrap();
        // 1000*10/1000 + 1000*10/1000 = 20
        assert_eq!(cm.spent_by("t"), d("20"));
        let q = QuotaLimiter::new(None, Some(cm), Budget::capped(d("20")));
        let err = q.check_pre_request("t", 1).await.unwrap_err();
        match err {
            PolicyError::BudgetExhausted { spent, ceiling, .. } => {
                assert_eq!(spent, d("20"));
                assert_eq!(ceiling, d("20"));
            }
            other => panic!("expected BudgetExhausted, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn budget_admit_below_ceiling() {
        let cm = meter();
        let q = QuotaLimiter::new(None, Some(cm), Budget::capped(d("100")));
        q.check_pre_request("t", 1).await.unwrap();
    }

    #[tokio::test]
    async fn budget_check_runs_before_rate_check() {
        let cm = meter();
        cm.charge("t", "x", &Usage::new(1000, 1000)).unwrap();
        let limiter: Arc<dyn RateLimiter> = Arc::new(TokenBucketLimiter::new(1, 1.0).unwrap());
        let q = QuotaLimiter::new(Some(limiter.clone()), Some(cm), Budget::capped(d("20")));
        // Budget exhausted; we should not consume a rate-limit token.
        let err = q.check_pre_request("t", 1).await.unwrap_err();
        assert!(matches!(err, PolicyError::BudgetExhausted { .. }));
        // Token is still available — verify by acquiring directly.
        limiter.try_acquire("t", 1).await.unwrap();
    }
}
