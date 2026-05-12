//! `TenantPolicy` — per-tenant aggregate of policy handles, plus
//! [`PolicyRegistry`] — the runtime registry that indexes them by
//! `tenant_id`.
//!
//! Each [`TenantPolicy`] field is `Option<Arc<...>>`. Absence means
//! "this primitive is disabled for this tenant" (pass-through). The
//! default policy handed out by `PolicyRegistry::policy_for` for an
//! unconfigured tenant is the all-`None` policy — i.e. the
//! single-tenant operator who never registers any policy still gets
//! a working SDK with zero policy enforcement.

use std::sync::Arc;

use dashmap::DashMap;
use parking_lot::RwLock;

use entelix_core::TenantId;

use crate::cost::CostMeter;
use crate::pii::PiiRedactor;
use crate::quota::QuotaLimiter;

/// Per-tenant aggregate of policy handles.
///
/// Every field is `Option<Arc<...>>`; absence means "this primitive is
/// disabled for this tenant" (pass-through). Cloning is cheap — every
/// primitive lives behind `Arc`, so handing a `TenantPolicy` to a
/// closure (e.g. [`PolicyRegistry::mutate_fallback`]) bumps refcounts
/// rather than deep-copying state.
///
/// Construct fluently:
///
/// ```ignore
/// let policy = TenantPolicy::new()
///     .with_redactor(redactor)
///     .with_cost_meter(meter);
/// ```
#[derive(Clone, Default)]
pub struct TenantPolicy {
    pub(crate) redactor: Option<Arc<dyn PiiRedactor>>,
    pub(crate) quota: Option<Arc<QuotaLimiter>>,
    pub(crate) cost_meter: Option<Arc<CostMeter>>,
}

impl std::fmt::Debug for TenantPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TenantPolicy")
            .field("redactor", &self.redactor.is_some())
            .field("quota", &self.quota.is_some())
            .field("cost_meter", &self.cost_meter.is_some())
            .finish()
    }
}

impl TenantPolicy {
    /// Empty policy — every primitive disabled (pass-through).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Install (or replace) the bidirectional [`PiiRedactor`]. The
    /// redactor runs on both `pre_request` and `post_response` —
    /// outbound-only redaction reintroduces F5.
    #[must_use]
    pub fn with_redactor(mut self, redactor: Arc<dyn PiiRedactor>) -> Self {
        self.redactor = Some(redactor);
        self
    }

    /// Install (or replace) the composite [`QuotaLimiter`] gate
    /// evaluated `pre_request` (rate-per-second + cumulative budget).
    #[must_use]
    pub fn with_quota(mut self, quota: Arc<QuotaLimiter>) -> Self {
        self.quota = Some(quota);
        self
    }

    /// Install (or replace) the [`CostMeter`]. Charges run only on
    /// the `Ok` branch of the response decoder — invariant 12 keeps
    /// failed calls out of the ledger.
    #[must_use]
    pub fn with_cost_meter(mut self, meter: Arc<CostMeter>) -> Self {
        self.cost_meter = Some(meter);
        self
    }

    /// True when no primitive is configured.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.redactor.is_none() && self.quota.is_none() && self.cost_meter.is_none()
    }

    /// Bidirectional PII redactor (F5). Runs on `pre_request` and
    /// `post_response`.
    #[must_use]
    pub fn redactor(&self) -> Option<&Arc<dyn PiiRedactor>> {
        self.redactor.as_ref()
    }

    /// Composite quota gate (rate + budget). Runs on `pre_request`.
    #[must_use]
    pub fn quota(&self) -> Option<&Arc<QuotaLimiter>> {
        self.quota.as_ref()
    }

    /// Cost meter (F4). Charged on `post_response`.
    #[must_use]
    pub fn cost_meter(&self) -> Option<&Arc<CostMeter>> {
        self.cost_meter.as_ref()
    }
}

/// Runtime registry mapping `tenant_id` → [`TenantPolicy`].
///
/// Cloning is cheap — internal state lives behind `Arc`s. Both
/// per-tenant entries and the fallback are interior-mutable, so
/// every clone of a registry observes (and can update) the same
/// policy state without coordinating exclusive ownership.
#[derive(Clone)]
pub struct PolicyRegistry {
    per_tenant: Arc<DashMap<TenantId, Arc<TenantPolicy>>>,
    fallback: Arc<RwLock<Arc<TenantPolicy>>>,
}

impl Default for PolicyRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PolicyRegistry {
    /// Empty manager; every lookup returns the empty fallback
    /// policy (pass-through).
    #[must_use]
    pub fn new() -> Self {
        Self {
            per_tenant: Arc::new(DashMap::new()),
            fallback: Arc::new(RwLock::new(Arc::new(TenantPolicy::default()))),
        }
    }

    /// Hot-swap the fallback policy used when a tenant has no
    /// explicit registration. Useful for "zero-configured tenants
    /// still get a basic redactor" deployments and for config-reload
    /// flows that update defaults without a process restart.
    pub fn replace_fallback(&self, fallback: TenantPolicy) {
        *self.fallback.write() = Arc::new(fallback);
    }

    /// Builder-style fallback installer.
    #[must_use]
    pub fn with_fallback(self, fallback: TenantPolicy) -> Self {
        self.replace_fallback(fallback);
        self
    }

    /// Register (or overwrite) a tenant's policy.
    pub fn register(&self, tenant_id: TenantId, policy: TenantPolicy) {
        self.per_tenant.insert(tenant_id, Arc::new(policy));
    }

    /// Builder-style register.
    #[must_use]
    pub fn with_tenant(self, tenant_id: TenantId, policy: TenantPolicy) -> Self {
        self.register(tenant_id, policy);
        self
    }

    /// Drop a tenant's registration. Subsequent lookups fall back to
    /// the fallback policy.
    pub fn unregister(&self, tenant_id: &TenantId) {
        self.per_tenant.remove(tenant_id);
    }

    /// Atomically derive a new fallback policy from the current one.
    /// `f` receives the current policy by reference and returns the
    /// next policy; the registry swaps the `Arc` slot in one step.
    ///
    /// Use this for **partial updates** — admin write paths that
    /// revise a single primitive (pricing reload, new redactor) keep
    /// the rest of the policy untouched without re-authoring the
    /// whole [`TenantPolicy`]:
    ///
    /// ```ignore
    /// registry.mutate_fallback(|p| p.clone().with_cost_meter(new_meter));
    /// ```
    ///
    /// **Lock contract**: `f` runs while the registry holds the
    /// fallback's write lock. Concurrent `replace_fallback` /
    /// `mutate_fallback` calls serialise behind it. Perform any I/O
    /// (database fetches, vendor catalogue pulls) **before** invoking
    /// — `f` must only assemble the new policy from precomputed
    /// parts. Calling back into the same registry from `f` is a
    /// deadlock bug.
    pub fn mutate_fallback<F>(&self, f: F)
    where
        F: FnOnce(&TenantPolicy) -> TenantPolicy,
    {
        let mut guard = self.fallback.write();
        let next = f(&guard);
        *guard = Arc::new(next);
    }

    /// Atomically derive a new policy for `tenant_id` from the
    /// current one (or the empty [`TenantPolicy::default`] when the
    /// tenant is not yet registered). `f` receives a borrow of the
    /// input policy and returns the next one; the registry installs
    /// it in a single shard-locked step so concurrent
    /// `mutate_tenant` calls on the same tenant cannot lose updates.
    ///
    /// **Lock contract**: same CPU-only rule as
    /// [`Self::mutate_fallback`] — `f` runs while the underlying
    /// `DashMap` shard lock is held, so I/O happens before the call
    /// and `f` only assembles the new value.
    pub fn mutate_tenant<F>(&self, tenant_id: &TenantId, f: F)
    where
        F: FnOnce(&TenantPolicy) -> TenantPolicy,
    {
        use dashmap::Entry;
        match self.per_tenant.entry(tenant_id.clone()) {
            Entry::Occupied(mut slot) => {
                let next = f(slot.get());
                slot.insert(Arc::new(next));
            }
            Entry::Vacant(slot) => {
                let next = f(&TenantPolicy::default());
                slot.insert(Arc::new(next));
            }
        }
    }

    /// Resolve a tenant's policy. Always returns *some* policy —
    /// the fallback when the tenant isn't explicitly registered.
    #[must_use]
    pub fn policy_for(&self, tenant_id: &TenantId) -> Arc<TenantPolicy> {
        self.per_tenant
            .get(tenant_id)
            .map_or_else(|| self.fallback.read().clone(), |entry| entry.clone())
    }

    /// Number of explicitly registered tenants. Excludes the
    /// fallback.
    #[must_use]
    pub fn tenant_count(&self) -> usize {
        self.per_tenant.len()
    }
}

impl std::fmt::Debug for PolicyRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let fallback = self.fallback.read().clone();
        f.debug_struct("PolicyRegistry")
            .field("tenants", &self.per_tenant.len())
            .field("has_fallback", &!fallback.is_empty())
            .finish()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::cost::{CostMeter, PricingTable};
    use crate::pii::RegexRedactor;

    #[test]
    fn empty_manager_returns_empty_fallback() {
        let mgr = PolicyRegistry::new();
        let p = mgr.policy_for(&TenantId::new("any-tenant"));
        assert!(p.is_empty());
        assert_eq!(mgr.tenant_count(), 0);
    }

    #[test]
    fn registered_tenant_overrides_fallback() {
        let mgr = PolicyRegistry::new();
        let policy = TenantPolicy::new().with_redactor(Arc::new(RegexRedactor::with_defaults()));
        mgr.register(TenantId::new("acme"), policy);
        let p = mgr.policy_for(&TenantId::new("acme"));
        assert!(p.redactor.is_some());
        // Unregistered still gets fallback.
        let other = mgr.policy_for(&TenantId::new("other"));
        assert!(other.is_empty());
    }

    #[test]
    fn fallback_can_be_replaced_with_a_real_policy() {
        let fb = TenantPolicy::new().with_cost_meter(Arc::new(CostMeter::new(PricingTable::new())));
        let mgr = PolicyRegistry::new().with_fallback(fb);
        let p = mgr.policy_for(&TenantId::new("never-registered"));
        assert!(p.cost_meter.is_some());
    }

    #[test]
    fn fallback_replacement_is_observed_by_cloned_registries() {
        let mgr = PolicyRegistry::new();
        let cloned = mgr.clone();

        let fb = TenantPolicy::new().with_cost_meter(Arc::new(CostMeter::new(PricingTable::new())));
        cloned.replace_fallback(fb);

        assert!(
            mgr.policy_for(&TenantId::new("never-registered"))
                .cost_meter
                .is_some(),
            "the original registry must observe a fallback installed via a clone"
        );
    }

    #[test]
    fn unregister_drops_tenant() {
        let mgr = PolicyRegistry::new();
        mgr.register(TenantId::new("acme"), TenantPolicy::default());
        assert_eq!(mgr.tenant_count(), 1);
        mgr.unregister(&TenantId::new("acme"));
        assert_eq!(mgr.tenant_count(), 0);
    }

    #[test]
    fn mutate_fallback_preserves_other_primitives_on_partial_swap() {
        // A pricing-reload admin path replaces just the cost meter;
        // the redactor and quota installed earlier must survive
        // unchanged. This is the canonical use case for the closure
        // form vs. the all-or-nothing `replace_fallback`.
        let mgr = PolicyRegistry::new();
        mgr.replace_fallback(
            TenantPolicy::new()
                .with_redactor(Arc::new(RegexRedactor::with_defaults()))
                .with_cost_meter(Arc::new(CostMeter::new(PricingTable::new()))),
        );
        let new_meter = Arc::new(CostMeter::new(PricingTable::new().add_model_pricing(
            "claude-opus-4-7",
            crate::cost::ModelPricing::new(
                rust_decimal::Decimal::ONE,
                rust_decimal::Decimal::ONE,
                rust_decimal::Decimal::ZERO,
                rust_decimal::Decimal::ZERO,
            ),
        )));
        let new_meter_for_assertion = new_meter.clone();
        mgr.mutate_fallback(|current| current.clone().with_cost_meter(new_meter));

        let fb = mgr.policy_for(&TenantId::new("never-registered"));
        assert!(fb.redactor.is_some(), "redactor must survive partial swap");
        let installed = fb.cost_meter.as_ref().unwrap();
        assert!(
            Arc::ptr_eq(installed, &new_meter_for_assertion),
            "cost meter must be the freshly installed one"
        );
    }

    #[test]
    fn mutate_tenant_on_registered_tenant_partial_swap() {
        let mgr = PolicyRegistry::new();
        mgr.register(
            TenantId::new("acme"),
            TenantPolicy::new().with_redactor(Arc::new(RegexRedactor::with_defaults())),
        );
        let new_meter = Arc::new(CostMeter::new(PricingTable::new()));
        mgr.mutate_tenant(&TenantId::new("acme"), |current| {
            current.clone().with_cost_meter(new_meter.clone())
        });

        let p = mgr.policy_for(&TenantId::new("acme"));
        assert!(p.redactor.is_some(), "redactor survives");
        assert!(p.cost_meter.is_some(), "cost meter installed");
    }

    #[test]
    fn mutate_tenant_on_unregistered_tenant_starts_from_default() {
        // First-time setup path: no prior registration, closure
        // receives the empty default and builds the initial policy.
        let mgr = PolicyRegistry::new();
        assert_eq!(mgr.tenant_count(), 0);
        let meter = Arc::new(CostMeter::new(PricingTable::new()));
        mgr.mutate_tenant(&TenantId::new("bravo"), |current| {
            assert!(current.is_empty(), "vacant tenant must see empty default");
            current.clone().with_cost_meter(meter.clone())
        });
        assert_eq!(mgr.tenant_count(), 1);
        assert!(mgr.policy_for(&TenantId::new("bravo")).cost_meter.is_some());
    }

    #[test]
    fn mutate_tenant_serialises_concurrent_updates_no_lost_writes() {
        // Two concurrent `mutate_tenant` calls install different
        // primitives on the same tenant. If the registry served them
        // through a non-atomic read-then-write, the second write
        // would overwrite the first's primitive. The DashMap entry
        // API serialises both calls behind the same shard lock so
        // BOTH primitives end up in the final policy.
        let mgr = Arc::new(PolicyRegistry::new());
        let meter = Arc::new(CostMeter::new(PricingTable::new()));
        let redactor: Arc<dyn PiiRedactor> = Arc::new(RegexRedactor::with_defaults());
        let barrier = Arc::new(std::sync::Barrier::new(2));

        // t1 owns `meter` (no shared reuse), t2 owns `redactor`. Both
        // need their own `mgr` + `barrier` clones; the original
        // `mgr` survives for the post-join assertion.
        let t1 = {
            let mgr = Arc::clone(&mgr);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                for _ in 0..1000 {
                    mgr.mutate_tenant(&TenantId::new("acme"), |current| {
                        current.clone().with_cost_meter(Arc::clone(&meter))
                    });
                }
            })
        };
        let t2 = {
            let mgr = Arc::clone(&mgr);
            std::thread::spawn(move || {
                barrier.wait();
                for _ in 0..1000 {
                    mgr.mutate_tenant(&TenantId::new("acme"), |current| {
                        current.clone().with_redactor(Arc::clone(&redactor))
                    });
                }
            })
        };
        t1.join().unwrap();
        t2.join().unwrap();

        let final_policy = mgr.policy_for(&TenantId::new("acme"));
        assert!(
            final_policy.cost_meter.is_some(),
            "cost_meter from t1 must survive the interleaving"
        );
        assert!(
            final_policy.redactor.is_some(),
            "redactor from t2 must survive the interleaving"
        );
    }

    #[test]
    fn mutate_fallback_visible_through_cloned_registries() {
        let mgr = PolicyRegistry::new();
        let cloned = mgr.clone();
        let meter = Arc::new(CostMeter::new(PricingTable::new()));
        cloned.mutate_fallback(|current| current.clone().with_cost_meter(meter.clone()));

        assert!(
            mgr.policy_for(&TenantId::new("never-registered"))
                .cost_meter
                .is_some(),
            "the original registry must observe a fallback mutation via a clone"
        );
    }

    #[test]
    fn re_register_overwrites() {
        let mgr = PolicyRegistry::new();
        mgr.register(TenantId::new("acme"), TenantPolicy::default());
        mgr.register(
            TenantId::new("acme"),
            TenantPolicy::new().with_redactor(Arc::new(RegexRedactor::with_defaults())),
        );
        assert!(mgr.policy_for(&TenantId::new("acme")).redactor.is_some());
        assert_eq!(mgr.tenant_count(), 1);
    }
}
