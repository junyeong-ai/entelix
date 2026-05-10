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

use entelix_core::{Result, TenantId};

use crate::cost::CostMeter;
use crate::pii::PiiRedactor;
use crate::quota::QuotaLimiter;

/// Per-tenant aggregate of policy handles. Cloning is cheap (`Arc`s
/// over the underlying primitives). Construct via
/// [`TenantPolicy::builder`].
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

    /// Start a builder that fills in optional handles fluently.
    #[must_use]
    pub fn builder() -> TenantPolicyBuilder {
        TenantPolicyBuilder::default()
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

/// Fluent builder for [`TenantPolicy`].
#[derive(Default)]
pub struct TenantPolicyBuilder {
    inner: TenantPolicy,
}

impl TenantPolicyBuilder {
    /// Bidirectional [`PiiRedactor`] applied on both `pre_request` and
    /// `post_response` (F5 — outbound-only redaction reintroduces the
    /// flaw).
    #[must_use]
    pub fn with_redactor(mut self, redactor: Arc<dyn PiiRedactor>) -> Self {
        self.inner.redactor = Some(redactor);
        self
    }

    /// Composite gate (rate + budget) evaluated `pre_request`.
    #[must_use]
    pub fn with_quota(mut self, quota: Arc<QuotaLimiter>) -> Self {
        self.inner.quota = Some(quota);
        self
    }

    /// `Ok`-branch-only metering — invariant 12 keeps failed calls
    /// out of the cost ledger.
    #[must_use]
    pub fn with_cost_meter(mut self, meter: Arc<CostMeter>) -> Self {
        self.inner.cost_meter = Some(meter);
        self
    }

    /// Returns `Result` for forward-compat with future validation
    /// (e.g. mutually-exclusive primitives, minimum-handle requirements);
    /// today the call cannot fail.
    pub fn build(self) -> Result<TenantPolicy> {
        Ok(self.inner)
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
        let policy = TenantPolicy::builder()
            .with_redactor(Arc::new(RegexRedactor::with_defaults()))
            .build()
            .unwrap();
        mgr.register(TenantId::new("acme"), policy);
        let p = mgr.policy_for(&TenantId::new("acme"));
        assert!(p.redactor.is_some());
        // Unregistered still gets fallback.
        let other = mgr.policy_for(&TenantId::new("other"));
        assert!(other.is_empty());
    }

    #[test]
    fn fallback_can_be_replaced_with_a_real_policy() {
        let fb = TenantPolicy::builder()
            .with_cost_meter(Arc::new(CostMeter::new(PricingTable::new())))
            .build()
            .unwrap();
        let mgr = PolicyRegistry::new().with_fallback(fb);
        let p = mgr.policy_for(&TenantId::new("never-registered"));
        assert!(p.cost_meter.is_some());
    }

    #[test]
    fn fallback_replacement_is_observed_by_cloned_registries() {
        let mgr = PolicyRegistry::new();
        let cloned = mgr.clone();

        let fb = TenantPolicy::builder()
            .with_cost_meter(Arc::new(CostMeter::new(PricingTable::new())))
            .build()
            .unwrap();
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
    fn re_register_overwrites() {
        let mgr = PolicyRegistry::new();
        mgr.register(TenantId::new("acme"), TenantPolicy::default());
        mgr.register(
            TenantId::new("acme"),
            TenantPolicy::builder()
                .with_redactor(Arc::new(RegexRedactor::with_defaults()))
                .build()
                .unwrap(),
        );
        assert!(mgr.policy_for(&TenantId::new("acme")).redactor.is_some());
        assert_eq!(mgr.tenant_count(), 1);
    }
}
