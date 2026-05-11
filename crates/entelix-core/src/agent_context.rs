//! Typed-deps carrier — separates infra context from operator-side handles.
//!
//! [`AgentContext<D>`] wraps an [`ExecutionContext`] (the D-free infra
//! carrier — cancellation, tenant scope, deadline, audit sink,
//! extensions) with a typed `D` slot for operator-side dependencies
//! (database pool, HTTP client, tenant config, ...).
//!
//! ## Why split?
//!
//! Layers and the tower service spine (`PolicyLayer`, `OtelLayer`,
//! `RetryService`, ...) operate on [`ExecutionContext`] and have no
//! reason to know `D`. Forcing `D` through every layer factory
//! leads to generic explosion. `AgentContext<D>` is the type the
//! `Tool` and validator surfaces see; [`AgentContext::core`] is
//! what layers see.
//!
//! Invariant 4 (Hand contract) and invariant 10 (no operator-side
//! handles via dynamic context) together require a typed slot that
//! does NOT ride [`Extensions`] — extensions are best-effort
//! dynamic carriers; `D` is a *static* contract on the agent's
//! shape and must be visible to the type system.
//!
//! ## `D` defaults to `()`
//!
//! Agents and tools that need no operator-side state stay
//! `AgentContext<()>`. [`Default`], [`From<ExecutionContext>`], and
//! the [`AgentContext::default`] constructor all target the unit
//! shape so the deps-less path is ergonomically zero-cost.
//!
//! ```
//! use entelix_core::AgentContext;
//!
//! // Deps-less — Default / From all target `()`.
//! let ctx = AgentContext::default();
//! assert_eq!(ctx.deps(), &());
//!
//! // Typed deps — operator-side handle threaded into tools.
//! #[derive(Clone, Debug, PartialEq)]
//! struct AppDeps {
//!     tenant_label: &'static str,
//! }
//! let ctx = AgentContext::new(
//!     Default::default(),
//!     AppDeps { tenant_label: "acme" },
//! );
//! assert_eq!(ctx.deps().tenant_label, "acme");
//! ```
//!
//! [`Extensions`]: crate::extensions::Extensions

use std::sync::Arc;

use tokio::time::Instant;

use crate::audit::AuditSinkHandle;
use crate::cancellation::CancellationToken;
use crate::context::ExecutionContext;
use crate::extensions::Extensions;
use crate::run_budget::RunBudget;
use crate::tenant_id::TenantId;
use crate::tools::{ToolProgressSinkHandle, ToolProgressStatus};

/// Per-run carrier of infra context + typed operator-side
/// dependencies. `D` defaults to `()` so deps-less agents pay no
/// type-system tax.
///
/// `AgentContext<D>` flows into the `Tool` and validator surfaces.
/// Layers consume [`ExecutionContext`] only, reached via
/// [`Self::core`]; this keeps the layer ecosystem (`tower::Service`
/// spine) D-free and avoids generic explosion.
///
/// Cloning a context with `D: Clone` is shallow: the inner
/// [`ExecutionContext`] clones cheaply (`Arc` refcounts on tenant
/// id, extensions, audit sink) and `D` clones with whatever
/// semantics the operator gave it. Cloning is the canonical path
/// for sub-agent dispatch — children share the parent's deps and
/// inherit a child cancellation token via [`Self::child`].
pub struct AgentContext<D = ()> {
    core: ExecutionContext,
    deps: D,
}

impl<D> AgentContext<D> {
    /// Construct from an explicit [`ExecutionContext`] and `D`.
    pub const fn new(core: ExecutionContext, deps: D) -> Self {
        Self { core, deps }
    }

    /// Borrow the wrapped [`ExecutionContext`]. Layers and the
    /// tower service spine (`ModelInvocation` / `ToolInvocation`)
    /// consume the core directly so they stay D-free.
    pub const fn core(&self) -> &ExecutionContext {
        &self.core
    }

    /// Mutable handle to the wrapped [`ExecutionContext`]. The
    /// retry middleware reaches for this to stamp an idempotency
    /// key on first entry of a retry loop; outside that path,
    /// prefer the `with_*` builder methods.
    pub const fn core_mut(&mut self) -> &mut ExecutionContext {
        &mut self.core
    }

    /// Borrow the typed operator-side deps. Tools consume this
    /// directly; the type system enforces that the deps shape
    /// matches the agent's `D` parameter.
    pub const fn deps(&self) -> &D {
        &self.deps
    }

    /// Mutable handle to the typed deps. Operators threading
    /// per-run mutable state (counters, accumulators) reach for
    /// this; most deps are `Arc<...>` and immutable, so the
    /// shared-borrow [`Self::deps`] suffices.
    pub const fn deps_mut(&mut self) -> &mut D {
        &mut self.deps
    }

    /// Decompose into `(core, deps)`. Useful when bridging to
    /// APIs that consume each half independently.
    #[allow(clippy::missing_const_for_fn)]
    pub fn into_parts(self) -> (ExecutionContext, D) {
        (self.core, self.deps)
    }

    /// Transform `D` into `E` while preserving the infra core.
    /// Sub-agents that need a narrower deps shape than the parent
    /// project the parent's deps through this combinator at
    /// dispatch time.
    pub fn map_deps<E, F>(self, f: F) -> AgentContext<E>
    where
        F: FnOnce(D) -> E,
    {
        AgentContext {
            core: self.core,
            deps: f(self.deps),
        }
    }

    // ---- forwarders to ExecutionContext (ergonomics) -----------

    /// Borrow the cancellation token (forwarded from
    /// [`ExecutionContext::cancellation`]).
    pub const fn cancellation(&self) -> &CancellationToken {
        self.core.cancellation()
    }

    /// Returns the deadline if one was attached.
    pub const fn deadline(&self) -> Option<Instant> {
        self.core.deadline()
    }

    /// Borrow the thread identifier if one was attached.
    pub fn thread_id(&self) -> Option<&str> {
        self.core.thread_id()
    }

    /// Borrow the tenant identifier (invariant 11). Always
    /// present — defaults to the process-shared
    /// [`TenantId::default`] when not explicitly set.
    pub const fn tenant_id(&self) -> &TenantId {
        self.core.tenant_id()
    }

    /// Borrow the per-execute correlation id, if one has been
    /// stamped.
    pub fn run_id(&self) -> Option<&str> {
        self.core.run_id()
    }

    /// Borrow the idempotency key for the current logical call,
    /// if one has been stamped (by `RetryService` on first entry
    /// of a retry loop, or pre-allocated by the caller).
    pub fn idempotency_key(&self) -> Option<&str> {
        self.core.idempotency_key()
    }

    /// Convenience: did the cancellation token fire?
    pub fn is_cancelled(&self) -> bool {
        self.core.is_cancelled()
    }

    /// Borrow the typed cross-cutting carrier.
    pub const fn extensions(&self) -> &Extensions {
        self.core.extensions()
    }

    /// Look up the extension entry registered for `T`. See
    /// [`ExecutionContext::extension`].
    #[must_use]
    pub fn extension<T>(&self) -> Option<Arc<T>>
    where
        T: Send + Sync + 'static,
    {
        self.core.extension::<T>()
    }

    /// Borrow the [`RunBudget`] attached to the run, if any.
    #[must_use]
    pub fn run_budget(&self) -> Option<Arc<RunBudget>> {
        self.core.run_budget()
    }

    /// Borrow the [`AuditSinkHandle`] attached to the run, if any.
    #[must_use]
    pub fn audit_sink(&self) -> Option<Arc<AuditSinkHandle>> {
        self.core.audit_sink()
    }

    /// Borrow the [`ToolProgressSinkHandle`] attached to the run, if
    /// any.
    #[must_use]
    pub fn tool_progress_sink(&self) -> Option<Arc<ToolProgressSinkHandle>> {
        self.core.tool_progress_sink()
    }

    /// Emit a tool-phase transition with no metadata. Silent no-op
    /// when no sink is attached or the call is outside a tool
    /// dispatch. Fire-and-forget — sink failures stay inside the
    /// sink. Forwards to [`ExecutionContext::record_phase`].
    pub async fn record_phase(&self, phase: impl Into<String> + Send, status: ToolProgressStatus)
    where
        D: Sync,
    {
        self.core.record_phase(phase, status).await;
    }

    /// Emit a tool-phase transition with structured metadata.
    /// Forwards to [`ExecutionContext::record_phase_with`].
    pub async fn record_phase_with(
        &self,
        phase: impl Into<String> + Send,
        status: ToolProgressStatus,
        metadata: serde_json::Value,
    ) where
        D: Sync,
    {
        self.core.record_phase_with(phase, status, metadata).await;
    }

    // ---- builder methods (delegate to core) --------------------

    /// Attach a deadline. Returns `self` for builder-style chaining.
    #[must_use]
    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.core = self.core.with_deadline(deadline);
        self
    }

    /// Attach a `thread_id`.
    #[must_use]
    pub fn with_thread_id(mut self, thread_id: impl Into<String>) -> Self {
        self.core = self.core.with_thread_id(thread_id);
        self
    }

    /// Override the tenant scope.
    #[must_use]
    pub fn with_tenant_id(mut self, tenant_id: TenantId) -> Self {
        self.core = self.core.with_tenant_id(tenant_id);
        self
    }

    /// Attach a `run_id`.
    #[must_use]
    pub fn with_run_id(mut self, run_id: impl Into<String>) -> Self {
        self.core = self.core.with_run_id(run_id);
        self
    }

    /// Attach an idempotency key.
    #[must_use]
    pub fn with_idempotency_key(mut self, key: impl Into<String>) -> Self {
        self.core = self.core.with_idempotency_key(key);
        self
    }

    /// Attach a [`RunBudget`].
    #[must_use]
    pub fn with_run_budget(mut self, budget: RunBudget) -> Self {
        self.core = self.core.with_run_budget(budget);
        self
    }

    /// Attach an [`AuditSinkHandle`].
    #[must_use]
    pub fn with_audit_sink(mut self, handle: AuditSinkHandle) -> Self {
        self.core = self.core.with_audit_sink(handle);
        self
    }

    /// Attach a [`ToolProgressSinkHandle`].
    #[must_use]
    pub fn with_tool_progress_sink(mut self, handle: ToolProgressSinkHandle) -> Self {
        self.core = self.core.with_tool_progress_sink(handle);
        self
    }

    /// Attach a typed cross-cutting value to the [`Extensions`]
    /// slot. See [`ExecutionContext::add_extension`] for the
    /// invariant 10 boundary.
    #[must_use]
    pub fn add_extension<T>(mut self, value: T) -> Self
    where
        T: Send + Sync + 'static,
    {
        self.core = self.core.add_extension(value);
        self
    }
}

impl<D: Clone> AgentContext<D> {
    /// Derive a scoped child. The child inherits `deadline`,
    /// `thread_id`, `tenant_id`, `run_id`, `idempotency_key`, and
    /// `extensions`, but holds a *child* cancellation token and a
    /// clone of `deps`. Cancelling the parent cascades; cancelling
    /// the child does not.
    ///
    /// Use for scope-bounded fan-out (sub-agent dispatch, scatter
    /// branches) where a fail-fast leaf should signal still-running
    /// siblings to abort cooperatively without tearing the whole
    /// request down.
    #[must_use]
    pub fn child(&self) -> Self {
        Self {
            core: self.core.child(),
            deps: self.deps.clone(),
        }
    }
}

impl<D: Clone> Clone for AgentContext<D> {
    fn clone(&self) -> Self {
        Self {
            core: self.core.clone(),
            deps: self.deps.clone(),
        }
    }
}

impl<D: std::fmt::Debug> std::fmt::Debug for AgentContext<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentContext")
            .field("core", &self.core)
            .field("deps", &self.deps)
            .finish()
    }
}

impl Default for AgentContext<()> {
    fn default() -> Self {
        Self {
            core: ExecutionContext::default(),
            deps: (),
        }
    }
}

impl From<ExecutionContext> for AgentContext<()> {
    fn from(core: ExecutionContext) -> Self {
        Self { core, deps: () }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct AppDeps {
        tenant_label: &'static str,
    }

    #[test]
    fn default_targets_unit_deps() {
        let ctx = AgentContext::<()>::default();
        assert_eq!(ctx.deps(), &());
        // Tenant defaults to the process-shared default tenant.
        assert_eq!(ctx.tenant_id(), &TenantId::default());
    }

    #[test]
    fn from_execution_context_wraps_with_unit_deps() {
        let core = ExecutionContext::new().with_thread_id("t-1");
        let ctx: AgentContext<()> = core.into();
        assert_eq!(ctx.deps(), &());
        assert_eq!(ctx.thread_id(), Some("t-1"));
    }

    #[test]
    fn typed_deps_thread_through_constructor() {
        let ctx = AgentContext::new(
            ExecutionContext::default(),
            AppDeps {
                tenant_label: "acme",
            },
        );
        assert_eq!(ctx.deps().tenant_label, "acme");
    }

    #[test]
    fn forwarders_match_core() {
        let core = ExecutionContext::new()
            .with_thread_id("t-2")
            .with_run_id("r-2");
        let ctx = AgentContext::new(core.clone(), AppDeps { tenant_label: "x" });
        assert_eq!(ctx.thread_id(), core.thread_id());
        assert_eq!(ctx.run_id(), core.run_id());
        assert_eq!(ctx.tenant_id(), core.tenant_id());
        assert_eq!(ctx.is_cancelled(), core.is_cancelled());
    }

    #[test]
    fn into_parts_decomposes_and_round_trips() {
        let deps = AppDeps {
            tenant_label: "round-trip",
        };
        let ctx = AgentContext::new(ExecutionContext::default(), deps.clone());
        let (core, recovered) = ctx.into_parts();
        assert_eq!(recovered, deps);
        // Re-wrap and confirm the deps survive a parts round-trip.
        let again = AgentContext::new(core, recovered);
        assert_eq!(again.deps().tenant_label, "round-trip");
    }

    #[test]
    fn map_deps_transforms_typed_handle() {
        let ctx = AgentContext::new(
            ExecutionContext::default(),
            AppDeps {
                tenant_label: "before",
            },
        );
        let mapped = ctx.map_deps(|d| d.tenant_label.to_owned());
        assert_eq!(mapped.deps(), "before");
    }

    #[test]
    fn child_clones_deps_and_branches_cancellation() {
        let parent = AgentContext::new(ExecutionContext::default(), AppDeps { tenant_label: "p" });
        let child = parent.child();
        // Deps clone.
        assert_eq!(child.deps(), parent.deps());
        // Cancelling the child does NOT cancel the parent.
        child.cancellation().cancel();
        assert!(child.is_cancelled());
        assert!(!parent.is_cancelled());
    }

    #[test]
    fn parent_cancellation_cascades_to_child() {
        let parent = AgentContext::new(ExecutionContext::default(), AppDeps { tenant_label: "p" });
        let child = parent.child();
        parent.cancellation().cancel();
        assert!(child.is_cancelled());
    }

    #[test]
    fn with_deadline_delegates_to_core() {
        let deadline = Instant::now() + std::time::Duration::from_mins(1);
        let ctx = AgentContext::default().with_deadline(deadline);
        assert_eq!(ctx.deadline(), Some(deadline));
        assert_eq!(ctx.core().deadline(), Some(deadline));
    }

    #[test]
    fn with_thread_id_threads_through_core() {
        let ctx = AgentContext::default().with_thread_id("t-3");
        assert_eq!(ctx.thread_id(), Some("t-3"));
        assert_eq!(ctx.core().thread_id(), Some("t-3"));
    }

    #[test]
    fn with_tenant_id_overrides_default() {
        let tid = TenantId::new("isolated");
        let ctx = AgentContext::default().with_tenant_id(tid.clone());
        assert_eq!(ctx.tenant_id(), &tid);
    }

    #[test]
    fn with_run_id_attaches_correlation() {
        let ctx = AgentContext::default().with_run_id("run-99");
        assert_eq!(ctx.run_id(), Some("run-99"));
    }

    #[test]
    fn with_idempotency_key_threads_through_core() {
        let ctx = AgentContext::default().with_idempotency_key("idem-99");
        assert_eq!(ctx.idempotency_key(), Some("idem-99"));
    }

    #[derive(Debug, PartialEq, Eq)]
    struct WorkspaceCtx {
        repo: &'static str,
    }

    #[test]
    fn add_extension_typed_lookup_via_forwarder() {
        let ctx = AgentContext::default().add_extension(WorkspaceCtx { repo: "entelix" });
        let got = ctx.extension::<WorkspaceCtx>().unwrap();
        assert_eq!(*got, WorkspaceCtx { repo: "entelix" });
    }

    #[test]
    fn add_extension_does_not_alter_deps() {
        let ctx = AgentContext::new(
            ExecutionContext::default(),
            AppDeps {
                tenant_label: "preserve",
            },
        )
        .add_extension(WorkspaceCtx { repo: "entelix" });
        assert_eq!(ctx.deps().tenant_label, "preserve");
        assert!(ctx.extension::<WorkspaceCtx>().is_some());
    }

    #[test]
    fn run_budget_forwarder_returns_attached_handle() {
        let budget = RunBudget::default();
        let ctx = AgentContext::default().with_run_budget(budget);
        assert!(ctx.run_budget().is_some());
    }

    #[test]
    fn clone_shares_extensions_via_arc_refcount() {
        let original = AgentContext::default().add_extension(WorkspaceCtx { repo: "entelix" });
        let cloned = original.clone();
        // Both see the same extension entry.
        assert!(original.extension::<WorkspaceCtx>().is_some());
        assert!(cloned.extension::<WorkspaceCtx>().is_some());
    }

    #[test]
    fn debug_includes_core_and_deps() {
        let ctx = AgentContext::new(
            ExecutionContext::default(),
            AppDeps {
                tenant_label: "debug",
            },
        );
        let formatted = format!("{ctx:?}");
        assert!(formatted.contains("AgentContext"));
        assert!(formatted.contains("debug"));
    }
}
