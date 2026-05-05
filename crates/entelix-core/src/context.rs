//! Request-scope execution context.
//!
//! `ExecutionContext` is the only context object that flows through
//! `Runnable::invoke`, `Tool::execute`, hooks, and codecs. It carries:
//!
//! - a `CancellationToken` (F3 mitigation — cooperative cancellation)
//! - an optional deadline (`tokio::time::Instant`)
//! - an optional `thread_id` — the durable conversation identifier used
//!   by `Checkpointer` and `SessionGraph` to scope persistence.
//! - a mandatory `tenant_id` — multi-tenant scope (invariant 11). Defaults
//!   to [`crate::DEFAULT_TENANT_ID`] when not specified explicitly.
//!
//! It deliberately does **not** embed a `CredentialProvider` (invariant 10 —
//! tokens never reach Tool input). Future fields will be added as the layers
//! materialize: span context, cost meter handle. Each addition is reviewed
//! against the no-tokens-in-tools rule.
//!
//! `ExecutionContext` is `Clone` (cheap) so combinators can fan it out to
//! parallel branches without lifetime gymnastics.

use std::sync::Arc;

use tokio::time::Instant;

use crate::audit::AuditSinkHandle;
use crate::cancellation::CancellationToken;
use crate::extensions::Extensions;
use crate::tenant_id::TenantId;

/// Carrier for request-scope state that every `Runnable`, `Tool`,
/// and codec sees.
///
/// Marked `#[non_exhaustive]`: fields are private; callers always go
/// through [`Self::new`] and the `with_*` builder methods, so adding
/// a new field is a non-breaking change.
///
/// `tenant_id` is a [`TenantId`] (validating `Arc<str>` newtype, see
/// [`crate::tenant_id`]). Cloning the context — done implicitly per
/// tool dispatch and per sub-agent — bumps the underlying refcount
/// instead of allocating. The default-tenant `TenantId` is shared
/// process-wide via a `OnceLock`, so a freshly-built context
/// allocates zero strings on the hot path.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ExecutionContext {
    cancellation: CancellationToken,
    deadline: Option<Instant>,
    thread_id: Option<String>,
    tenant_id: TenantId,
    run_id: Option<String>,
    /// Idempotency key for the current logical call — vendor-side
    /// dedupe identifier shared across every retry attempt of the
    /// same logical call. `RetryService` stamps it on first entry
    /// when absent so two retries of one timed-out request do not
    /// produce two charges (invariant #17 — vendor-authoritative
    /// guarantees beat self-jitter).
    idempotency_key: Option<Arc<str>>,
    extensions: Extensions,
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionContext {
    /// Create a fresh context with a new cancellation token, no deadline,
    /// no thread ID, and the [`TenantId::default()`] tenant scope
    /// (which is the process-wide shared `"default"` `TenantId`).
    pub fn new() -> Self {
        Self {
            cancellation: CancellationToken::new(),
            deadline: None,
            thread_id: None,
            tenant_id: TenantId::default(),
            run_id: None,
            idempotency_key: None,
            extensions: Extensions::new(),
        }
    }

    /// Create a context bound to an existing cancellation token (e.g. a parent
    /// agent's token, so cancelling the parent cascades to children).
    pub fn with_cancellation(cancellation: CancellationToken) -> Self {
        Self {
            cancellation,
            deadline: None,
            thread_id: None,
            tenant_id: TenantId::default(),
            run_id: None,
            idempotency_key: None,
            extensions: Extensions::new(),
        }
    }

    /// Attach a deadline. Returns `self` for builder-style chaining.
    #[must_use]
    pub const fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Attach a `thread_id` — the durable conversation key used by
    /// `Checkpointer`s to scope persistence.
    #[must_use]
    pub fn with_thread_id(mut self, thread_id: impl Into<String>) -> Self {
        self.thread_id = Some(thread_id.into());
        self
    }

    /// Override the tenant scope. Multi-tenant operators set this per
    /// request; single-tenant deployments leave it at
    /// [`TenantId::default()`] (invariant 11).
    #[must_use]
    pub fn with_tenant_id(mut self, tenant_id: TenantId) -> Self {
        self.tenant_id = tenant_id;
        self
    }

    /// Attach a `run_id` — the per-execute correlation key the agent
    /// runtime stamps on every `AgentEvent`, OTel span, and tool
    /// invocation event. Recipes composing agents inside a larger
    /// graph pre-allocate the id so all events share a tree;
    /// otherwise `Agent::execute` generates a fresh UUID v7 on entry.
    #[must_use]
    pub fn with_run_id(mut self, run_id: impl Into<String>) -> Self {
        self.run_id = Some(run_id.into());
        self
    }

    /// Attach a [`crate::RunBudget`] — five-axis usage cap shared
    /// across the run (parent agent + every sub-agent it
    /// dispatches). Cloning the context bumps the budget's
    /// internal `Arc` refcount so sub-agent dispatches accumulate
    /// into the same counters.
    ///
    /// Stored in [`Extensions`] under `RunBudget`'s `TypeId`, so
    /// a second call replaces the prior budget. Read sites
    /// reach for it via [`Self::run_budget`]; absent budget
    /// makes every dispatch site's check / observe a no-op.
    #[must_use]
    pub fn with_run_budget(self, budget: crate::RunBudget) -> Self {
        self.add_extension(budget)
    }

    /// Borrow the [`crate::RunBudget`] attached via
    /// [`Self::with_run_budget`], if any. Dispatch sites
    /// (`ChatModel::complete_full` / `complete_typed` /
    /// `stream_deltas`, `ToolRegistry::dispatch_*`) gate budget
    /// checks on `Some(_)` so a context without a budget incurs
    /// zero overhead beyond the `TypeId` lookup.
    #[must_use]
    pub fn run_budget(&self) -> Option<std::sync::Arc<crate::RunBudget>> {
        self.extension::<crate::RunBudget>()
    }

    /// Attach an idempotency key — the vendor-side dedupe identifier
    /// shared across every retry attempt of one logical call.
    /// Operators that pre-allocate the key (e.g. for cross-process
    /// dedupe through a sticky job id) call this before dispatch;
    /// otherwise `RetryService` stamps a UUID on first entry.
    #[must_use]
    pub fn with_idempotency_key(mut self, key: impl Into<String>) -> Self {
        self.idempotency_key = Some(Arc::from(key.into()));
        self
    }

    /// Mutable handle to the idempotency-key slot — `RetryService`
    /// uses this to stamp a fresh key on first entry of a retry
    /// loop (so attempts 2..N share the slot the first attempt set).
    /// Outside the retry middleware, prefer
    /// [`Self::with_idempotency_key`].
    pub fn ensure_idempotency_key<F>(&mut self, generate: F) -> &str
    where
        F: FnOnce() -> String,
    {
        if self.idempotency_key.is_none() {
            self.idempotency_key = Some(Arc::from(generate()));
        }
        // SAFETY: branch above guarantees Some.
        self.idempotency_key.as_deref().unwrap_or("")
    }

    /// Attach an [`AuditSinkHandle`] for the current run. Sub-agents,
    /// supervisors, and memory tools look it up via
    /// [`Self::audit_sink`] and emit typed
    /// [`crate::audit::AuditSink`] events; the absent handle makes
    /// every emit site a no-op.
    ///
    /// Stored in [`Extensions`] under `AuditSinkHandle`'s `TypeId`,
    /// so a second call replaces the prior handle (single sink per
    /// run by design — recipes that need fan-out wrap two sinks
    /// behind one impl).
    #[must_use]
    pub fn with_audit_sink(self, handle: AuditSinkHandle) -> Self {
        self.add_extension(handle)
    }

    /// Borrow the [`AuditSinkHandle`] for the current run, if one
    /// has been attached via [`Self::with_audit_sink`]. Emit sites
    /// gate on `Some(_)` so a context without a sink incurs no
    /// overhead beyond the `TypeId` lookup.
    #[must_use]
    pub fn audit_sink(&self) -> Option<Arc<AuditSinkHandle>> {
        self.extension::<AuditSinkHandle>()
    }

    /// Attach a typed cross-cutting value to the context's
    /// [`Extensions`] slot. The returned context carries a fresh
    /// `Extensions` Arc with `value` registered under `T`'s
    /// `TypeId`; the caller's previous context is unchanged
    /// (copy-on-write).
    ///
    /// One entry per type — calling this twice with the same `T`
    /// replaces the earlier value. Operators threading multiple
    /// values of the same logical category wrap them in a domain
    /// type (`struct WorkspaceCtx { repo: ..., flags: ... }`) so
    /// `T` stays the canonical key.
    ///
    /// Read back via [`Self::extension`] inside any tool, codec,
    /// or runnable that sees the context.
    ///
    /// **Invariant 10**: do not stash credentials or bearer
    /// tokens here — `ExecutionContext` flows into `Tool::execute`
    /// and an extension carrying a token would surface it to
    /// every tool the agent dispatches. Credentials live in
    /// transports (`CredentialProvider`).
    #[must_use]
    pub fn add_extension<T>(mut self, value: T) -> Self
    where
        T: Send + Sync + 'static,
    {
        self.extensions = self.extensions.inserted(value);
        self
    }

    /// Borrow the cancellation token. Long-running tools should periodically
    /// check `is_cancelled()` and cooperatively shut down.
    pub const fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }

    /// Returns the deadline if one was attached.
    pub const fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    /// Borrow the thread identifier if one was attached.
    pub fn thread_id(&self) -> Option<&str> {
        self.thread_id.as_deref()
    }

    /// Borrow the tenant identifier (invariant 11). Always present —
    /// defaults to the process-shared [`TenantId::default()`] when
    /// not explicitly set.
    pub const fn tenant_id(&self) -> &TenantId {
        &self.tenant_id
    }

    /// Borrow the per-execute correlation id, if one has been
    /// stamped (typically by `Agent::execute` on entry, or by a
    /// caller pre-allocating via [`Self::with_run_id`]).
    pub fn run_id(&self) -> Option<&str> {
        self.run_id.as_deref()
    }

    /// Borrow the idempotency key for the current logical call, if
    /// one has been stamped (by `RetryService` on first entry of a
    /// retry loop, or by a caller pre-allocating via
    /// [`Self::with_idempotency_key`]). `DirectTransport` and the
    /// cloud transports forward this on the `Idempotency-Key`
    /// request header so vendors dedupe retries server-side.
    pub fn idempotency_key(&self) -> Option<&str> {
        self.idempotency_key.as_deref()
    }

    /// Borrow the typed cross-cutting carrier. Use
    /// [`Self::extension`] for the common single-type lookup;
    /// reach for the carrier directly only when iterating over
    /// cardinality or composing a downstream context that should
    /// inherit the same set.
    pub const fn extensions(&self) -> &Extensions {
        &self.extensions
    }

    /// Look up the extension entry registered for `T`, returning
    /// a refcounted handle independent of the context's lifetime.
    /// Returns `None` when the entry is absent — operators reading
    /// optional cross-cutting state code their own default.
    #[must_use]
    pub fn extension<T>(&self) -> Option<Arc<T>>
    where
        T: Send + Sync + 'static,
    {
        self.extensions.get::<T>()
    }

    /// Convenience: did the cancellation token fire?
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }

    /// Derive a scoped child context. The child inherits `deadline`,
    /// `thread_id`, and `tenant_id` but holds a *child* cancellation
    /// token: cancelling the child does NOT cancel the parent, but
    /// cancelling the parent cascades to the child.
    ///
    /// Use for scope-bounded fan-out (e.g. [`scatter`]) where a
    /// fail-fast branch should signal still-running siblings to
    /// abort cooperatively without tearing the whole request down.
    ///
    /// [`scatter`]: ../entelix_graph/fn.scatter.html
    #[must_use]
    pub fn child(&self) -> Self {
        Self {
            cancellation: self.cancellation.child_token(),
            deadline: self.deadline,
            thread_id: self.thread_id.clone(),
            tenant_id: self.tenant_id.clone(),
            run_id: self.run_id.clone(),
            idempotency_key: self.idempotency_key.clone(),
            extensions: self.extensions.clone(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod extension_tests {
    use super::*;

    #[derive(Debug, PartialEq, Eq)]
    struct WorkspaceCtx {
        repo: &'static str,
    }

    #[test]
    fn fresh_context_has_no_extensions() {
        let ctx = ExecutionContext::new();
        assert!(ctx.extensions().is_empty());
        assert!(ctx.extension::<WorkspaceCtx>().is_none());
    }

    #[test]
    fn add_extension_threads_typed_value() {
        let ctx = ExecutionContext::new().add_extension(WorkspaceCtx { repo: "entelix" });
        let got = ctx.extension::<WorkspaceCtx>().unwrap();
        assert_eq!(*got, WorkspaceCtx { repo: "entelix" });
        assert_eq!(ctx.extensions().len(), 1);
    }

    #[test]
    fn add_extension_is_copy_on_write() {
        let original = ExecutionContext::new();
        let extended = original
            .clone()
            .add_extension(WorkspaceCtx { repo: "entelix" });
        // Original unchanged.
        assert!(original.extension::<WorkspaceCtx>().is_none());
        // Extended carries the value.
        assert!(extended.extension::<WorkspaceCtx>().is_some());
    }

    #[test]
    fn child_inherits_extensions() {
        let parent = ExecutionContext::new().add_extension(WorkspaceCtx { repo: "entelix" });
        let child = parent.child();
        let got = child.extension::<WorkspaceCtx>().unwrap();
        assert_eq!(*got, WorkspaceCtx { repo: "entelix" });
    }

    #[test]
    fn extension_arc_outlives_dropped_context() {
        let ctx = ExecutionContext::new().add_extension(WorkspaceCtx { repo: "entelix" });
        let arc = ctx.extension::<WorkspaceCtx>().unwrap();
        drop(ctx);
        assert_eq!(*arc, WorkspaceCtx { repo: "entelix" });
    }
}
