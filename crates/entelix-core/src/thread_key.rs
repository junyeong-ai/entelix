//! `ThreadKey` — the canonical `(tenant_id, thread_id)` addressing
//! tuple for every persistence and session operation in entelix.
//!
//! Encodes Invariant 11 (multi-tenant isolation) at the type level so
//! impls — `Checkpointer<S>`, `SessionLog`, future per-tenant state
//! handlers — cannot accidentally drop the tenant scope. A backend
//! receiving a `&ThreadKey` parameter cannot run a query missing the
//! `WHERE tenant_id = …` clause, because the tenant component is
//! syntactically required to read the thread component.
//!
//! Lives in `entelix-core` (rather than `entelix-graph`) so all
//! tenant-scoped subsystems — checkpointer, session log, future memory
//! companion crates — can address through one type without taking an
//! upward dependency on `entelix-graph`.

use serde::{Deserialize, Serialize};

use crate::context::ExecutionContext;
use crate::error::{Error, Result};
use crate::tenant_id::TenantId;

/// Canonical addressing tuple for every tenant-scoped persistence
/// operation — `(tenant_id, thread_id)`. Encodes Invariant 11
/// (multi-tenant isolation) at the type level so impls cannot
/// accidentally drop the tenant scope. The `tenant_id` carries the
/// validating [`TenantId`] newtype, so a `ThreadKey` whose serde
/// payload arrived with an empty tenant is rejected at deserialize
/// time rather than constructed and then silently mis-routed.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct ThreadKey {
    tenant_id: TenantId,
    thread_id: String,
}

impl ThreadKey {
    /// Build a key from a [`TenantId`] (already validated by its
    /// constructor) and a `thread_id` literal.
    ///
    /// # Panics
    ///
    /// Panics when `thread_id` is empty — empty would silently
    /// produce ambiguous keys (`"tenant:"`) that collide across
    /// logically distinct callers, defeating the Invariant 11
    /// isolation this type exists to enforce. Use [`Self::from_ctx`]
    /// in production paths; `new` is intended for tests and
    /// migration tooling that have already validated the inputs.
    #[must_use]
    pub fn new(tenant_id: TenantId, thread_id: impl Into<String>) -> Self {
        let thread_id = thread_id.into();
        assert!(
            !thread_id.is_empty(),
            "ThreadKey::new: thread_id must be non-empty"
        );
        Self {
            tenant_id,
            thread_id,
        }
    }

    /// Derive a key from an [`ExecutionContext`]. Returns
    /// [`Error::Config`] when the context carries no `thread_id` or
    /// the `thread_id` is empty — every persistence call requires
    /// both components be non-empty so two rows under different
    /// intent cannot share a key. The `tenant_id` is taken from the
    /// context's [`TenantId`] (already validated).
    pub fn from_ctx(ctx: &ExecutionContext) -> Result<Self> {
        let thread_id = ctx.thread_id().ok_or_else(|| {
            Error::config(
                "ThreadKey::from_ctx requires ExecutionContext::thread_id; \
                 set it via ExecutionContext::with_thread_id(...)",
            )
        })?;
        if thread_id.is_empty() {
            return Err(Error::config(
                "ThreadKey::from_ctx: thread_id must be non-empty",
            ));
        }
        Ok(Self {
            tenant_id: ctx.tenant_id().clone(),
            thread_id: thread_id.to_owned(),
        })
    }

    /// Borrow the tenant scope.
    #[must_use]
    pub const fn tenant_id(&self) -> &TenantId {
        &self.tenant_id
    }

    /// Borrow the conversation thread identifier.
    #[must_use]
    pub fn thread_id(&self) -> &str {
        &self.thread_id
    }
}
