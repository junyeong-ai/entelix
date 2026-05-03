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

/// Canonical addressing tuple for every tenant-scoped persistence
/// operation — `(tenant_id, thread_id)`. Encodes Invariant 11
/// (multi-tenant isolation) at the type level so impls cannot
/// accidentally drop the tenant scope.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct ThreadKey {
    tenant_id: String,
    thread_id: String,
}

impl ThreadKey {
    /// Build a key from raw components. Both `tenant_id` and
    /// `thread_id` must be non-empty — empty strings would
    /// silently produce ambiguous keys (`":"`, `"tenant:"`) that
    /// collide across logically distinct callers, defeating the
    /// Invariant 11 isolation this type exists to enforce.
    ///
    /// Use [`Self::from_ctx`] in production paths; `new` is
    /// intended for tests and migration tooling that have already
    /// validated the inputs.
    ///
    /// # Panics
    ///
    /// Panics when either component is empty. The panic is
    /// programmer-error grade — every production call site has a
    /// non-empty source (`ExecutionContext::tenant_id` defaults to
    /// `"default"`, and [`Self::from_ctx`] rejects a missing
    /// `thread_id` with [`Error::Config`]).
    #[must_use]
    pub fn new(tenant_id: impl Into<String>, thread_id: impl Into<String>) -> Self {
        let tenant_id = tenant_id.into();
        let thread_id = thread_id.into();
        assert!(
            !tenant_id.is_empty(),
            "ThreadKey::new: tenant_id must be non-empty"
        );
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
    /// [`Error::Config`] when the context carries no `thread_id`
    /// or either identifier is empty — every persistence call
    /// requires both components be non-empty so two rows under
    /// different intent cannot share a key.
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
        let tenant_id = ctx.tenant_id();
        if tenant_id.is_empty() {
            return Err(Error::config(
                "ThreadKey::from_ctx: tenant_id must be non-empty",
            ));
        }
        Ok(Self {
            tenant_id: tenant_id.to_owned(),
            thread_id: thread_id.to_owned(),
        })
    }

    /// Borrow the tenant scope.
    #[must_use]
    pub fn tenant_id(&self) -> &str {
        &self.tenant_id
    }

    /// Borrow the conversation thread identifier.
    #[must_use]
    pub fn thread_id(&self) -> &str {
        &self.thread_id
    }
}
