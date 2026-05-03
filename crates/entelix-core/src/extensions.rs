//! `Extensions` — type-keyed cross-cutting carrier rendered through
//! [`crate::ExecutionContext`].
//!
//! ## Why a `TypeMap` slot
//!
//! The `ExecutionContext` first-class fields (`tenant_id`,
//! `cancellation`, `deadline`, `run_id`, `thread_id`) cover the
//! cross-cutting concerns the SDK itself reasons about. Operators
//! who need to thread *their own* request-scope data — a workspace
//! handle, a per-run cache, a tenant-specific rate-limiter, a custom
//! telemetry tag — would otherwise reach for `thread_local!` or
//! `Arc`-passing through bespoke parameters. `Extensions` gives them
//! a typed slot on the same context the rest of the SDK already
//! threads through `Runnable`, `Tool`, and codec layers.
//!
//! The pattern mirrors `http::Extensions`, `axum::Extensions`, and
//! `tower::Service` request extensions — industry-standard cross-cutting
//! carriers. `Extensions` differs only in being thread-safe by
//! construction (entries are `Send + Sync`, the map is shared via
//! `Arc`) so multi-tenant deployments can clone the context across
//! parallel branches without lock juggling.
//!
//! ## Copy-on-write semantics
//!
//! `Extensions` is *immutable after construction*. The builder
//! pattern on [`crate::ExecutionContext::add_extension`] returns a
//! new `ExecutionContext` carrying a fresh `Arc<Extensions>`; the
//! caller's original context is unchanged. That preserves the
//! "context is cheap to clone" guarantee — combinators that fan out
//! to parallel branches see consistent, non-mutating state.
//!
//! ## Invariant 10 (no tokens in tools)
//!
//! `Extensions` is *not* a credentials channel. Operators who stash
//! a [`crate::auth::CredentialProvider`] handle here would surface
//! the credential value to every `Tool::execute` site that touches
//! the context — a structural violation. Credentials live in
//! transports; the trait surface for this slot deliberately
//! does not advertise an "auth" affordance.
//!
//! ## Type-erased storage
//!
//! Entries are stored as `Arc<dyn Any + Send + Sync>` keyed by
//! [`std::any::TypeId`]. Insertions of the same `T` overwrite (one
//! entry per type) — this matches the `http::Extensions` shape and
//! avoids the "list of `T`" question that ad-hoc carriers struggle
//! with.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;

/// Type-keyed cross-cutting state attached to an
/// [`crate::ExecutionContext`]. Operators add their own values via
/// [`crate::ExecutionContext::add_extension`] and read them back
/// via [`crate::ExecutionContext::extension`].
///
/// Cloning is cheap — internally an `Arc` over the underlying map,
/// so cloning a context that already carries extensions does not
/// duplicate the entries.
#[derive(Clone, Default)]
pub struct Extensions {
    inner: Arc<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>,
}

impl Extensions {
    /// Create an empty `Extensions`. Equivalent to
    /// [`Default::default`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of entries currently stored. Diagnostic helper —
    /// production code rarely cares.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// True when the carrier has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Look up the entry registered for `T`. Returns `None` when
    /// no value of that type has been inserted.
    ///
    /// The returned `Arc<T>` is a fresh refcount bump — the caller
    /// can hold it across awaits without keeping the
    /// `ExecutionContext` alive.
    #[must_use]
    pub fn get<T>(&self) -> Option<Arc<T>>
    where
        T: Send + Sync + 'static,
    {
        self.inner
            .get(&TypeId::of::<T>())
            .and_then(|entry| Arc::clone(entry).downcast::<T>().ok())
    }

    /// True when the carrier has an entry for `T`.
    #[must_use]
    pub fn contains<T>(&self) -> bool
    where
        T: Send + Sync + 'static,
    {
        self.inner.contains_key(&TypeId::of::<T>())
    }

    /// Return a new `Extensions` with `value` inserted under the
    /// type id of `T`. An existing entry for the same type is
    /// replaced (one entry per type, mirroring `http::Extensions`).
    ///
    /// Internal helper — operators reach this through
    /// [`crate::ExecutionContext::add_extension`].
    #[must_use]
    pub(crate) fn inserted<T>(&self, value: T) -> Self
    where
        T: Send + Sync + 'static,
    {
        let mut next: HashMap<TypeId, Arc<dyn Any + Send + Sync>> = (*self.inner).clone();
        next.insert(TypeId::of::<T>(), Arc::new(value));
        Self {
            inner: Arc::new(next),
        }
    }
}

impl std::fmt::Debug for Extensions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The values themselves are `dyn Any` so we can't print
        // them — surfacing the cardinality keeps the debug output
        // honest for log review and crash dumps.
        f.debug_struct("Extensions")
            .field("len", &self.inner.len())
            .finish()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Eq)]
    struct Workspace(&'static str);

    #[derive(Debug, PartialEq, Eq)]
    struct RequestId(u64);

    #[test]
    fn empty_extensions_have_no_entries() {
        let ext = Extensions::new();
        assert!(ext.is_empty());
        assert_eq!(ext.len(), 0);
        assert!(ext.get::<Workspace>().is_none());
        assert!(!ext.contains::<Workspace>());
    }

    #[test]
    fn insert_and_get_round_trip() {
        let ext = Extensions::new().inserted(Workspace("repo-a"));
        assert_eq!(ext.len(), 1);
        let got = ext.get::<Workspace>().unwrap();
        assert_eq!(*got, Workspace("repo-a"));
    }

    #[test]
    fn multiple_distinct_types_coexist() {
        let ext = Extensions::new()
            .inserted(Workspace("repo-a"))
            .inserted(RequestId(42));
        assert_eq!(ext.len(), 2);
        assert_eq!(*ext.get::<Workspace>().unwrap(), Workspace("repo-a"));
        assert_eq!(*ext.get::<RequestId>().unwrap(), RequestId(42));
    }

    #[test]
    fn second_insert_of_same_type_replaces() {
        let ext = Extensions::new()
            .inserted(Workspace("repo-a"))
            .inserted(Workspace("repo-b"));
        assert_eq!(ext.len(), 1, "one entry per type");
        assert_eq!(*ext.get::<Workspace>().unwrap(), Workspace("repo-b"));
    }

    #[test]
    fn copy_on_write_does_not_mutate_original() {
        let original = Extensions::new().inserted(Workspace("repo-a"));
        let extended = original.inserted(RequestId(7));
        // Original unchanged: still 1 entry, no RequestId.
        assert_eq!(original.len(), 1);
        assert!(original.get::<RequestId>().is_none());
        // Extended carries both.
        assert_eq!(extended.len(), 2);
        assert!(extended.get::<RequestId>().is_some());
    }

    #[test]
    fn absent_type_returns_none() {
        let ext = Extensions::new().inserted(Workspace("repo-a"));
        assert!(ext.get::<RequestId>().is_none());
    }

    #[test]
    fn contains_reflects_insertion() {
        let ext = Extensions::new();
        assert!(!ext.contains::<Workspace>());
        let ext = ext.inserted(Workspace("repo-a"));
        assert!(ext.contains::<Workspace>());
    }

    #[test]
    fn debug_surfaces_cardinality() {
        let ext = Extensions::new().inserted(Workspace("repo-a"));
        let debug_str = format!("{ext:?}");
        assert!(debug_str.contains("len: 1"), "{debug_str}");
    }

    #[test]
    fn arc_returned_from_get_outlives_extensions_clone() {
        let ext = Extensions::new().inserted(Workspace("repo-a"));
        let arc = ext.get::<Workspace>().unwrap();
        drop(ext);
        // The Arc is independent; can still be read after the
        // owning Extensions are dropped.
        assert_eq!(*arc, Workspace("repo-a"));
    }
}
