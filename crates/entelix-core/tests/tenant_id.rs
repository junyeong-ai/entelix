//! `ExecutionContext::tenant_id` defaults + override (invariant 11
//! strengthening per ADR-0017).

use entelix_core::{DEFAULT_TENANT_ID, ExecutionContext};

#[test]
fn new_context_has_default_tenant() {
    let ctx = ExecutionContext::new();
    assert_eq!(ctx.tenant_id(), DEFAULT_TENANT_ID);
    assert_eq!(ctx.tenant_id(), "default");
}

#[test]
fn default_impl_matches_new() {
    let a = ExecutionContext::default();
    let b = ExecutionContext::new();
    assert_eq!(a.tenant_id(), b.tenant_id());
}

#[test]
fn with_tenant_id_overrides() {
    let ctx = ExecutionContext::new().with_tenant_id("acme-corp");
    assert_eq!(ctx.tenant_id(), "acme-corp");
}

#[test]
fn tenant_id_propagates_through_clone() {
    let ctx = ExecutionContext::new().with_tenant_id("alpha");
    #[allow(clippy::redundant_clone)]
    let cloned = ctx.clone();
    assert_eq!(cloned.tenant_id(), "alpha");
}

#[test]
fn tenant_id_independent_of_thread_id() {
    let ctx = ExecutionContext::new()
        .with_tenant_id("tenant-x")
        .with_thread_id("thread-y");
    assert_eq!(ctx.tenant_id(), "tenant-x");
    assert_eq!(ctx.thread_id(), Some("thread-y"));
}

#[test]
fn cancellation_inheritance_preserves_tenant() {
    let parent = ExecutionContext::new().with_tenant_id("parent-tenant");
    let cancellation = parent.cancellation().clone();
    let child = ExecutionContext::with_cancellation(cancellation);
    // Child gets a fresh default tenant — explicit propagation is the
    // caller's responsibility (multi-tenant requests should never silently
    // borrow a parent's scope).
    assert_eq!(child.tenant_id(), "default");
    assert_eq!(parent.tenant_id(), "parent-tenant");
}
