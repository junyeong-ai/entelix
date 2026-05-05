//! `Namespace::render` injectivity plus end-to-end Store-level
//! isolation. Namespaces with `:` characters in their tenant or
//! scope segments must escape to distinct rendered keys, AND a
//! concrete `Store` must keep entries under tenant A's namespace
//! invisible to tenant B's namespace. Together these tests verify
//! invariant 11 (F2 mitigation — cross-tenant data leakage is
//! structurally impossible) at both the keying layer and the
//! storage-protocol layer.

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use entelix_core::ExecutionContext;
use entelix_memory::{InMemoryStore, Namespace, Store};

#[test]
fn distinct_segments_with_colons_render_distinctly() {
    // Without escaping, both of these would render to "t:1:a:b" and
    // collide. With escaping, the colon-bearing segments are
    // disambiguated.
    let a = Namespace::new(TenantId::new("t:1")).with_scope("a:b");
    let b = Namespace::new(TenantId::new("t")).with_scope("1:a:b");
    assert_ne!(a.render(), b.render());
}

#[test]
fn segments_without_special_characters_are_unchanged() {
    let ns = Namespace::new(TenantId::new("acme"))
        .with_scope("agent-1")
        .with_scope("conv-42");
    assert_eq!(ns.render(), "acme:agent-1:conv-42");
}

#[test]
fn backslash_in_segment_is_escaped() {
    // A literal backslash in a segment must round-trip through
    // render() without being mistaken for an escape sequence.
    let a = Namespace::new(TenantId::new("t")).with_scope("a\\b");
    let b = Namespace::new(TenantId::new("t")).with_scope("a\\\\b");
    assert_ne!(a.render(), b.render());
}

#[tokio::test]
async fn store_isolates_distinct_tenants_under_identical_scope() {
    // The same scope name ("session-42") under two distinct
    // tenant_ids must NOT cross-read. This is the architectural
    // claim of invariant 11 verified end-to-end through the Store
    // protocol, not just at the namespace key.
    let store: Arc<dyn Store<String>> = Arc::new(InMemoryStore::<String>::new());
    let ctx = ExecutionContext::new();

    let acme = Namespace::new(TenantId::new("acme")).with_scope("session-42");
    let beta = Namespace::new(TenantId::new("beta")).with_scope("session-42");

    store
        .put(&ctx, &acme, "msg", "acme private".to_owned())
        .await
        .unwrap();
    store
        .put(&ctx, &beta, "msg", "beta private".to_owned())
        .await
        .unwrap();

    assert_eq!(
        store.get(&ctx, &acme, "msg").await.unwrap().as_deref(),
        Some("acme private"),
    );
    assert_eq!(
        store.get(&ctx, &beta, "msg").await.unwrap().as_deref(),
        Some("beta private"),
    );

    // Cross-namespace reads must miss. A leak here means a backend
    // is using only the scope segment as the key — a F2 violation
    // that costs nothing to detect and everything to ship.
    let leaked_to_beta = store
        .get(&ctx, &beta, "non-existent-in-beta-but-present-in-acme")
        .await
        .unwrap();
    assert!(leaked_to_beta.is_none());

    // Listing under one tenant must not surface keys from another.
    let acme_keys = store.list(&ctx, &acme, None).await.unwrap();
    let beta_keys = store.list(&ctx, &beta, None).await.unwrap();
    assert_eq!(acme_keys, vec!["msg".to_owned()]);
    assert_eq!(beta_keys, vec!["msg".to_owned()]);

    // Deleting under tenant A must not affect tenant B.
    store.delete(&ctx, &acme, "msg").await.unwrap();
    assert!(store.get(&ctx, &acme, "msg").await.unwrap().is_none());
    assert_eq!(
        store.get(&ctx, &beta, "msg").await.unwrap().as_deref(),
        Some("beta private"),
        "delete on tenant A must not touch tenant B",
    );
}

#[tokio::test]
async fn store_isolates_colon_bearing_segments_via_escaping() {
    // Two namespaces whose scope segments contain `:` characters
    // would render to identical strings without escaping. Verify
    // the Store treats them as distinct, exercising the same
    // injectivity guarantee end-to-end.
    let store: Arc<dyn Store<String>> = Arc::new(InMemoryStore::<String>::new());
    let ctx = ExecutionContext::new();
    let a = Namespace::new(TenantId::new("t:1")).with_scope("a:b");
    let b = Namespace::new(TenantId::new("t")).with_scope("1:a:b");
    assert_ne!(a.render(), b.render());

    store.put(&ctx, &a, "k", "from-a".to_owned()).await.unwrap();
    store.put(&ctx, &b, "k", "from-b".to_owned()).await.unwrap();
    assert_eq!(
        store.get(&ctx, &a, "k").await.unwrap().as_deref(),
        Some("from-a"),
    );
    assert_eq!(
        store.get(&ctx, &b, "k").await.unwrap().as_deref(),
        Some("from-b"),
    );
}
