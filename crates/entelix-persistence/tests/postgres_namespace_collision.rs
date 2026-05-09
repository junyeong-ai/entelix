//! Postgres-backend mirror of the reference [F2 / invariant 11]
//! tenant-isolation suite at
//! `crates/entelix-memory/tests/namespace_collision.rs`.
//!
//! Invariant 13 declares that every backend (`Store`, `SessionLog`,
//! `Checkpointer`, `DistributedLock`) must run an equivalent suite
//! against a real backend — trusting the rendered namespace key
//! alone is insufficient because a backend that accidentally drops
//! the tenant prefix on a SQL `WHERE` clause would leak across
//! tenants regardless of what the in-memory tests verify.
//!
//! Three Postgres-specific gaps the existing
//! `postgres_integration.rs` does not cover:
//!
//! 1. **Store** — colon-bearing tenant + scope segments. Without
//!    `Namespace::render`'s escape sequence, `("t:1", "a:b")` and
//!    `("t", "1:a:b")` would collide on a `LIKE 't:1:a:b%'` query.
//!    Verifies the Postgres backend keys the table on the rendered
//!    string verbatim, not on a weakly-escaped variant.
//! 2. **SessionLog** — same `thread_id` under two distinct
//!    `tenant_id`s must yield independent event lists. Catches a
//!    backend that keys the table on `thread_id` alone.
//! 3. **DistributedLock** — same logical thread under two tenants
//!    must derive distinct advisory keys. The hash already includes
//!    `tenant`, but verifying end-to-end against Postgres confirms
//!    no truncation drops the tenant component before reaching the
//!    `pg_try_advisory_xact_lock` call.
//!
//! Run with:
//!
//! ```text
//! cargo test -p entelix-persistence --features postgres --test postgres_namespace_collision -- --ignored
//! ```

#![cfg(feature = "postgres")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::doc_markdown
)]

use std::time::Duration;

use entelix_core::{ExecutionContext, TenantId, ThreadKey};
use entelix_graph::{Checkpoint, Checkpointer};
use entelix_memory::{Namespace, Store};
use entelix_persistence::postgres::PostgresPersistence;
use entelix_persistence::{AdvisoryKey, DistributedLock};
use entelix_session::{GraphEvent, SessionLog};
use testcontainers_modules::postgres::Postgres;
use testcontainers_modules::testcontainers::runners::AsyncRunner;

async fn boot_persistence() -> (
    PostgresPersistence,
    testcontainers_modules::testcontainers::ContainerAsync<Postgres>,
) {
    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let url = format!("postgres://postgres:postgres@127.0.0.1:{port}/postgres");
    let pers = PostgresPersistence::builder()
        .with_connection_string(url)
        .connect_and_migrate()
        .await
        .unwrap();
    (pers, container)
}

#[tokio::test]
#[ignore = "requires docker"]
async fn store_isolates_colon_bearing_segments_via_escaping() {
    // Without escaping, both rendered keys would be `t:1:a:b` and the
    // backend would conflate two distinct namespaces. The
    // `Namespace::render` escape contract has to survive the trip
    // through SQL — a backend doing a substring or LIKE query would
    // leak.
    let (pers, _container) = boot_persistence().await;
    let store = pers.store::<String>();
    let ctx = ExecutionContext::new();

    let a = Namespace::new(TenantId::new("t:1")).with_scope("a:b");
    let b = Namespace::new(TenantId::new("t")).with_scope("1:a:b");
    assert_ne!(
        a.render(),
        b.render(),
        "render must escape colon-bearing segments"
    );

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

    // List under `a` must not surface the `b` entry even though the
    // unescaped strings would prefix-match.
    let keys_a = store.list(&ctx, &a, None).await.unwrap();
    let keys_b = store.list(&ctx, &b, None).await.unwrap();
    assert_eq!(keys_a, vec!["k".to_owned()]);
    assert_eq!(keys_b, vec!["k".to_owned()]);

    // Delete under `a` must not affect `b`.
    store.delete(&ctx, &a, "k").await.unwrap();
    assert!(store.get(&ctx, &a, "k").await.unwrap().is_none());
    assert_eq!(
        store.get(&ctx, &b, "k").await.unwrap().as_deref(),
        Some("from-b"),
        "delete on namespace `a` must leave `b` untouched"
    );
}

#[tokio::test]
#[ignore = "requires docker"]
async fn session_log_isolates_distinct_tenants_under_identical_thread_id() {
    // Same thread_id under two distinct tenants. A backend keying the
    // table on `thread_id` alone (or on a tenant-stripped composite)
    // would reveal tenant A's audit log to tenant B.
    let (pers, _container) = boot_persistence().await;
    let log = pers.session_log();

    let acme_key = ThreadKey::new(TenantId::new("acme"), "shared-thread-name");
    let beta_key = ThreadKey::new(TenantId::new("beta"), "shared-thread-name");

    let acme_event = vec![GraphEvent::UserMessage {
        content: vec![entelix_core::ir::ContentPart::text("acme private")],
        timestamp: chrono::Utc::now(),
    }];
    let beta_event = vec![GraphEvent::UserMessage {
        content: vec![entelix_core::ir::ContentPart::text("beta private")],
        timestamp: chrono::Utc::now(),
    }];

    log.append(&acme_key, &acme_event).await.unwrap();
    log.append(&beta_key, &beta_event).await.unwrap();

    let acme_loaded = log.load_since(&acme_key, 0).await.unwrap();
    let beta_loaded = log.load_since(&beta_key, 0).await.unwrap();
    assert_eq!(acme_loaded.len(), 1);
    assert_eq!(beta_loaded.len(), 1);

    // Sanity check the contents — easy to fake equal-length vectors
    // by accident if both ended up reading the same row.
    let acme_text = match &acme_loaded[0] {
        GraphEvent::UserMessage { content, .. } => content
            .iter()
            .filter_map(|p| match p {
                entelix_core::ir::ContentPart::Text { text, .. } => Some(text.clone()),
                _ => None,
            })
            .collect::<String>(),
        other => panic!("expected UserMessage, got {other:?}"),
    };
    let beta_text = match &beta_loaded[0] {
        GraphEvent::UserMessage { content, .. } => content
            .iter()
            .filter_map(|p| match p {
                entelix_core::ir::ContentPart::Text { text, .. } => Some(text.clone()),
                _ => None,
            })
            .collect::<String>(),
        other => panic!("expected UserMessage, got {other:?}"),
    };
    assert_eq!(acme_text, "acme private");
    assert_eq!(beta_text, "beta private");

    // Archive on tenant `acme` must not affect tenant `beta` — a
    // misimplemented backend that ARCHIVED-by-thread_id would purge
    // beta's row when acme called archive_before.
    let archived = log.archive_before(&acme_key, 100).await.unwrap();
    assert_eq!(archived, 1, "acme archive should remove acme's lone event");
    let beta_post = log.load_since(&beta_key, 0).await.unwrap();
    assert_eq!(
        beta_post.len(),
        1,
        "archive on tenant acme must not touch tenant beta"
    );
}

#[tokio::test]
#[ignore = "requires docker"]
async fn distributed_lock_isolates_distinct_tenants_under_identical_thread_id() {
    // `AdvisoryKey::for_session` includes the tenant in the hash, so
    // same thread_id under two tenants must derive distinct keys and
    // therefore non-blocking lock acquisition.
    let (pers, _container) = boot_persistence().await;
    let lock = pers.lock();

    let acme = AdvisoryKey::for_session("acme", "shared-thread-name");
    let beta = AdvisoryKey::for_session("beta", "shared-thread-name");
    assert_ne!(
        acme.raw(),
        beta.raw(),
        "advisory key derivation must include tenant"
    );

    let g_acme = lock
        .try_acquire(&acme, Duration::from_secs(5))
        .await
        .unwrap()
        .expect("acme acquire must succeed");
    let g_beta = lock
        .try_acquire(&beta, Duration::from_secs(5))
        .await
        .unwrap()
        .expect("beta acquire must succeed even while acme holds its own");

    // Tenant beta cannot acquire acme's key while acme holds it —
    // that's the lock semantic. The cross-tenant property is that
    // beta's OWN key remains acquirable.
    let g_acme_again = lock
        .try_acquire(&acme, Duration::from_secs(5))
        .await
        .unwrap();
    assert!(
        g_acme_again.is_none(),
        "second acme acquire must fail while first is held"
    );

    lock.release(g_acme).await.unwrap();
    lock.release(g_beta).await.unwrap();
}

#[tokio::test]
#[ignore = "requires docker"]
async fn checkpointer_isolates_state_across_tenants_at_same_thread_id() {
    // Same thread_id under two tenants. The checkpointer must
    // segregate by `(tenant_id, thread_id)` — `latest`/`history`/
    // `by_id` against tenant A must never surface tenant B's row.
    // A backend keying on `thread_id` alone (or omitting tenant
    // from a composite WHERE clause) would conflate the two.
    let (pers, _container) = boot_persistence().await;
    let cp = pers.checkpointer::<u64>();

    let key_a = ThreadKey::new(TenantId::new("tenant-a"), "shared-thread");
    let key_b = ThreadKey::new(TenantId::new("tenant-b"), "shared-thread");
    let cp_a = Checkpoint::new(&key_a, 0, 100u64, Some("next".into()));
    let cp_b = Checkpoint::new(&key_b, 0, 200u64, Some("next".into()));
    let id_a = cp_a.id.clone();
    let id_b = cp_b.id.clone();
    cp.put(cp_a).await.unwrap();
    cp.put(cp_b).await.unwrap();

    let latest_a = cp.get_latest(&key_a).await.unwrap().unwrap();
    let latest_b = cp.get_latest(&key_b).await.unwrap().unwrap();
    assert_eq!(
        latest_a.state, 100,
        "tenant A latest must be tenant A's state"
    );
    assert_eq!(
        latest_b.state, 200,
        "tenant B latest must be tenant B's state"
    );
    assert_eq!(latest_a.id, id_a);
    assert_eq!(latest_b.id, id_b);

    // by_id with the wrong tenant must miss — the checkpoint id is
    // not a global namespace, it is scoped to (tenant, thread).
    assert!(
        cp.get_by_id(&key_a, &id_b).await.unwrap().is_none(),
        "tenant A must NOT find tenant B's checkpoint id"
    );
    assert!(
        cp.get_by_id(&key_b, &id_a).await.unwrap().is_none(),
        "tenant B must NOT find tenant A's checkpoint id"
    );

    let history_a = cp.list_history(&key_a, 10).await.unwrap();
    let history_b = cp.list_history(&key_b, 10).await.unwrap();
    assert_eq!(history_a.len(), 1);
    assert_eq!(history_b.len(), 1);
    assert_eq!(history_a[0].state, 100);
    assert_eq!(history_b[0].state, 200);
}
