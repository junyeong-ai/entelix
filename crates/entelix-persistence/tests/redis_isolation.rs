//! Redis row-level isolation suite — Invariant 13 enforcement.
//!
//! Postgres backends ride on row-level WHERE clauses; Redis enforces
//! the same boundary by encoding `tenant_id` into every key. The
//! tests below exercise the actual key paths against a live Redis
//! testcontainer to prove that:
//!
//! - `Store::get/list/delete` against tenant A cannot read writes
//!   from tenant B at the same logical key,
//! - `Checkpointer::latest/history/by_id` against tenant A cannot
//!   surface checkpoints written under tenant B,
//! - `SessionLog::load_since/archive_before` against tenant A cannot
//!   touch tenant B's event stream,
//! - the `DistributedLock` namespace separates by `tenant_id`, so a
//!   lock held in tenant A does not block tenant B's identical
//!   `thread_id`.
//!
//! Run with:
//!
//! ```text
//! cargo test -p entelix-persistence --features redis --test redis_isolation -- --ignored
//! ```

#![cfg(feature = "redis")]
#![allow(
    clippy::unwrap_used,
    clippy::needless_pass_by_value,
    clippy::indexing_slicing,
    clippy::expect_used
)]

use std::time::Duration;

use entelix_core::ExecutionContext;
use entelix_core::ThreadKey;
use entelix_graph::{Checkpoint, Checkpointer};
use entelix_memory::{Namespace, Store};
use entelix_persistence::redis::RedisPersistence;
use entelix_persistence::{AdvisoryKey, DistributedLock};
use entelix_session::{GraphEvent, SessionLog};
use testcontainers_modules::redis::{REDIS_PORT, Redis};
use testcontainers_modules::testcontainers::runners::AsyncRunner;

async fn boot_persistence() -> (
    RedisPersistence,
    testcontainers_modules::testcontainers::ContainerAsync<Redis>,
) {
    let container = Redis::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(REDIS_PORT).await.unwrap();
    let url = format!("redis://127.0.0.1:{port}/");
    let pers = RedisPersistence::builder()
        .with_connection_string(url)
        .connect()
        .await
        .unwrap();
    (pers, container)
}

#[tokio::test]
#[ignore = "requires docker"]
async fn store_writes_under_one_tenant_invisible_to_other_tenants() {
    let (pers, _container) = boot_persistence().await;
    let store = pers.store::<String>();
    let ctx = ExecutionContext::new();
    let ns_a = Namespace::new("tenant-a").with_scope("scope1");
    let ns_b = Namespace::new("tenant-b").with_scope("scope1");

    store
        .put(&ctx, &ns_a, "shared-key", "value-a".to_owned())
        .await
        .unwrap();
    store
        .put(&ctx, &ns_b, "shared-key", "value-b".to_owned())
        .await
        .unwrap();

    // Reading the same logical key from each tenant returns each
    // tenant's own write — no cross-tenant bleed.
    assert_eq!(
        store
            .get(&ctx, &ns_a, "shared-key")
            .await
            .unwrap()
            .as_deref(),
        Some("value-a"),
        "tenant A must read its own value, not tenant B's"
    );
    assert_eq!(
        store
            .get(&ctx, &ns_b, "shared-key")
            .await
            .unwrap()
            .as_deref(),
        Some("value-b"),
        "tenant B must read its own value, not tenant A's"
    );

    // Listing under each tenant returns only that tenant's keys.
    let a_keys = store.list(&ctx, &ns_a, None).await.unwrap();
    let b_keys = store.list(&ctx, &ns_b, None).await.unwrap();
    assert_eq!(a_keys, vec!["shared-key".to_owned()]);
    assert_eq!(b_keys, vec!["shared-key".to_owned()]);

    // Deleting tenant A's key MUST NOT touch tenant B.
    store.delete(&ctx, &ns_a, "shared-key").await.unwrap();
    assert!(
        store
            .get(&ctx, &ns_a, "shared-key")
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(
        store
            .get(&ctx, &ns_b, "shared-key")
            .await
            .unwrap()
            .as_deref(),
        Some("value-b"),
        "tenant B's value must survive tenant A's delete"
    );
}

#[tokio::test]
#[ignore = "requires docker"]
async fn checkpointer_isolates_state_across_tenants_at_same_thread_id() {
    let (pers, _container) = boot_persistence().await;
    let cp = pers.checkpointer::<u64>();

    let key_a = ThreadKey::new("tenant-a", "shared-thread");
    let key_b = ThreadKey::new("tenant-b", "shared-thread");
    let cp_a = Checkpoint::new(&key_a, 0, 100u64, Some("next".into()));
    let cp_b = Checkpoint::new(&key_b, 0, 200u64, Some("next".into()));
    let id_a = cp_a.id.clone();
    let id_b = cp_b.id.clone();
    cp.put(cp_a).await.unwrap();
    cp.put(cp_b).await.unwrap();

    let latest_a = cp.latest(&key_a).await.unwrap().unwrap();
    let latest_b = cp.latest(&key_b).await.unwrap().unwrap();
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

    // by_id with the wrong tenant must miss — id is not a global
    // namespace, it is scoped to (tenant, thread).
    assert!(
        cp.by_id(&key_a, &id_b).await.unwrap().is_none(),
        "tenant A must NOT find tenant B's checkpoint id"
    );
    assert!(
        cp.by_id(&key_b, &id_a).await.unwrap().is_none(),
        "tenant B must NOT find tenant A's checkpoint id"
    );

    let history_a = cp.history(&key_a, 10).await.unwrap();
    let history_b = cp.history(&key_b, 10).await.unwrap();
    assert_eq!(history_a.len(), 1);
    assert_eq!(history_b.len(), 1);
    assert_eq!(history_a[0].state, 100);
    assert_eq!(history_b[0].state, 200);
}

#[tokio::test]
#[ignore = "requires docker"]
async fn session_log_event_streams_do_not_leak_across_tenants() {
    let (pers, _container) = boot_persistence().await;
    let log = pers.session_log();

    let evt_a = vec![GraphEvent::UserMessage {
        content: vec![entelix_core::ir::ContentPart::text("from-a")],
        timestamp: chrono::Utc::now(),
    }];
    let evt_b = vec![GraphEvent::UserMessage {
        content: vec![entelix_core::ir::ContentPart::text("from-b")],
        timestamp: chrono::Utc::now(),
    }];

    let key_a = ThreadKey::new("tenant-a", "shared-thread");
    let key_b = ThreadKey::new("tenant-b", "shared-thread");
    log.append(&key_a, &evt_a).await.unwrap();
    log.append(&key_b, &evt_b).await.unwrap();

    let loaded_a = log.load_since(&key_a, 0).await.unwrap();
    let loaded_b = log.load_since(&key_b, 0).await.unwrap();
    assert_eq!(loaded_a.len(), 1);
    assert_eq!(loaded_b.len(), 1);
    let a_text = match &loaded_a[0] {
        GraphEvent::UserMessage { content, .. } => match &content[0] {
            entelix_core::ir::ContentPart::Text { text, .. } => text.as_str(),
            _ => panic!("expected Text part"),
        },
        _ => panic!("expected UserMessage"),
    };
    let b_text = match &loaded_b[0] {
        GraphEvent::UserMessage { content, .. } => match &content[0] {
            entelix_core::ir::ContentPart::Text { text, .. } => text.as_str(),
            _ => panic!("expected Text part"),
        },
        _ => panic!("expected UserMessage"),
    };
    assert_eq!(a_text, "from-a");
    assert_eq!(b_text, "from-b");

    // Archiving tenant A's stream must NOT touch tenant B.
    let archived = log.archive_before(&key_a, u64::MAX).await.unwrap();
    assert!(archived >= 1);
    let post_archive_a = log.load_since(&key_a, 0).await.unwrap();
    let post_archive_b = log.load_since(&key_b, 0).await.unwrap();
    assert!(
        post_archive_a.is_empty(),
        "tenant A archive must clear tenant A's stream"
    );
    assert_eq!(
        post_archive_b.len(),
        1,
        "tenant A archive must NOT touch tenant B's stream"
    );
}

#[tokio::test]
#[ignore = "requires docker"]
async fn distributed_lock_namespaces_by_tenant_id() {
    let (pers, _container) = boot_persistence().await;
    let lock = pers.lock();

    let key_a = AdvisoryKey::for_session("tenant-a", "shared-thread");
    let key_b = AdvisoryKey::for_session("tenant-b", "shared-thread");

    let g_a = lock
        .try_acquire(&key_a, Duration::from_secs(5))
        .await
        .unwrap()
        .expect("tenant A acquire must succeed");
    // Tenant B's key collides on thread_id but must NOT collide
    // on the lock key — Invariant 13 demands the lock space is
    // partitioned by tenant.
    let g_b = lock
        .try_acquire(&key_b, Duration::from_secs(5))
        .await
        .unwrap()
        .expect("tenant B acquire must succeed despite shared thread_id");

    lock.release(g_a).await.unwrap();
    lock.release(g_b).await.unwrap();
}

#[tokio::test]
#[ignore = "requires docker"]
async fn store_isolates_colon_bearing_segments_via_escaping() {
    // Without escaping, both rendered keys would be `t:1:a:b` and the
    // backend would conflate two distinct namespaces. Redis encodes
    // `tenant_id` into every key — the `Namespace::render` escape
    // contract has to survive the trip into the keyspace, otherwise
    // an unescaped `:` could glue tenant + scope segments into a
    // colliding key. Mirrors the Postgres test of the same name.
    let (pers, _container) = boot_persistence().await;
    let store = pers.store::<String>();
    let ctx = ExecutionContext::new();

    let a = Namespace::new("t:1").with_scope("a:b");
    let b = Namespace::new("t").with_scope("1:a:b");
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

    // List under `a` must not surface `b` even though the unescaped
    // strings would prefix-match a SCAN/KEYS pattern.
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
