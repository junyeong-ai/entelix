//! End-to-end Postgres integration tests using `testcontainers`.
//! Requires a working docker daemon. Run with:
//!
//! ```text
//! cargo test -p entelix-persistence --features postgres --test postgres_integration -- --ignored
//! ```

#![cfg(feature = "postgres")]
#![allow(clippy::unwrap_used, clippy::needless_pass_by_value)]

use entelix_core::TenantId;
use std::time::Duration;

use entelix_core::ExecutionContext;
use entelix_core::ThreadKey;
use entelix_graph::{Checkpoint, Checkpointer};
use entelix_memory::{Namespace, NamespacePrefix, Store};
use entelix_persistence::postgres::PostgresPersistence;
use entelix_persistence::{AdvisoryKey, DistributedLock, with_session_lock};
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
async fn checkpointer_round_trip() {
    let (pers, _container) = boot_persistence().await;
    let cp = pers.checkpointer::<u32>();

    let key = ThreadKey::new(TenantId::new("default"), "t1");
    let checkpoint = Checkpoint::new(&key, 0, 42u32, Some("next".into()));
    let id = checkpoint.id.clone();
    cp.put(checkpoint).await.unwrap();
    let loaded = cp.latest(&key).await.unwrap().unwrap();
    assert_eq!(loaded.state, 42);
    assert_eq!(loaded.id, id);

    let by_id = cp.by_id(&key, &id).await.unwrap().unwrap();
    assert_eq!(by_id.id, id);

    let history = cp.history(&key, 10).await.unwrap();
    assert_eq!(history.len(), 1);

    // Tenant isolation: different tenant_id must return None / empty.
    let other_key = ThreadKey::new(TenantId::new("other"), "t1");
    assert!(cp.latest(&other_key).await.unwrap().is_none());
    assert!(cp.history(&other_key, 10).await.unwrap().is_empty());
}

#[tokio::test]
#[ignore = "requires docker"]
async fn store_round_trip_with_namespace_isolation() {
    let (pers, _container) = boot_persistence().await;
    let store = pers.store::<String>();

    let ns_a = Namespace::new(TenantId::new("tenant-a")).with_scope("scope1");
    let ns_b = Namespace::new(TenantId::new("tenant-b")).with_scope("scope1");
    let ctx = ExecutionContext::new();

    store
        .put(&ctx, &ns_a, "k", "value-a".to_owned())
        .await
        .unwrap();
    store
        .put(&ctx, &ns_b, "k", "value-b".to_owned())
        .await
        .unwrap();

    assert_eq!(
        store.get(&ctx, &ns_a, "k").await.unwrap().as_deref(),
        Some("value-a")
    );
    assert_eq!(
        store.get(&ctx, &ns_b, "k").await.unwrap().as_deref(),
        Some("value-b")
    );

    // Cross-tenant list isolation.
    let keys_a = store.list(&ctx, &ns_a, None).await.unwrap();
    assert_eq!(keys_a, vec!["k".to_owned()]);

    store.delete(&ctx, &ns_a, "k").await.unwrap();
    assert!(store.get(&ctx, &ns_a, "k").await.unwrap().is_none());
    // Tenant B unaffected.
    assert_eq!(
        store.get(&ctx, &ns_b, "k").await.unwrap().as_deref(),
        Some("value-b")
    );
}

#[tokio::test]
#[ignore = "requires docker"]
async fn list_namespaces_returns_typed_scopes_round_tripped_through_render() {
    let (pers, _container) = boot_persistence().await;
    let store = pers.store::<String>();
    let ctx = ExecutionContext::new();

    // Three distinct namespaces under the same agent prefix, plus
    // one sibling under a different agent. The `:`-bearing scope
    // segment exercises the escape round-trip through SQL storage.
    let ns_a = Namespace::new(TenantId::new("acme")).with_scope("agent-a");
    let ns_b = Namespace::new(TenantId::new("acme"))
        .with_scope("agent-a")
        .with_scope("conv-1");
    let ns_c = Namespace::new(TenantId::new("acme"))
        .with_scope("agent-a")
        .with_scope("k8s:pod:foo");
    let ns_other = Namespace::new(TenantId::new("acme")).with_scope("agent-b");
    for ns in [&ns_a, &ns_b, &ns_c, &ns_other] {
        store.put(&ctx, ns, "k", "v".into()).await.unwrap();
    }

    let prefix = NamespacePrefix::new(TenantId::new("acme")).with_scope("agent-a");
    let mut found = store.list_namespaces(&ctx, &prefix).await.unwrap();
    assert_eq!(found.len(), 3);
    found.sort_by_key(Namespace::render);
    let mut expected: Vec<Namespace> = vec![ns_a.clone(), ns_b.clone(), ns_c.clone()];
    expected.sort_by_key(Namespace::render);
    // Each returned namespace must match the originally stored
    // structural value, not a prefix-shape clone.
    assert_eq!(found, expected);
}

#[tokio::test]
#[ignore = "requires docker"]
async fn session_log_append_load_archive() {
    let (pers, _container) = boot_persistence().await;
    let log = pers.session_log();

    let events = vec![GraphEvent::UserMessage {
        content: vec![entelix_core::ir::ContentPart::text("first")],
        timestamp: chrono::Utc::now(),
    }];
    let key = ThreadKey::new(TenantId::new("tenant-x"), "thread-1");
    let head = log.append(&key, &events).await.unwrap();
    assert_eq!(head, 1);

    let loaded = log.load_since(&key, 0).await.unwrap();
    assert_eq!(loaded.len(), 1);

    let archived = log.archive_before(&key, 1).await.unwrap();
    assert_eq!(archived, 1);

    let post_archive = log.load_since(&key, 0).await.unwrap();
    assert_eq!(post_archive.len(), 0);
}

#[tokio::test]
#[ignore = "requires docker"]
async fn distributed_lock_acquire_release() {
    let (pers, _container) = boot_persistence().await;
    let lock = pers.lock();
    let key = AdvisoryKey::for_session("tenant-x", "thread-7");

    let g1 = lock
        .try_acquire(&key, Duration::from_secs(5))
        .await
        .unwrap()
        .unwrap();
    let g2 = lock
        .try_acquire(&key, Duration::from_secs(5))
        .await
        .unwrap();
    assert!(g2.is_none(), "second acquire must fail while first is held");
    lock.release(g1).await.unwrap();
    let g3 = lock
        .try_acquire(&key, Duration::from_secs(5))
        .await
        .unwrap();
    assert!(g3.is_some(), "acquire must succeed once released");
    lock.release(g3.unwrap()).await.unwrap();
}

#[tokio::test]
#[ignore = "requires docker"]
async fn with_session_lock_via_postgres() {
    let (pers, _container) = boot_persistence().await;
    let lock = pers.lock();
    let result: entelix_persistence::PersistenceResult<u32> = with_session_lock(
        &lock,
        "tenant-x",
        "thread-7",
        Some(Duration::from_secs(5)),
        Some(Duration::from_secs(5)),
        || async move { Ok::<u32, entelix_persistence::PersistenceError>(7) },
    )
    .await;
    assert_eq!(result.unwrap(), 7);
}
