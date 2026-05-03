//! Postgres row-level-security regression — proves the
//! `tenant_isolation` policy installed by migration v2 (ADR-0041)
//! actually rejects cross-tenant access at the database layer.
//!
//! The default test container's `postgres` role is `SUPERUSER`,
//! which *bypasses* every RLS policy regardless of `FORCE` settings.
//! To exercise the gate we create a second `NOSUPERUSER`,
//! `NOBYPASSRLS` role inside the container, grant it CRUD on the
//! tenant-scoped tables, and connect a second pool as that role.
//! Queries issued through that pool obey the policy.
//!
//! Run with:
//!
//! ```text
//! cargo test -p entelix-persistence --features postgres --test postgres_rls -- --ignored
//! ```

#![cfg(feature = "postgres")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::doc_markdown
)]

use entelix_core::ExecutionContext;
use entelix_memory::{Namespace, Store};
use entelix_persistence::postgres::PostgresPersistence;
use sqlx::Executor;
use sqlx::postgres::{PgPool, PgPoolOptions};
use testcontainers_modules::postgres::Postgres;
use testcontainers_modules::testcontainers::ContainerAsync;
use testcontainers_modules::testcontainers::runners::AsyncRunner;

const APP_ROLE: &str = "entelix_app";
const APP_PASSWORD: &str = "apppwd";

/// Boot postgres, run the SDK migrations as superuser, then mint a
/// second `NOSUPERUSER NOBYPASSRLS` role and return both pools.
/// Returns `(superuser_pers, app_pers, container)`.
async fn boot_with_app_role() -> (
    PostgresPersistence,
    PostgresPersistence,
    ContainerAsync<Postgres>,
) {
    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let super_url = format!("postgres://postgres:postgres@127.0.0.1:{port}/postgres");

    // Migrations + role setup run as the superuser pool.
    let super_pers = PostgresPersistence::builder()
        .with_connection_string(super_url)
        .connect_and_migrate()
        .await
        .unwrap();
    let super_pool = super_pers.pool();
    super_pool
        .execute(
            format!(
                "CREATE ROLE {APP_ROLE} WITH LOGIN PASSWORD '{APP_PASSWORD}' \
                 NOSUPERUSER NOBYPASSRLS"
            )
            .as_str(),
        )
        .await
        .unwrap();
    super_pool
        .execute(
            format!(
                "GRANT SELECT, INSERT, UPDATE, DELETE ON \
                 memory_items, session_events, checkpoints TO {APP_ROLE}"
            )
            .as_str(),
        )
        .await
        .unwrap();
    super_pool
        .execute(format!("GRANT USAGE ON SCHEMA public TO {APP_ROLE}").as_str())
        .await
        .unwrap();

    let app_url = format!("postgres://{APP_ROLE}:{APP_PASSWORD}@127.0.0.1:{port}/postgres");
    let app_pers = PostgresPersistence::builder()
        .with_connection_string(app_url)
        .connect()
        .await
        .unwrap();

    (super_pers, app_pers, container)
}

#[tokio::test]
#[ignore = "requires docker"]
async fn rls_blocks_cross_tenant_reads_at_db_layer() {
    let (_super_pers, app_pers, _container) = boot_with_app_role().await;
    let app_store = app_pers.store::<String>();
    let ctx = ExecutionContext::new();

    // SDK path stamps `entelix.tenant_id` per transaction — the
    // policy lets the row in.
    let ns = Namespace::new("tenant-A").with_scope("scope");
    app_store.put(&ctx, &ns, "k", "v".into()).await.unwrap();
    assert_eq!(
        app_store.get(&ctx, &ns, "k").await.unwrap().as_deref(),
        Some("v"),
        "SDK read of own-tenant row must succeed"
    );

    // Bypass the SDK — raw read with no `entelix.tenant_id` set.
    // The policy treats `tenant_id = NULL` as false, hiding every row.
    let app_pool = app_pers.pool();
    let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM memory_items")
        .fetch_one(app_pool)
        .await
        .unwrap();
    assert_eq!(
        count.0, 0,
        "RLS must hide rows when entelix.tenant_id is unset (defense vs forgotten SET LOCAL)"
    );

    // Set the wrong tenant. The policy filters out the row even
    // though it is physically present.
    assert_eq!(count_in_tx(app_pool, "tenant-B").await, 0);
    // Set the right tenant. Row visible.
    assert_eq!(count_in_tx(app_pool, "tenant-A").await, 1);
}

#[tokio::test]
#[ignore = "requires docker"]
async fn rls_with_check_blocks_cross_tenant_writes() {
    let (_super_pers, app_pers, _container) = boot_with_app_role().await;
    let app_pool = app_pers.pool();

    // Set tenant-A, try to INSERT a row with tenant_id = tenant-B.
    // The WITH CHECK clause rejects the write.
    let mut tx = app_pool.begin().await.unwrap();
    sqlx::query("SELECT set_config('entelix.tenant_id', 'tenant-A', true)")
        .execute(&mut *tx)
        .await
        .unwrap();
    let result = sqlx::query(
        r#"
        INSERT INTO memory_items (tenant_id, namespace, key, value)
        VALUES ('tenant-B', 'tenant-B:scope', 'k', '"v"')
        "#,
    )
    .execute(&mut *tx)
    .await;
    assert!(
        result.is_err(),
        "WITH CHECK must reject INSERT whose tenant_id differs from session var"
    );
}

#[tokio::test]
#[ignore = "requires docker"]
async fn rls_applies_to_session_events_and_checkpoints() {
    // Smoke-check the same gate is wired for the other two
    // tenant-scoped tables. SDK paths set the variable, so writes
    // succeed; raw reads without the variable surface 0 rows.
    use entelix_core::ThreadKey;
    use entelix_graph::{Checkpoint, Checkpointer};
    use entelix_session::{GraphEvent, SessionLog};

    let (_super_pers, app_pers, _container) = boot_with_app_role().await;
    let log = app_pers.session_log();
    let cp = app_pers.checkpointer::<i32>();
    let key = ThreadKey::new("tenant-A", "thread-1");

    log.append(
        &key,
        &[GraphEvent::UserMessage {
            content: vec![entelix_core::ir::ContentPart::text("hi")],
            timestamp: chrono::Utc::now(),
        }],
    )
    .await
    .unwrap();
    cp.put(Checkpoint::new(&key, 0, 42i32, None)).await.unwrap();

    let app_pool = app_pers.pool();
    let count_events: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM session_events")
        .fetch_one(app_pool)
        .await
        .unwrap();
    let count_checkpoints: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM checkpoints")
        .fetch_one(app_pool)
        .await
        .unwrap();
    assert_eq!(
        count_events.0, 0,
        "session_events RLS must hide rows when tenant unset"
    );
    assert_eq!(
        count_checkpoints.0, 0,
        "checkpoints RLS must hide rows when tenant unset"
    );
}

async fn count_in_tx(pool: &PgPool, tenant: &str) -> i64 {
    let mut tx = pool.begin().await.unwrap();
    sqlx::query("SELECT set_config('entelix.tenant_id', $1, true)")
        .bind(tenant)
        .execute(&mut *tx)
        .await
        .unwrap();
    let row: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM memory_items")
        .fetch_one(&mut *tx)
        .await
        .unwrap();
    tx.commit().await.unwrap();
    row.0
}

// Suppress unused-import warning when sqlx options module isn't
// referenced directly (PoolOptions reserved for future tuning).
#[allow(dead_code)]
fn _silence(_o: PgPoolOptions) {}
