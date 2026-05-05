//! Row-level-security regression for `entelix-graphmemory-pg` —
//! mirror of `entelix-persistence/tests/postgres_rls.rs`. Proves
//! the `tenant_isolation` policy installed by the bootstrap
//! actually rejects cross-tenant access at the database layer.
//!
//! The default `postgres` role is `SUPERUSER` and bypasses every
//! RLS policy regardless of `FORCE`. To exercise the gate this
//! suite mints a second `NOSUPERUSER NOBYPASSRLS` role inside
//! the container, grants it CRUD on `graph_nodes` /
//! `graph_edges`, and connects a second pool through it.
//!
//! Run with:
//!
//! ```text
//! cargo test -p entelix-graphmemory-pg --test postgres_rls -- --ignored
//! ```

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::doc_markdown
)]

use entelix_core::TenantId;
use std::sync::Arc;

use chrono::Utc;
use entelix_core::ExecutionContext;
use entelix_graphmemory_pg::PgGraphMemory;
use entelix_memory::{GraphMemory, Namespace};
use sqlx::Executor;
use sqlx::postgres::{PgPool, PgPoolOptions};
use testcontainers_modules::postgres::Postgres;
use testcontainers_modules::testcontainers::ContainerAsync;
use testcontainers_modules::testcontainers::runners::AsyncRunner;

const APP_ROLE: &str = "graph_app";
const APP_PASSWORD: &str = "apppwd";

/// Boot postgres, run the bootstrap as superuser (which both
/// creates the schema and installs RLS policies), then mint a
/// second NOSUPERUSER NOBYPASSRLS role and return both backends.
async fn boot_with_app_role() -> (
    PgGraphMemory<String, String>,
    PgGraphMemory<String, String>,
    ContainerAsync<Postgres>,
) {
    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let super_url = format!("postgres://postgres:postgres@127.0.0.1:{port}/postgres");

    let super_pool = Arc::new(PgPoolOptions::new().connect(&super_url).await.unwrap());
    // Build once as superuser to run the bootstrap (schema +
    // ENABLE/FORCE RLS + policy). Subsequent app-role queries
    // hit the same schema.
    let super_graph = PgGraphMemory::<String, String>::builder()
        .with_pool(Arc::clone(&super_pool))
        .build()
        .await
        .unwrap();

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
                 graph_nodes, graph_edges TO {APP_ROLE}"
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
    let app_pool = Arc::new(PgPoolOptions::new().connect(&app_url).await.unwrap());
    let app_graph = PgGraphMemory::<String, String>::builder()
        .with_pool(Arc::clone(&app_pool))
        // Skip auto_migrate — the superuser pool already ran it,
        // and the app role lacks DDL privileges.
        .with_auto_migrate(false)
        .build()
        .await
        .unwrap();

    (super_graph, app_graph, container)
}

#[tokio::test]
#[ignore = "requires docker"]
async fn rls_blocks_cross_tenant_node_lookup_at_db_layer() {
    let (_super_graph, app_graph, container) = boot_with_app_role().await;
    let ctx = ExecutionContext::new();

    // SDK path stamps `entelix.tenant_id` per transaction — write
    // through the app role succeeds because (a) tenant_id matches
    // the session var (WITH CHECK), and (b) reads through SDK use
    // the same matching var (USING).
    let ns_a = Namespace::new(TenantId::new("tenant-A")).with_scope("scope");
    let id = app_graph
        .add_node(&ctx, &ns_a, "alice".into())
        .await
        .unwrap();
    assert_eq!(
        app_graph.node(&ctx, &ns_a, &id).await.unwrap().as_deref(),
        Some("alice"),
    );

    // Bypass the SDK — raw read with no `entelix.tenant_id` set.
    // Policy treats `tenant_id = NULL` as false, hides every row.
    let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM graph_nodes")
        .fetch_one(&*pool_for(APP_ROLE, APP_PASSWORD, &container).await)
        .await
        .unwrap();
    assert_eq!(
        count.0, 0,
        "RLS must hide rows when entelix.tenant_id is unset (defense vs forgotten SET LOCAL)"
    );
}

#[tokio::test]
#[ignore = "requires docker"]
async fn rls_with_check_blocks_mismatched_tenant_inserts() {
    let (_super_graph, _app_graph, container) = boot_with_app_role().await;
    let app_pool = pool_for(APP_ROLE, APP_PASSWORD, &container).await;

    // Set tenant-A in the session, try to INSERT a row whose
    // tenant_id is tenant-B. The WITH CHECK clause rejects.
    let mut tx = app_pool.begin().await.unwrap();
    sqlx::query("SELECT set_config('entelix.tenant_id', 'tenant-A', true)")
        .execute(&mut *tx)
        .await
        .unwrap();
    let result = sqlx::query(
        r#"
        INSERT INTO graph_nodes (tenant_id, namespace_key, id, payload)
        VALUES ('tenant-B', 'tenant-B:scope', 'node-1', '"v"')
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
async fn rls_applies_to_edges_table_too() {
    let (_super_graph, app_graph, container) = boot_with_app_role().await;
    let ctx = ExecutionContext::new();
    let ns = Namespace::new(TenantId::new("tenant-A"));
    let now = Utc::now();
    let a = app_graph.add_node(&ctx, &ns, "a".into()).await.unwrap();
    let b = app_graph.add_node(&ctx, &ns, "b".into()).await.unwrap();
    app_graph
        .add_edge(&ctx, &ns, &a, &b, "ab".into(), now)
        .await
        .unwrap();

    let app_pool = pool_for(APP_ROLE, APP_PASSWORD, &container).await;
    let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM graph_edges")
        .fetch_one(&*app_pool)
        .await
        .unwrap();
    assert_eq!(
        count.0, 0,
        "graph_edges RLS must hide rows when tenant unset"
    );
}

#[tokio::test]
#[ignore = "requires docker"]
async fn correct_tenant_session_returns_rows() {
    let (_super_graph, app_graph, container) = boot_with_app_role().await;
    let ctx = ExecutionContext::new();
    let ns = Namespace::new(TenantId::new("tenant-X")).with_scope("conv");
    app_graph
        .add_node(&ctx, &ns, "node-1".into())
        .await
        .unwrap();

    let app_pool = pool_for(APP_ROLE, APP_PASSWORD, &container).await;
    let count = count_in_tx(&app_pool, "tenant-X").await;
    assert_eq!(count, 1, "row visible when correct tenant set");
    let mismatched = count_in_tx(&app_pool, "tenant-Y").await;
    assert_eq!(mismatched, 0, "row hidden when wrong tenant set");
}

async fn pool_for(user: &str, pw: &str, container: &ContainerAsync<Postgres>) -> Arc<PgPool> {
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let url = format!("postgres://{user}:{pw}@127.0.0.1:{port}/postgres");
    Arc::new(PgPoolOptions::new().connect(&url).await.unwrap())
}

async fn count_in_tx(pool: &PgPool, tenant: &str) -> i64 {
    let mut tx = pool.begin().await.unwrap();
    sqlx::query("SELECT set_config('entelix.tenant_id', $1, true)")
        .bind(tenant)
        .execute(&mut *tx)
        .await
        .unwrap();
    let row: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM graph_nodes")
        .fetch_one(&mut *tx)
        .await
        .unwrap();
    tx.commit().await.unwrap();
    row.0
}
