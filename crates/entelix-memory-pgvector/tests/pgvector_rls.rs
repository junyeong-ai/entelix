//! Row-level-security regression for `entelix-memory-pgvector` —
//! mirror of `entelix-graphmemory-pg`'s `postgres_rls.rs` and
//! `entelix-persistence`'s `postgres_rls.rs`. Proves the
//! `tenant_isolation` policy installed by the bootstrap actually
//! rejects cross-tenant access at the database layer.
//!
//! Default `postgres` role is `SUPERUSER` and bypasses every RLS
//! policy regardless of `FORCE`. To exercise the gate this suite
//! mints a second `NOSUPERUSER NOBYPASSRLS` role inside the
//! container, grants it CRUD on `entelix_vectors`, and connects
//! a second pool through it.
//!
//! Run with:
//!
//! ```text
//! cargo test -p entelix-memory-pgvector --test pgvector_rls -- --ignored
//! ```

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::doc_markdown
)]

use entelix_core::context::ExecutionContext;
use entelix_memory::{Document, Namespace, VectorStore};
use entelix_memory_pgvector::PgVectorStore;
use serde_json::json;
use sqlx::Executor;
use sqlx::postgres::{PgPool, PgPoolOptions};
use testcontainers::ContainerAsync;
use testcontainers::core::{ContainerPort, WaitFor};
use testcontainers::runners::AsyncRunner;
use testcontainers::{GenericImage, ImageExt};

const DIMENSION: usize = 4;
const APP_ROLE: &str = "vec_app";
const APP_PASSWORD: &str = "apppwd";

async fn boot_with_app_role() -> (ContainerAsync<GenericImage>, PgVectorStore, PgPool) {
    let container = GenericImage::new("pgvector/pgvector", "pg17")
        .with_exposed_port(ContainerPort::Tcp(5432))
        .with_wait_for(WaitFor::message_on_stderr(
            "database system is ready to accept connections",
        ))
        .with_env_var("POSTGRES_PASSWORD", "postgres")
        .with_env_var("POSTGRES_USER", "postgres")
        .with_env_var("POSTGRES_DB", "entelix")
        .start()
        .await
        .expect("postgres+pgvector container started");
    let port = container
        .get_host_port_ipv4(ContainerPort::Tcp(5432))
        .await
        .expect("postgres port");
    let super_url = format!("postgres://postgres:postgres@127.0.0.1:{port}/entelix");

    // Build once as superuser to create schema + RLS policy.
    let super_pool = PgPoolOptions::new().connect(&super_url).await.unwrap();
    super_pool
        .execute("CREATE EXTENSION IF NOT EXISTS vector")
        .await
        .unwrap();
    let _super_store = PgVectorStore::builder(DIMENSION)
        .with_pool(super_pool.clone())
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
            format!("GRANT SELECT, INSERT, UPDATE, DELETE ON entelix_vectors TO {APP_ROLE}")
                .as_str(),
        )
        .await
        .unwrap();
    super_pool
        .execute(format!("GRANT USAGE ON SCHEMA public TO {APP_ROLE}").as_str())
        .await
        .unwrap();

    let app_url = format!("postgres://{APP_ROLE}:{APP_PASSWORD}@127.0.0.1:{port}/entelix");
    let app_pool = PgPoolOptions::new().connect(&app_url).await.unwrap();
    let app_store = PgVectorStore::builder(DIMENSION)
        .with_pool(app_pool.clone())
        .with_auto_migrate(false)
        .build()
        .await
        .unwrap();
    (container, app_store, app_pool)
}

fn doc(content: &str) -> Document {
    Document {
        doc_id: None,
        content: content.into(),
        metadata: json!({}),
        score: None,
    }
}

fn vec4(seed: f32) -> Vec<f32> {
    vec![seed, seed * 0.5, -seed, 1.0 - seed]
}

#[tokio::test]
#[ignore = "requires docker"]
async fn rls_blocks_cross_tenant_search_at_db_layer() {
    let (_c, app_store, app_pool) = boot_with_app_role().await;
    let ctx = ExecutionContext::new();

    // SDK path stamps `entelix.tenant_id` per transaction → write
    // succeeds (WITH CHECK satisfied), read with same tenant
    // succeeds (USING satisfied).
    let ns_a = Namespace::new("tenant-A").with_scope("scope");
    app_store
        .add(&ctx, &ns_a, doc("hello"), vec4(0.1))
        .await
        .unwrap();
    let hits = app_store.search(&ctx, &ns_a, &vec4(0.1), 5).await.unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].content, "hello");

    // Bypass the SDK — raw read with no `entelix.tenant_id` set.
    // Policy treats `tenant_id = NULL` as false, hides every row.
    let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM entelix_vectors")
        .fetch_one(&app_pool)
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
    let (_c, _app_store, app_pool) = boot_with_app_role().await;

    let mut tx = app_pool.begin().await.unwrap();
    sqlx::query("SELECT set_config('entelix.tenant_id', 'tenant-A', true)")
        .execute(&mut *tx)
        .await
        .unwrap();
    // Try to INSERT a row whose tenant_id is tenant-B. Embedding
    // is a 4-dim vector literal matching DIMENSION.
    let result = sqlx::query(
        "
        INSERT INTO entelix_vectors
            (tenant_id, namespace_key, doc_id, content, metadata, embedding)
        VALUES ('tenant-B', 'tenant-B:scope', 'd1', 'x', '{}'::jsonb, '[0.1,0.2,0.3,0.4]')
        ",
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
async fn correct_tenant_session_returns_rows_wrong_tenant_does_not() {
    let (_c, app_store, app_pool) = boot_with_app_role().await;
    let ctx = ExecutionContext::new();
    let ns = Namespace::new("tenant-X").with_scope("conv");
    app_store
        .add(&ctx, &ns, doc("alpha"), vec4(0.5))
        .await
        .unwrap();

    let visible = count_in_tx(&app_pool, "tenant-X").await;
    assert_eq!(visible, 1, "row visible when correct tenant set");
    let hidden = count_in_tx(&app_pool, "tenant-Y").await;
    assert_eq!(hidden, 0, "row hidden when wrong tenant set");
}

async fn count_in_tx(pool: &PgPool, tenant: &str) -> i64 {
    let mut tx = pool.begin().await.unwrap();
    sqlx::query("SELECT set_config('entelix.tenant_id', $1, true)")
        .bind(tenant)
        .execute(&mut *tx)
        .await
        .unwrap();
    let row: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM entelix_vectors")
        .fetch_one(&mut *tx)
        .await
        .unwrap();
    tx.commit().await.unwrap();
    row.0
}
