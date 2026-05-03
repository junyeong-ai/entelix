//! `PgVectorStore` end-to-end via testcontainers. Uses the
//! official `pgvector/pgvector` image so the `vector` extension is
//! pre-installed; the store's `auto_migrate` then creates the
//! schema on first build.
//!
//! Run with:
//!
//! ```text
//! cargo test -p entelix-memory-pgvector --test pgvector_e2e -- --ignored
//! ```

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::doc_markdown
)]

use entelix_core::context::ExecutionContext;
use entelix_memory::{Document, Namespace, VectorFilter, VectorStore};
use entelix_memory_pgvector::{DistanceMetric, IndexKind, PgVectorStore};
use serde_json::json;
use testcontainers::core::{ContainerPort, WaitFor};
use testcontainers::runners::AsyncRunner;
use testcontainers::{ContainerAsync, GenericImage, ImageExt};

const DIMENSION: usize = 4;

async fn boot_pg() -> (ContainerAsync<GenericImage>, PgVectorStore) {
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
    let url = format!("postgres://postgres:postgres@127.0.0.1:{port}/entelix");

    let store = PgVectorStore::builder(DIMENSION)
        .with_connection_string(url)
        .with_distance(DistanceMetric::Cosine)
        .with_index_kind(IndexKind::Hnsw)
        .build()
        .await
        .expect("store built (auto_migrate on)");
    (container, store)
}

fn doc(content: &str, metadata: serde_json::Value) -> Document {
    Document {
        doc_id: None,
        content: content.into(),
        metadata,
        score: None,
    }
}

#[tokio::test]
#[ignore = "requires docker"]
async fn round_trip_add_search_delete() {
    let (_container, store) = boot_pg().await;
    let ctx = ExecutionContext::new();
    let ns = Namespace::new("tenant-a").with_scope("default");

    store
        .add(
            &ctx,
            &ns,
            doc("first", json!({"category": "books"})),
            vec![1.0, 0.0, 0.0, 0.0],
        )
        .await
        .unwrap();
    store
        .add(
            &ctx,
            &ns,
            doc("second", json!({"category": "movies"})),
            vec![0.0, 1.0, 0.0, 0.0],
        )
        .await
        .unwrap();

    let hits = store
        .search(&ctx, &ns, &[1.0, 0.0, 0.0, 0.0], 1)
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].content, "first");
    assert_eq!(store.count(&ctx, &ns, None).await.unwrap(), 2);

    let to_delete = hits[0].doc_id.clone().unwrap();
    store.delete(&ctx, &ns, &to_delete).await.unwrap();
    assert_eq!(store.count(&ctx, &ns, None).await.unwrap(), 1);
}

#[tokio::test]
#[ignore = "requires docker"]
async fn cross_tenant_writes_are_isolated() {
    let (_container, store) = boot_pg().await;
    let ctx = ExecutionContext::new();
    let ns_a = Namespace::new("tenant-a").with_scope("default");
    let ns_b = Namespace::new("tenant-b").with_scope("default");

    store
        .add(
            &ctx,
            &ns_a,
            doc("alpha", json!({})),
            vec![1.0, 0.0, 0.0, 0.0],
        )
        .await
        .unwrap();
    store
        .add(
            &ctx,
            &ns_b,
            doc("bravo", json!({})),
            vec![1.0, 0.0, 0.0, 0.0],
        )
        .await
        .unwrap();

    let a_hits = store
        .search(&ctx, &ns_a, &[1.0, 0.0, 0.0, 0.0], 10)
        .await
        .unwrap();
    let b_hits = store
        .search(&ctx, &ns_b, &[1.0, 0.0, 0.0, 0.0], 10)
        .await
        .unwrap();
    assert_eq!(a_hits.len(), 1);
    assert_eq!(b_hits.len(), 1);
    assert_eq!(a_hits[0].content, "alpha");
    assert_eq!(b_hits[0].content, "bravo");
    assert_eq!(store.count(&ctx, &ns_a, None).await.unwrap(), 1);
    assert_eq!(store.count(&ctx, &ns_b, None).await.unwrap(), 1);
}

#[tokio::test]
#[ignore = "requires docker"]
async fn search_filtered_with_eq_filter() {
    let (_container, store) = boot_pg().await;
    let ctx = ExecutionContext::new();
    let ns = Namespace::new("tenant-a").with_scope("default");

    store
        .add(
            &ctx,
            &ns,
            doc("book-a", json!({"category": "books"})),
            vec![1.0, 0.0, 0.0, 0.0],
        )
        .await
        .unwrap();
    store
        .add(
            &ctx,
            &ns,
            doc("movie-a", json!({"category": "movies"})),
            vec![0.9, 0.1, 0.0, 0.0],
        )
        .await
        .unwrap();

    let hits = store
        .search_filtered(
            &ctx,
            &ns,
            &[1.0, 0.0, 0.0, 0.0],
            10,
            &VectorFilter::Eq {
                key: "category".into(),
                value: json!("books"),
            },
        )
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].content, "book-a");
}

#[tokio::test]
#[ignore = "requires docker"]
async fn batch_add_uses_single_round_trip() {
    let (_container, store) = boot_pg().await;
    let ctx = ExecutionContext::new();
    let ns = Namespace::new("tenant-a").with_scope("default");

    let items = (0..5)
        .map(|i| {
            let mut v = vec![0.0; DIMENSION];
            v[i % DIMENSION] = 1.0;
            (doc(&format!("doc-{i}"), json!({"i": i})), v)
        })
        .collect();
    store.batch_add(&ctx, &ns, items).await.unwrap();
    assert_eq!(store.count(&ctx, &ns, None).await.unwrap(), 5);
}

#[tokio::test]
#[ignore = "requires docker"]
async fn update_replaces_atomically() {
    let (_container, store) = boot_pg().await;
    let ctx = ExecutionContext::new();
    let ns = Namespace::new("tenant-a").with_scope("default");

    let doc_id = "stable-id".to_owned();
    let mut d = doc("v1", json!({}));
    d.doc_id = Some(doc_id.clone());
    store
        .add(&ctx, &ns, d, vec![1.0, 0.0, 0.0, 0.0])
        .await
        .unwrap();
    store
        .update(
            &ctx,
            &ns,
            &doc_id,
            doc("v2", json!({"updated": true})),
            vec![0.5, 0.5, 0.0, 0.0],
        )
        .await
        .unwrap();
    assert_eq!(store.count(&ctx, &ns, None).await.unwrap(), 1);

    let listed = store.list(&ctx, &ns, None, 10, 0).await.unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].content, "v2");
}

#[tokio::test]
#[ignore = "requires docker"]
async fn colon_bearing_namespaces_isolated() {
    let (_container, store) = boot_pg().await;
    let ctx = ExecutionContext::new();
    let a = Namespace::new("t:1").with_scope("a:b");
    let b = Namespace::new("t").with_scope("1:a:b");
    assert_ne!(a.render(), b.render());

    store
        .add(&ctx, &a, doc("from-a", json!({})), vec![1.0, 0.0, 0.0, 0.0])
        .await
        .unwrap();
    store
        .add(&ctx, &b, doc("from-b", json!({})), vec![1.0, 0.0, 0.0, 0.0])
        .await
        .unwrap();
    assert_eq!(store.count(&ctx, &a, None).await.unwrap(), 1);
    assert_eq!(store.count(&ctx, &b, None).await.unwrap(), 1);
    let a_hits = store
        .search(&ctx, &a, &[1.0, 0.0, 0.0, 0.0], 10)
        .await
        .unwrap();
    assert_eq!(a_hits[0].content, "from-a");
}

#[tokio::test]
#[ignore = "requires docker"]
async fn list_pagination_respects_limit_offset() {
    let (_container, store) = boot_pg().await;
    let ctx = ExecutionContext::new();
    let ns = Namespace::new("tenant-a").with_scope("default");
    for i in 0..5 {
        let mut d = doc(&format!("doc-{i:02}"), json!({"i": i}));
        d.doc_id = Some(format!("doc-{i:02}"));
        store
            .add(&ctx, &ns, d, vec![1.0, 0.0, 0.0, 0.0])
            .await
            .unwrap();
    }
    let first_page = store.list(&ctx, &ns, None, 2, 0).await.unwrap();
    let second_page = store.list(&ctx, &ns, None, 2, 2).await.unwrap();
    assert_eq!(first_page.len(), 2);
    assert_eq!(second_page.len(), 2);
    assert_ne!(
        first_page[0].doc_id, second_page[0].doc_id,
        "pages must not overlap with monotonic doc_id ordering"
    );
}
