//! `QdrantVectorStore` end-to-end via testcontainers. All tests
//! are `#[ignore]` so a host without docker still passes
//! `cargo test --workspace`.
//!
//! Run with:
//!
//! ```text
//! cargo test -p entelix-memory-qdrant --test qdrant_e2e -- --ignored
//! ```

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::doc_markdown
)]

use entelix_core::TenantId;
use entelix_core::context::ExecutionContext;
use entelix_memory::{Document, Namespace, VectorFilter, VectorStore};
use entelix_memory_qdrant::{DistanceMetric, QdrantVectorStore};
use serde_json::json;
use testcontainers::core::{ContainerPort, WaitFor};
use testcontainers::runners::AsyncRunner;
use testcontainers::{ContainerAsync, GenericImage};

const DIMENSION: usize = 4;

async fn boot_qdrant() -> (ContainerAsync<GenericImage>, QdrantVectorStore) {
    let container = GenericImage::new("qdrant/qdrant", "v1.13.0")
        .with_exposed_port(ContainerPort::Tcp(6334))
        .with_wait_for(WaitFor::message_on_stdout("Qdrant gRPC listening on"))
        .start()
        .await
        .expect("qdrant container started");
    let port = container
        .get_host_port_ipv4(ContainerPort::Tcp(6334))
        .await
        .expect("qdrant gRPC port");
    let url = format!("http://127.0.0.1:{port}");

    let store = QdrantVectorStore::builder("test_collection", DIMENSION)
        .with_url(url)
        .with_distance(DistanceMetric::Cosine)
        .build()
        .await
        .expect("store built");
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
    let (_container, store) = boot_qdrant().await;
    let ctx = ExecutionContext::new();
    let ns = Namespace::new(TenantId::new("tenant-a")).with_scope("default");

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

    // Delete uses the deterministic PointId derivation.
    let to_delete = hits[0].doc_id.clone().unwrap();
    store.delete(&ctx, &ns, &to_delete).await.unwrap();
    assert_eq!(store.count(&ctx, &ns, None).await.unwrap(), 1);
}

#[tokio::test]
#[ignore = "requires docker"]
async fn cross_tenant_writes_are_isolated() {
    let (_container, store) = boot_qdrant().await;
    let ctx = ExecutionContext::new();
    let ns_a = Namespace::new(TenantId::new("tenant-a")).with_scope("default");
    let ns_b = Namespace::new(TenantId::new("tenant-b")).with_scope("default");

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

    // Search in tenant-a returns alpha; tenant-b returns bravo.
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
    let (_container, store) = boot_qdrant().await;
    let ctx = ExecutionContext::new();
    let ns = Namespace::new(TenantId::new("tenant-a")).with_scope("default");

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
async fn add_batch_uses_single_round_trip() {
    let (_container, store) = boot_qdrant().await;
    let ctx = ExecutionContext::new();
    let ns = Namespace::new(TenantId::new("tenant-a")).with_scope("default");

    let items = (0..5)
        .map(|i| {
            let mut v = vec![0.0; DIMENSION];
            v[i % DIMENSION] = 1.0;
            (doc(&format!("doc-{i}"), json!({"i": i})), v)
        })
        .collect();
    store.add_batch(&ctx, &ns, items).await.unwrap();
    assert_eq!(store.count(&ctx, &ns, None).await.unwrap(), 5);
}

#[tokio::test]
#[ignore = "requires docker"]
async fn update_replaces_atomically() {
    let (_container, store) = boot_qdrant().await;
    let ctx = ExecutionContext::new();
    let ns = Namespace::new(TenantId::new("tenant-a")).with_scope("default");

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

    let hits = store
        .search(&ctx, &ns, &[0.5, 0.5, 0.0, 0.0], 1)
        .await
        .unwrap();
    assert_eq!(hits[0].content, "v2");
}

#[tokio::test]
#[ignore = "requires docker"]
async fn colon_bearing_namespaces_isolated() {
    let (_container, store) = boot_qdrant().await;
    let ctx = ExecutionContext::new();
    let a = Namespace::new(TenantId::new("t:1")).with_scope("a:b");
    let b = Namespace::new(TenantId::new("t")).with_scope("1:a:b");
    assert_ne!(
        a.render(),
        b.render(),
        "namespace render must escape colons"
    );

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
