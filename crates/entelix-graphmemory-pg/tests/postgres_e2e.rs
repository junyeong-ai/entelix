//! End-to-end Postgres integration tests using `testcontainers`.
//! Requires a working docker daemon. Run with:
//!
//! ```text
//! cargo test -p entelix-graphmemory-pg --test postgres_e2e -- --ignored
//! ```

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::doc_markdown,
    clippy::many_single_char_names
)]

use entelix_core::TenantId;
use std::sync::Arc;

use chrono::{Duration, Utc};
use entelix_core::ExecutionContext;
use entelix_graphmemory_pg::PgGraphMemory;
use entelix_memory::{Direction, GraphMemory, Namespace};
use testcontainers_modules::postgres::Postgres;
use testcontainers_modules::testcontainers::ContainerAsync;
use testcontainers_modules::testcontainers::runners::AsyncRunner;

async fn boot() -> (PgGraphMemory<String, String>, ContainerAsync<Postgres>) {
    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let url = format!("postgres://postgres:postgres@127.0.0.1:{port}/postgres");
    let graph = PgGraphMemory::<String, String>::builder()
        .with_connection_string(url)
        .build()
        .await
        .unwrap();
    (graph, container)
}

#[tokio::test]
#[ignore = "requires docker"]
async fn add_and_lookup_node_round_trip() {
    let (graph, _c) = boot().await;
    let ns = Namespace::new(TenantId::new("acme")).with_scope("agent-a");
    let ctx = ExecutionContext::new();

    let id = graph.add_node(&ctx, &ns, "alice".into()).await.unwrap();
    let got = graph.node(&ctx, &ns, &id).await.unwrap();
    assert_eq!(got.as_deref(), Some("alice"));

    // Same id under a different namespace must not leak.
    let other_ns = Namespace::new(TenantId::new("acme")).with_scope("agent-b");
    assert!(graph.node(&ctx, &other_ns, &id).await.unwrap().is_none());
}

#[tokio::test]
#[ignore = "requires docker"]
async fn add_edges_batch_inserts_all_in_one_round_trip() {
    // Bulk-insert path: 50 edges via a single INSERT … SELECT FROM
    // UNNEST(…) — verifies the per-column array binds round-trip
    // through sqlx, the JSONB column accepts the marshaled payloads,
    // every assigned EdgeId resolves back to a hop with the right
    // endpoints, and the count rises by exactly the input length.
    let (graph, _c) = boot().await;
    let ns = Namespace::new(TenantId::new("acme"));
    let ctx = ExecutionContext::new();
    let now = Utc::now();
    // Five hub nodes; build 50 edges in a star-of-stars pattern.
    let mut hubs = Vec::new();
    for i in 0..5 {
        hubs.push(graph.add_node(&ctx, &ns, format!("hub-{i}")).await.unwrap());
    }
    let mut leaves = Vec::new();
    for i in 0..50 {
        leaves.push(
            graph
                .add_node(&ctx, &ns, format!("leaf-{i}"))
                .await
                .unwrap(),
        );
    }
    let edges: Vec<_> = leaves
        .iter()
        .enumerate()
        .map(|(i, leaf)| {
            (
                hubs[i % hubs.len()].clone(),
                leaf.clone(),
                format!("edge-{i}"),
                now,
            )
        })
        .collect();
    let count_before = graph.edge_count(&ctx, &ns).await.unwrap();
    let ids = graph.add_edges_batch(&ctx, &ns, edges).await.unwrap();
    let count_after = graph.edge_count(&ctx, &ns).await.unwrap();
    assert_eq!(ids.len(), 50);
    assert_eq!(count_after - count_before, 50);
    // Spot-check round-trip on the first and last assigned ids.
    let first = graph.edge(&ctx, &ns, &ids[0]).await.unwrap();
    let last = graph.edge(&ctx, &ns, &ids[49]).await.unwrap();
    assert!(first.is_some() && last.is_some());
    assert_eq!(first.unwrap().edge, "edge-0");
    assert_eq!(last.unwrap().edge, "edge-49");
}

#[tokio::test]
#[ignore = "requires docker"]
async fn add_edges_batch_empty_input_is_a_noop() {
    let (graph, _c) = boot().await;
    let ns = Namespace::new(TenantId::new("acme"));
    let ctx = ExecutionContext::new();
    let count_before = graph.edge_count(&ctx, &ns).await.unwrap();
    let ids = graph.add_edges_batch(&ctx, &ns, Vec::new()).await.unwrap();
    assert!(ids.is_empty());
    assert_eq!(graph.edge_count(&ctx, &ns).await.unwrap(), count_before);
}

#[tokio::test]
#[ignore = "requires docker"]
async fn neighbors_returns_outgoing_incoming_and_both() {
    let (graph, _c) = boot().await;
    let ns = Namespace::new(TenantId::new("acme"));
    let ctx = ExecutionContext::new();
    let now = Utc::now();

    let a = graph.add_node(&ctx, &ns, "A".into()).await.unwrap();
    let b = graph.add_node(&ctx, &ns, "B".into()).await.unwrap();
    let c = graph.add_node(&ctx, &ns, "C".into()).await.unwrap();
    let _ab = graph
        .add_edge(&ctx, &ns, &a, &b, "ab".into(), now)
        .await
        .unwrap();
    let _ca = graph
        .add_edge(&ctx, &ns, &c, &a, "ca".into(), now)
        .await
        .unwrap();

    let outgoing = graph
        .neighbors(&ctx, &ns, &a, Direction::Outgoing)
        .await
        .unwrap();
    assert_eq!(outgoing.len(), 1);
    assert_eq!(outgoing[0].1, b);

    let incoming = graph
        .neighbors(&ctx, &ns, &a, Direction::Incoming)
        .await
        .unwrap();
    assert_eq!(incoming.len(), 1);
    assert_eq!(incoming[0].1, c);

    let both = graph
        .neighbors(&ctx, &ns, &a, Direction::Both)
        .await
        .unwrap();
    assert_eq!(both.len(), 2);
}

#[tokio::test]
#[ignore = "requires docker"]
async fn traverse_bfs_respects_max_depth() {
    let (graph, _c) = boot().await;
    let ns = Namespace::new(TenantId::new("acme"));
    let ctx = ExecutionContext::new();
    let now = Utc::now();

    // Chain a → b → c → d
    let a = graph.add_node(&ctx, &ns, "a".into()).await.unwrap();
    let b = graph.add_node(&ctx, &ns, "b".into()).await.unwrap();
    let c = graph.add_node(&ctx, &ns, "c".into()).await.unwrap();
    let d = graph.add_node(&ctx, &ns, "d".into()).await.unwrap();
    let _ = graph
        .add_edge(&ctx, &ns, &a, &b, "1".into(), now)
        .await
        .unwrap();
    let _ = graph
        .add_edge(&ctx, &ns, &b, &c, "2".into(), now)
        .await
        .unwrap();
    let _ = graph
        .add_edge(&ctx, &ns, &c, &d, "3".into(), now)
        .await
        .unwrap();

    let depth_1 = graph
        .traverse(&ctx, &ns, &a, Direction::Outgoing, 1)
        .await
        .unwrap();
    assert_eq!(depth_1.len(), 1);
    assert_eq!(depth_1[0].to, b);

    let depth_3 = graph
        .traverse(&ctx, &ns, &a, Direction::Outgoing, 3)
        .await
        .unwrap();
    assert_eq!(depth_3.len(), 3);
    assert_eq!(depth_3.last().unwrap().to, d);
}

#[tokio::test]
#[ignore = "requires docker"]
async fn find_path_returns_shortest_or_none() {
    let (graph, _c) = boot().await;
    let ns = Namespace::new(TenantId::new("acme"));
    let ctx = ExecutionContext::new();
    let now = Utc::now();

    let a = graph.add_node(&ctx, &ns, "a".into()).await.unwrap();
    let b = graph.add_node(&ctx, &ns, "b".into()).await.unwrap();
    let c = graph.add_node(&ctx, &ns, "c".into()).await.unwrap();
    let isolated = graph.add_node(&ctx, &ns, "iso".into()).await.unwrap();
    let _ = graph
        .add_edge(&ctx, &ns, &a, &b, "1".into(), now)
        .await
        .unwrap();
    let _ = graph
        .add_edge(&ctx, &ns, &b, &c, "2".into(), now)
        .await
        .unwrap();

    let path = graph
        .find_path(&ctx, &ns, &a, &c, Direction::Outgoing, 5)
        .await
        .unwrap()
        .expect("a→c path should exist");
    assert_eq!(path.len(), 2);
    assert_eq!(path[0].from, a);
    assert_eq!(path[0].to, b);
    assert_eq!(path[1].from, b);
    assert_eq!(path[1].to, c);

    // Same-node case yields empty path, not None.
    let same = graph
        .find_path(&ctx, &ns, &a, &a, Direction::Outgoing, 5)
        .await
        .unwrap();
    assert!(same.is_some_and(|v| v.is_empty()));

    // Unreachable returns None.
    let none = graph
        .find_path(&ctx, &ns, &a, &isolated, Direction::Outgoing, 5)
        .await
        .unwrap();
    assert!(none.is_none());
}

#[tokio::test]
#[ignore = "requires docker"]
async fn traverse_terminates_on_cycle() {
    // a → b → c → a forms a cycle. The recursive CTE's per-row
    // `visited` array must keep the walk finite — no row may
    // revisit a node already in its path. Without cycle prevention
    // the recursion would saturate `max_depth` filling repeated
    // rows; with it, the dedupe layer keeps exactly one hop per
    // reachable destination.
    let (graph, _c) = boot().await;
    let ns = Namespace::new(TenantId::new("acme"));
    let ctx = ExecutionContext::new();
    let now = Utc::now();
    let a = graph.add_node(&ctx, &ns, "a".into()).await.unwrap();
    let b = graph.add_node(&ctx, &ns, "b".into()).await.unwrap();
    let c = graph.add_node(&ctx, &ns, "c".into()).await.unwrap();
    let _ = graph
        .add_edge(&ctx, &ns, &a, &b, "ab".into(), now)
        .await
        .unwrap();
    let _ = graph
        .add_edge(&ctx, &ns, &b, &c, "bc".into(), now)
        .await
        .unwrap();
    let _ = graph
        .add_edge(&ctx, &ns, &c, &a, "ca".into(), now)
        .await
        .unwrap();

    let hops = graph
        .traverse(&ctx, &ns, &a, Direction::Outgoing, 10)
        .await
        .unwrap();
    // Two distinct destinations reachable from `a` in this cycle:
    // `b` (one hop) and `c` (two hops). `a` itself is the seed and
    // is excluded from the dedupe surface.
    assert_eq!(hops.len(), 2);
    let destinations: Vec<_> = hops.iter().map(|h| h.to.clone()).collect();
    assert!(destinations.contains(&b));
    assert!(destinations.contains(&c));
}

#[tokio::test]
#[ignore = "requires docker"]
async fn find_path_picks_shortest_among_multiple() {
    // Two paths from a to d:
    //   short: a → b → d  (2 hops)
    //   long:  a → c → e → d (3 hops)
    // The recursive CTE's `ORDER BY depth ASC LIMIT 1` on the
    // `shortest` projection must pick the 2-hop traversal.
    let (graph, _c) = boot().await;
    let ns = Namespace::new(TenantId::new("acme"));
    let ctx = ExecutionContext::new();
    let now = Utc::now();
    let a = graph.add_node(&ctx, &ns, "a".into()).await.unwrap();
    let b = graph.add_node(&ctx, &ns, "b".into()).await.unwrap();
    let c = graph.add_node(&ctx, &ns, "c".into()).await.unwrap();
    let d = graph.add_node(&ctx, &ns, "d".into()).await.unwrap();
    let e = graph.add_node(&ctx, &ns, "e".into()).await.unwrap();
    let _ = graph
        .add_edge(&ctx, &ns, &a, &b, "ab".into(), now)
        .await
        .unwrap();
    let _ = graph
        .add_edge(&ctx, &ns, &b, &d, "bd".into(), now)
        .await
        .unwrap();
    let _ = graph
        .add_edge(&ctx, &ns, &a, &c, "ac".into(), now)
        .await
        .unwrap();
    let _ = graph
        .add_edge(&ctx, &ns, &c, &e, "ce".into(), now)
        .await
        .unwrap();
    let _ = graph
        .add_edge(&ctx, &ns, &e, &d, "ed".into(), now)
        .await
        .unwrap();

    let path = graph
        .find_path(&ctx, &ns, &a, &d, Direction::Outgoing, 5)
        .await
        .unwrap()
        .expect("a→d path exists");
    assert_eq!(path.len(), 2);
    assert_eq!(path[0].from, a);
    assert_eq!(path[0].to, b);
    assert_eq!(path[1].from, b);
    assert_eq!(path[1].to, d);
}

#[tokio::test]
#[ignore = "requires docker"]
async fn traverse_max_depth_zero_returns_empty() {
    // `max_depth = 0` means "no edges" — the recursive CTE's
    // `WHERE w.depth < $3` is `< 0` for the first expansion, so
    // only the depth-0 base row exists, and it's filtered out by
    // `WHERE depth > 0` in the ranked projection.
    let (graph, _c) = boot().await;
    let ns = Namespace::new(TenantId::new("acme"));
    let ctx = ExecutionContext::new();
    let now = Utc::now();
    let a = graph.add_node(&ctx, &ns, "a".into()).await.unwrap();
    let b = graph.add_node(&ctx, &ns, "b".into()).await.unwrap();
    let _ = graph
        .add_edge(&ctx, &ns, &a, &b, "ab".into(), now)
        .await
        .unwrap();

    let hops = graph
        .traverse(&ctx, &ns, &a, Direction::Outgoing, 0)
        .await
        .unwrap();
    assert!(hops.is_empty());
}

#[tokio::test]
#[ignore = "requires docker"]
async fn traverse_direction_both_handles_cycles() {
    // Bidirectional walk on a small cycle: a ↔ b ↔ c ↔ a where the
    // edges are stored as a→b, b→c, c→a. From `a` with
    // `Direction::Both` and `max_depth = 5`, every node should be
    // reachable, and the visited-array cycle prevention must keep
    // the walk finite (the `(from_node OR to_node)` join with the
    // `CASE WHEN ... END` next-node expression is the most complex
    // shape — verifying termination here exercises that branch).
    let (graph, _c) = boot().await;
    let ns = Namespace::new(TenantId::new("acme"));
    let ctx = ExecutionContext::new();
    let now = Utc::now();
    let a = graph.add_node(&ctx, &ns, "a".into()).await.unwrap();
    let b = graph.add_node(&ctx, &ns, "b".into()).await.unwrap();
    let c = graph.add_node(&ctx, &ns, "c".into()).await.unwrap();
    let _ = graph
        .add_edge(&ctx, &ns, &a, &b, "ab".into(), now)
        .await
        .unwrap();
    let _ = graph
        .add_edge(&ctx, &ns, &b, &c, "bc".into(), now)
        .await
        .unwrap();
    let _ = graph
        .add_edge(&ctx, &ns, &c, &a, "ca".into(), now)
        .await
        .unwrap();

    let hops = graph
        .traverse(&ctx, &ns, &a, Direction::Both, 5)
        .await
        .unwrap();
    // `b` and `c` are both reachable; the seed `a` is excluded.
    let destinations: std::collections::HashSet<_> = hops.iter().map(|h| h.to.clone()).collect();
    let origins: std::collections::HashSet<_> = hops.iter().map(|h| h.from.clone()).collect();
    let touched = destinations.union(&origins).cloned().collect::<Vec<_>>();
    assert!(touched.contains(&b));
    assert!(touched.contains(&c));
}

#[tokio::test]
#[ignore = "requires docker"]
async fn temporal_filter_picks_window() {
    let (graph, _c) = boot().await;
    let ns = Namespace::new(TenantId::new("acme"));
    let ctx = ExecutionContext::new();
    let base = Utc::now();

    let a = graph.add_node(&ctx, &ns, "a".into()).await.unwrap();
    let b = graph.add_node(&ctx, &ns, "b".into()).await.unwrap();
    let c = graph.add_node(&ctx, &ns, "c".into()).await.unwrap();
    let _ = graph
        .add_edge(
            &ctx,
            &ns,
            &a,
            &b,
            "old".into(),
            base - Duration::seconds(120),
        )
        .await
        .unwrap();
    let _ = graph
        .add_edge(
            &ctx,
            &ns,
            &b,
            &c,
            "mid".into(),
            base - Duration::seconds(30),
        )
        .await
        .unwrap();
    let _ = graph
        .add_edge(
            &ctx,
            &ns,
            &c,
            &a,
            "new".into(),
            base + Duration::seconds(30),
        )
        .await
        .unwrap();

    let window = graph
        .temporal_filter(
            &ctx,
            &ns,
            base - Duration::seconds(60),
            base + Duration::seconds(60),
        )
        .await
        .unwrap();
    assert_eq!(window.len(), 2);
    assert_eq!(window[0].edge, "mid");
    assert_eq!(window[1].edge, "new");
}

#[tokio::test]
#[ignore = "requires docker"]
async fn namespaces_isolate_nodes_and_edges() {
    let (graph, _c) = boot().await;
    let ctx = ExecutionContext::new();
    let now = Utc::now();
    let ns_a = Namespace::new(TenantId::new("tenant-a"));
    let ns_b = Namespace::new(TenantId::new("tenant-b"));

    let a_node = graph.add_node(&ctx, &ns_a, "a".into()).await.unwrap();
    let b_node = graph.add_node(&ctx, &ns_b, "b".into()).await.unwrap();
    let _ = graph
        .add_edge(&ctx, &ns_a, &a_node, &a_node, "self-a".into(), now)
        .await
        .unwrap();
    let _ = graph
        .add_edge(&ctx, &ns_b, &b_node, &b_node, "self-b".into(), now)
        .await
        .unwrap();

    let from_a = graph
        .neighbors(&ctx, &ns_a, &a_node, Direction::Both)
        .await
        .unwrap();
    let from_b = graph
        .neighbors(&ctx, &ns_b, &b_node, Direction::Both)
        .await
        .unwrap();
    assert_eq!(from_a.len(), 1);
    assert_eq!(from_b.len(), 1);

    // Cross-namespace lookup must miss.
    assert!(graph.node(&ctx, &ns_b, &a_node).await.unwrap().is_none());
    assert!(graph.node(&ctx, &ns_a, &b_node).await.unwrap().is_none());
}

#[tokio::test]
#[ignore = "requires docker"]
async fn edge_lookup_returns_full_hop_at_db_layer() {
    let (graph, _c) = boot().await;
    let ns = Namespace::new(TenantId::new("acme"));
    let ctx = ExecutionContext::new();
    let a = graph.add_node(&ctx, &ns, "a".into()).await.unwrap();
    let b = graph.add_node(&ctx, &ns, "b".into()).await.unwrap();
    let now = Utc::now();
    let id = graph
        .add_edge(&ctx, &ns, &a, &b, "ab".into(), now)
        .await
        .unwrap();
    let hop = graph.edge(&ctx, &ns, &id).await.unwrap().expect("present");
    assert_eq!(hop.edge_id, id);
    assert_eq!(hop.from, a);
    assert_eq!(hop.to, b);
    assert_eq!(hop.edge, "ab");

    // Same id under a different namespace → None (RLS gate +
    // namespace anchor both prevent the leak).
    let other_ns = Namespace::new(TenantId::new("acme")).with_scope("other");
    assert!(graph.edge(&ctx, &other_ns, &id).await.unwrap().is_none());
}

#[tokio::test]
#[ignore = "requires docker"]
async fn node_count_and_edge_count_at_db_layer() {
    let (graph, _c) = boot().await;
    let ns = Namespace::new(TenantId::new("acme"));
    let ctx = ExecutionContext::new();
    assert_eq!(graph.node_count(&ctx, &ns).await.unwrap(), 0);
    assert_eq!(graph.edge_count(&ctx, &ns).await.unwrap(), 0);
    let a = graph.add_node(&ctx, &ns, "a".into()).await.unwrap();
    let b = graph.add_node(&ctx, &ns, "b".into()).await.unwrap();
    graph
        .add_edge(&ctx, &ns, &a, &b, "ab".into(), Utc::now())
        .await
        .unwrap();
    assert_eq!(graph.node_count(&ctx, &ns).await.unwrap(), 2);
    assert_eq!(graph.edge_count(&ctx, &ns).await.unwrap(), 1);
    // Cross-namespace isolation.
    let other = Namespace::new(TenantId::new("acme")).with_scope("other");
    assert_eq!(graph.node_count(&ctx, &other).await.unwrap(), 0);
    assert_eq!(graph.edge_count(&ctx, &other).await.unwrap(), 0);
}

#[tokio::test]
#[ignore = "requires docker"]
async fn list_edges_and_records_paginate_at_db_layer() {
    let (graph, _c) = boot().await;
    let ns = Namespace::new(TenantId::new("acme"));
    let ctx = ExecutionContext::new();
    let a = graph.add_node(&ctx, &ns, "a".into()).await.unwrap();
    let b = graph.add_node(&ctx, &ns, "b".into()).await.unwrap();
    let now = Utc::now();
    let mut ids = Vec::new();
    for label in ["e1", "e2", "e3"] {
        ids.push(
            graph
                .add_edge(&ctx, &ns, &a, &b, label.into(), now)
                .await
                .unwrap(),
        );
    }
    let mut sorted = ids.clone();
    sorted.sort();

    let edge_ids = graph.list_edges(&ns, 100, 0).await.unwrap();
    assert_eq!(edge_ids, sorted);

    let records = graph.list_edge_records(&ns, 100, 0).await.unwrap();
    assert_eq!(records.len(), 3);
    let payloads: Vec<&str> = records.iter().map(|h| h.edge.as_str()).collect();
    assert!(payloads.contains(&"e1"));
    assert!(payloads.contains(&"e2"));
    assert!(payloads.contains(&"e3"));

    // Cross-namespace isolation.
    let other = Namespace::new(TenantId::new("acme")).with_scope("other");
    assert!(graph.list_edges(&other, 100, 0).await.unwrap().is_empty());
}

#[tokio::test]
#[ignore = "requires docker"]
async fn list_node_records_returns_payloads_in_one_round_trip() {
    let (graph, _c) = boot().await;
    let ns = Namespace::new(TenantId::new("acme"));
    let ctx = ExecutionContext::new();
    let mut id_payloads: Vec<(_, String)> = Vec::new();
    for label in ["alpha", "bravo", "charlie"] {
        let id = graph.add_node(&ctx, &ns, label.into()).await.unwrap();
        id_payloads.push((id, label.into()));
    }
    let mut sorted = id_payloads.clone();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));

    let records = graph.list_node_records(&ns, 100, 0).await.unwrap();
    assert_eq!(records, sorted);

    // Cross-namespace isolation.
    let other = Namespace::new(TenantId::new("acme")).with_scope("other");
    assert!(
        graph
            .list_node_records(&other, 100, 0)
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
#[ignore = "requires docker"]
async fn list_nodes_paginates_at_db_layer() {
    let (graph, _c) = boot().await;
    let ns = Namespace::new(TenantId::new("acme"));
    let ctx = ExecutionContext::new();
    let mut ids = Vec::new();
    for label in ["a", "b", "c", "d", "e"] {
        ids.push(graph.add_node(&ctx, &ns, label.into()).await.unwrap());
    }
    let mut sorted = ids.clone();
    sorted.sort();
    let first = graph.list_nodes(&ns, 3, 0).await.unwrap();
    assert_eq!(first, sorted[..3]);
    let next = graph.list_nodes(&ns, 3, 3).await.unwrap();
    assert_eq!(next, sorted[3..]);
    // Cross-namespace isolation — different ns sees zero of these.
    let other_ns = Namespace::new(TenantId::new("acme")).with_scope("other");
    assert!(
        graph
            .list_nodes(&other_ns, 100, 0)
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
#[ignore = "requires docker"]
async fn delete_edge_is_idempotent() {
    let (graph, _c) = boot().await;
    let ns = Namespace::new(TenantId::new("acme"));
    let ctx = ExecutionContext::new();
    let a = graph.add_node(&ctx, &ns, "a".into()).await.unwrap();
    let b = graph.add_node(&ctx, &ns, "b".into()).await.unwrap();
    let id = graph
        .add_edge(&ctx, &ns, &a, &b, "ab".into(), Utc::now())
        .await
        .unwrap();
    graph.delete_edge(&ctx, &ns, &id).await.unwrap();
    // Second delete on the now-absent edge succeeds.
    graph.delete_edge(&ctx, &ns, &id).await.unwrap();
    let outgoing = graph
        .neighbors(&ctx, &ns, &a, Direction::Outgoing)
        .await
        .unwrap();
    assert!(outgoing.is_empty());
}

#[tokio::test]
#[ignore = "requires docker"]
async fn delete_node_cascades_to_incident_edges_at_db_layer() {
    let (graph, _c) = boot().await;
    let ns = Namespace::new(TenantId::new("acme"));
    let ctx = ExecutionContext::new();
    let a = graph.add_node(&ctx, &ns, "a".into()).await.unwrap();
    let b = graph.add_node(&ctx, &ns, "b".into()).await.unwrap();
    let c = graph.add_node(&ctx, &ns, "c".into()).await.unwrap();
    let now = Utc::now();
    graph
        .add_edge(&ctx, &ns, &a, &b, "ab".into(), now)
        .await
        .unwrap();
    graph
        .add_edge(&ctx, &ns, &a, &c, "ac".into(), now)
        .await
        .unwrap();
    graph
        .add_edge(&ctx, &ns, &b, &a, "ba".into(), now)
        .await
        .unwrap();
    let removed = graph.delete_node(&ctx, &ns, &a).await.unwrap();
    assert_eq!(removed, 3);
    assert!(graph.node(&ctx, &ns, &a).await.unwrap().is_none());
    assert!(graph.node(&ctx, &ns, &b).await.unwrap().is_some());
    let b_in = graph
        .neighbors(&ctx, &ns, &b, Direction::Incoming)
        .await
        .unwrap();
    assert!(b_in.is_empty());
}

#[tokio::test]
#[ignore = "requires docker"]
async fn prune_orphan_nodes_drops_zero_edge_nodes_at_db_layer() {
    let (graph, _c) = boot().await;
    let ns = Namespace::new(TenantId::new("acme"));
    let ctx = ExecutionContext::new();
    let connected_a = graph.add_node(&ctx, &ns, "a".into()).await.unwrap();
    let connected_b = graph.add_node(&ctx, &ns, "b".into()).await.unwrap();
    let lonely = graph.add_node(&ctx, &ns, "lonely".into()).await.unwrap();
    graph
        .add_edge(
            &ctx,
            &ns,
            &connected_a,
            &connected_b,
            "ab".into(),
            Utc::now(),
        )
        .await
        .unwrap();

    let removed = graph.prune_orphan_nodes(&ns).await.unwrap();
    assert_eq!(removed, 1);
    assert!(graph.node(&ctx, &ns, &connected_a).await.unwrap().is_some());
    assert!(graph.node(&ctx, &ns, &connected_b).await.unwrap().is_some());
    assert!(graph.node(&ctx, &ns, &lonely).await.unwrap().is_none());
}

#[tokio::test]
#[ignore = "requires docker"]
async fn prune_older_than_drops_stale_edges_at_db_layer() {
    let (graph, _c) = boot().await;
    let ns = Namespace::new(TenantId::new("acme"));
    let ctx = ExecutionContext::new();
    let now = Utc::now();
    let a = graph.add_node(&ctx, &ns, "a".into()).await.unwrap();
    let b = graph.add_node(&ctx, &ns, "b".into()).await.unwrap();
    graph
        .add_edge(
            &ctx,
            &ns,
            &a,
            &b,
            "old".into(),
            now - Duration::seconds(120),
        )
        .await
        .unwrap();
    graph
        .add_edge(
            &ctx,
            &ns,
            &a,
            &b,
            "fresh".into(),
            now - Duration::seconds(5),
        )
        .await
        .unwrap();

    let removed = graph
        .prune_older_than(&ctx, &ns, std::time::Duration::from_secs(60))
        .await
        .unwrap();
    assert_eq!(removed, 1);

    // Edge-only sweep — both nodes still present.
    assert!(graph.node(&ctx, &ns, &a).await.unwrap().is_some());
    assert!(graph.node(&ctx, &ns, &b).await.unwrap().is_some());
    let outgoing = graph
        .neighbors(&ctx, &ns, &a, Direction::Outgoing)
        .await
        .unwrap();
    assert_eq!(outgoing.len(), 1);
    assert_eq!(outgoing[0].2, "fresh");
}

// Suppress unused-import warning.
#[allow(dead_code)]
fn _silence(_p: Arc<()>) {}
