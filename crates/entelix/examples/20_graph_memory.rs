//! `20_graph_memory` — typed-node, timestamped-edge knowledge graph.
//!
//! Build: `cargo build --example 20_graph_memory -p entelix`
//! Run:   `cargo run   --example 20_graph_memory -p entelix`
//!
//! Demonstrates the first-class `GraphMemory<N, E>` tier — distinct
//! from the five `Memory*` patterns over `Store<V>` because typed
//! traversal, neighbour expansion, and shortest-path BFS cannot be
//! composed atop a flat KV.
//!
//! Walks through the four canonical knowledge-graph operations:
//! - `add_node` / `add_edge` — populate
//! - `neighbors` — one-hop expansion
//! - `traverse(Direction::Both, depth=2)` — BFS with depth cap
//! - `find_path` — shortest unweighted path
//!
//! Backend is `InMemoryGraphMemory<String, String>`. Companion
//! crate `entelix-graphmemory-pg` swaps in a Postgres backend with
//! row-level security and `WITH RECURSIVE` BFS — same trait, no
//! caller change.

#![allow(clippy::print_stdout)]

use std::sync::Arc;

use chrono::{TimeZone, Utc};
use entelix::{
    Direction, ExecutionContext, GraphMemory, InMemoryGraphMemory, Namespace, NodeId, Result,
    TenantId,
};

#[tokio::main]
async fn main() -> Result<()> {
    let graph: Arc<dyn GraphMemory<String, String>> = Arc::new(InMemoryGraphMemory::new());
    let ctx = ExecutionContext::new();
    let ns = Namespace::new(TenantId::new("acme")).with_scope("knowledge");

    // ── Insert five entity nodes ────────────────────────────────
    let alice = graph.add_node(&ctx, &ns, "Alice".to_owned()).await?;
    let bob = graph.add_node(&ctx, &ns, "Bob".to_owned()).await?;
    let carol = graph.add_node(&ctx, &ns, "Carol".to_owned()).await?;
    let dave = graph.add_node(&ctx, &ns, "Dave".to_owned()).await?;
    let acme = graph.add_node(&ctx, &ns, "Acme".to_owned()).await?;

    // Deterministic timestamps so the example output is stable in CI.
    let t0 = Utc.with_ymd_and_hms(2026, 1, 1, 9, 0, 0).unwrap();

    // ── Wire relationships ──────────────────────────────────────
    // Alice ──knows──→ Bob ──reports_to──→ Carol ──reports_to──→ Dave
    //   └──works_at──→ Acme ←──works_at── Carol
    graph
        .add_edge(&ctx, &ns, &alice, &bob, "knows".to_owned(), t0)
        .await?;
    graph
        .add_edge(&ctx, &ns, &bob, &carol, "reports_to".to_owned(), t0)
        .await?;
    graph
        .add_edge(&ctx, &ns, &carol, &dave, "reports_to".to_owned(), t0)
        .await?;
    graph
        .add_edge(&ctx, &ns, &alice, &acme, "works_at".to_owned(), t0)
        .await?;
    graph
        .add_edge(&ctx, &ns, &carol, &acme, "works_at".to_owned(), t0)
        .await?;

    // ── 1) Neighbours: one-hop outgoing ─────────────────────────
    let alice_out = graph.neighbors(&ctx, &ns, &alice, Direction::Outgoing).await?;
    println!("=== one-hop outgoing from Alice ===");
    for (_id, target, edge) in &alice_out {
        let target_name = graph.node(&ctx, &ns, target).await?.unwrap_or_default();
        println!("  Alice --{edge}--> {target_name}");
    }

    // ── 2) BFS traversal: 2 hops, undirected ────────────────────
    let reached = graph
        .traverse(&ctx, &ns, &alice, Direction::Both, 2)
        .await?;
    println!("\n=== BFS up to 2 hops from Alice (Direction::Both) ===");
    for hop in &reached {
        let from = graph.node(&ctx, &ns, &hop.from).await?.unwrap_or_default();
        let to = graph.node(&ctx, &ns, &hop.to).await?.unwrap_or_default();
        println!("  {from} --{edge}--> {to}", edge = hop.edge);
    }

    // ── 3) Shortest path: Alice → Dave ──────────────────────────
    let path = graph
        .find_path(&ctx, &ns, &alice, &dave, Direction::Outgoing, 5)
        .await?;
    println!("\n=== shortest outgoing path Alice → Dave ===");
    print_path(graph.as_ref(), &ctx, &ns, &path).await?;

    // ── 4) No-path case: Dave → Alice (graph is a DAG outgoing) ─
    let no_path = graph
        .find_path(&ctx, &ns, &dave, &alice, Direction::Outgoing, 5)
        .await?;
    println!("\n=== outgoing path Dave → Alice (none expected) ===");
    print_path(graph.as_ref(), &ctx, &ns, &no_path).await?;

    // Same query with `Direction::Both` resolves through the
    // `reports_to` chain in reverse plus the `knows` edge.
    let undirected = graph
        .find_path(&ctx, &ns, &dave, &alice, Direction::Both, 5)
        .await?;
    println!("\n=== undirected path Dave → Alice ===");
    print_path(graph.as_ref(), &ctx, &ns, &undirected).await?;

    Ok(())
}

async fn print_path(
    graph: &dyn GraphMemory<String, String>,
    ctx: &ExecutionContext,
    ns: &Namespace,
    path: &Option<Vec<entelix::GraphHop<String>>>,
) -> Result<()> {
    match path {
        None => println!("  (no path within depth cap)"),
        Some(hops) if hops.is_empty() => println!("  (already at destination)"),
        Some(hops) => {
            for hop in hops {
                let from = node_name(graph, ctx, ns, &hop.from).await?;
                let to = node_name(graph, ctx, ns, &hop.to).await?;
                println!("  {from} --{edge}--> {to}", edge = hop.edge);
            }
        }
    }
    Ok(())
}

async fn node_name(
    graph: &dyn GraphMemory<String, String>,
    ctx: &ExecutionContext,
    ns: &Namespace,
    id: &NodeId,
) -> Result<String> {
    Ok(graph.node(ctx, ns, id).await?.unwrap_or_default())
}
