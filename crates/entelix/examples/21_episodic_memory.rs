//! `21_episodic_memory` — append-only, time-ordered domain log.
//!
//! Build: `cargo build --example 21_episodic_memory -p entelix`
//! Run:   `cargo run   --example 21_episodic_memory -p entelix`
//!
//! Demonstrates the `EpisodicMemory<V>` pattern over `Store<V>` —
//! the time axis the other four memory patterns don't address
//! (`BufferMemory` is a sliding window, `SummaryMemory` is one
//! string, `EntityMemory` is current-fact-only, `SemanticMemory`
//! is similarity-keyed).
//!
//! Walks through the canonical episodic queries:
//! - `append_at` — write at a deterministic timestamp
//! - `recent(n)` — last N episodes
//! - `range(start, end)` — episodes in a closed interval
//! - `since(start)` — episodes from a checkpoint onward
//! - `prune_older_than(ttl)` — TTL sweep with a count
//!
//! Backend is `InMemoryStore<Vec<Episode<TaskEvent>>>`. The same
//! API drops onto a Postgres `Store` (companion crate) without
//! caller change.

#![allow(clippy::print_stdout)]

use std::sync::Arc;

use chrono::{Duration as ChronoDuration, Utc};
use entelix::{
    Episode, EpisodicMemory, ExecutionContext, InMemoryStore, Namespace, Result, Store, TenantId,
};
use serde::{Deserialize, Serialize};

/// Operator-domain payload — a per-task lifecycle event.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct TaskEvent {
    task: String,
    phase: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let store: Arc<dyn Store<Vec<Episode<TaskEvent>>>> = Arc::new(InMemoryStore::new());
    let ns = Namespace::new(TenantId::new("acme")).with_scope("worklog");
    let memory = EpisodicMemory::new(store, ns);
    let ctx = ExecutionContext::new();

    // Anchor every episode relative to `now` so the prune result is
    // deterministic regardless of when the example runs in CI.
    let now = Utc::now();
    let day = ChronoDuration::days(1);
    let t_minus_8 = now - day * 8;
    let t_minus_4 = now - day * 4;
    let t_minus_3 = now - day * 3;
    let t_minus_2 = now - day * 2;
    let t_minus_1 = now - day;

    memory.append_at(&ctx, evt("ingest", "started"), t_minus_8).await?;
    memory.append_at(&ctx, evt("ingest", "complete"), t_minus_8).await?;
    memory.append_at(&ctx, evt("classify", "started"), t_minus_4).await?;
    memory.append_at(&ctx, evt("classify", "complete"), t_minus_3).await?;
    memory.append_at(&ctx, evt("publish", "started"), t_minus_2).await?;
    memory.append_at(&ctx, evt("publish", "complete"), t_minus_1).await?;

    println!("=== total episodes ===");
    println!("  {}", memory.count(&ctx).await?);

    println!("\n=== recent(3) — last three in reverse chronological ===");
    for e in memory.recent(&ctx, 3).await? {
        print_episode(&e, now);
    }

    println!("\n=== range(t-4d, t-3d) — closed interval ===");
    for e in memory.range(&ctx, t_minus_4, t_minus_3).await? {
        print_episode(&e, now);
    }

    println!("\n=== since(t-2d) — checkpoint and forward ===");
    for e in memory.since(&ctx, t_minus_2).await? {
        print_episode(&e, now);
    }

    // 7-day TTL — drops the two `t-8d` ingest episodes, keeps the
    // four within the last week. Result is deterministic because
    // every fixture is anchored to `now`.
    let pruned = memory
        .prune_older_than(&ctx, std::time::Duration::from_secs(7 * 24 * 3_600))
        .await?;
    println!("\n=== prune_older_than(7 days) ===");
    println!("  pruned: {pruned}");
    println!("  remaining: {}", memory.count(&ctx).await?);

    Ok(())
}

fn evt(task: &str, phase: &str) -> TaskEvent {
    TaskEvent {
        task: task.to_owned(),
        phase: phase.to_owned(),
    }
}

fn print_episode(e: &Episode<TaskEvent>, now: chrono::DateTime<Utc>) {
    let days_back = (now - e.timestamp).num_days();
    println!(
        "  [t-{days_back}d] {task} → {phase}",
        task = e.payload.task,
        phase = e.payload.phase,
    );
}
