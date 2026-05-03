//! `05_memory` — Tier-3 cross-thread memory demo.
//!
//! Build: `cargo build --example 05_memory -p entelix`
//! Run:   `cargo run --example 05_memory -p entelix`
//!
//! Demonstrates:
//! - `Namespace::new(tenant_id)` — F2 mitigation, tenant id mandatory.
//! - `InMemoryStore<V>` shared across two memory abstractions.
//! - `BufferMemory` — bounded conversation buffer.
//! - `EntityMemory` — entity → fact map.
//! - Tenant + scope isolation (two users see disjoint memories).
//!
//! No external API dependency — runs deterministically in CI.

#![allow(clippy::print_stdout)] // example output goes to the terminal

use std::collections::HashMap;
use std::sync::Arc;

use entelix::ir::Message;
use entelix::{
    BufferMemory, EntityMemory, EntityRecord, ExecutionContext, InMemoryStore, Namespace, Result,
    Store,
};

#[tokio::main]
async fn main() -> Result<()> {
    // One in-process store backs both kinds of memory. Persistent
    // backends swap InMemoryStore for a Postgres / Redis impl without
    // changing the higher-level types.
    let buffer_store: Arc<dyn Store<Vec<Message>>> = Arc::new(InMemoryStore::<Vec<Message>>::new());
    let entity_store: Arc<dyn Store<HashMap<String, EntityRecord>>> =
        Arc::new(InMemoryStore::<HashMap<String, EntityRecord>>::new());

    let ctx = ExecutionContext::new();

    // Two users sharing the same tenant but distinct scopes — F2 in
    // action: structurally impossible to read across users.
    let alice_buf = BufferMemory::new(buffer_store.clone(), namespace_for("alice"), 5);
    let bob_buf = BufferMemory::new(buffer_store, namespace_for("bob"), 5);

    let alice_entities = EntityMemory::new(entity_store.clone(), namespace_for("alice"));
    let bob_entities = EntityMemory::new(entity_store, namespace_for("bob"));

    // Record facts about each user.
    alice_entities
        .set_entity(&ctx, "preference", "vegetarian")
        .await?;
    alice_entities.set_entity(&ctx, "language", "ko").await?;
    bob_entities
        .set_entity(&ctx, "preference", "spicy food")
        .await?;
    bob_entities.set_entity(&ctx, "language", "en").await?;

    // Simulate two concurrent conversations — Alice asks 7 things, only
    // the last 5 stay in her buffer. Bob has only one turn.
    for i in 1..=7 {
        alice_buf
            .append(&ctx, Message::user(format!("alice turn {i}")))
            .await?;
    }
    bob_buf.append(&ctx, Message::user("hi from bob")).await?;

    println!("=== Alice's view ===");
    println!(
        "buffer ({} retained, max 5):",
        alice_buf.messages(&ctx).await?.len()
    );
    for msg in alice_buf.messages(&ctx).await? {
        if let Some(entelix::ir::ContentPart::Text { text, .. }) = msg.content.first() {
            println!("  - {text}");
        }
    }
    println!("entities:");
    for (k, v) in alice_entities.all(&ctx).await? {
        println!("  - {k}: {v}");
    }

    println!();
    println!("=== Bob's view ===");
    println!(
        "buffer ({} retained, max 5):",
        bob_buf.messages(&ctx).await?.len()
    );
    for msg in bob_buf.messages(&ctx).await? {
        if let Some(entelix::ir::ContentPart::Text { text, .. }) = msg.content.first() {
            println!("  - {text}");
        }
    }
    println!("entities:");
    for (k, v) in bob_entities.all(&ctx).await? {
        println!("  - {k}: {v}");
    }

    println!();
    println!("=== F2 verification ===");
    // Alice's namespace lookup for an arbitrary key returns None — there's
    // no API path from `alice_entities` into Bob's data. The Namespace
    // type makes cross-tenant reads structurally impossible.
    let across_users = alice_entities.entity(&ctx, "bob-only-key").await?;
    println!("alice's view of 'bob-only-key' (no path across users): {across_users:?}");
    Ok(())
}

fn namespace_for(user: &str) -> Namespace {
    Namespace::new("tenant-acme")
        .with_scope("agent-helper")
        .with_scope(format!("user-{user}"))
}
