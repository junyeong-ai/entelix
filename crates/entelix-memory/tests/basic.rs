//! `Namespace` + `Store<V>` + `InMemoryStore` + `BufferMemory` tests.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::needless_borrows_for_generic_args
)]

use std::sync::Arc;

use entelix_core::ExecutionContext;
use entelix_core::Result;
use entelix_core::ir::Message;
use entelix_memory::{BufferMemory, Document, InMemoryStore, Namespace, Store};

// ── Namespace ──────────────────────────────────────────────────────────────

#[test]
fn namespace_render_with_no_scope_is_just_tenant() {
    let ns = Namespace::new("tenant-A");
    assert_eq!(ns.tenant_id(), "tenant-A");
    assert!(ns.scope().is_empty());
    assert_eq!(ns.render(), "tenant-A");
}

#[test]
fn namespace_render_joins_scope_with_colons() {
    let ns = Namespace::new("tenant-A")
        .with_scope("agent-1")
        .with_scope("conversation-42");
    assert_eq!(ns.render(), "tenant-A:agent-1:conversation-42");
    assert_eq!(
        ns.scope(),
        &["agent-1".to_owned(), "conversation-42".to_owned()]
    );
}

#[test]
fn namespaces_are_distinct_when_tenant_differs() {
    let a = Namespace::new("alpha").with_scope("x");
    let b = Namespace::new("beta").with_scope("x");
    assert_ne!(a.render(), b.render());
}

// ── InMemoryStore ────────────────────────────────────────────────────────────

#[tokio::test]
async fn memory_store_put_get_delete_round_trip() -> Result<()> {
    let store = InMemoryStore::<i64>::new();
    let ns = Namespace::new("t");
    let ctx = ExecutionContext::new();
    store.put(&ctx, &ns, "k1", 42).await?;
    assert_eq!(store.get(&ctx, &ns, "k1").await?, Some(42));
    store.delete(&ctx, &ns, "k1").await?;
    assert_eq!(store.get(&ctx, &ns, "k1").await?, None);
    Ok(())
}

#[tokio::test]
async fn memory_store_namespaces_are_isolated() -> Result<()> {
    let store = InMemoryStore::<i64>::new();
    let alpha = Namespace::new("alpha");
    let beta = Namespace::new("beta");
    let ctx = ExecutionContext::new();

    store.put(&ctx, &alpha, "k", 1).await?;
    store.put(&ctx, &beta, "k", 2).await?;

    assert_eq!(store.get(&ctx, &alpha, "k").await?, Some(1));
    assert_eq!(store.get(&ctx, &beta, "k").await?, Some(2));
    assert_eq!(store.total_entries(), 2);
    Ok(())
}

#[tokio::test]
async fn memory_store_scope_segments_isolate_namespaces() -> Result<()> {
    let store = InMemoryStore::<String>::new();
    let agent_a = Namespace::new("tenant").with_scope("agent-A");
    let agent_b = Namespace::new("tenant").with_scope("agent-B");
    let ctx = ExecutionContext::new();

    store.put(&ctx, &agent_a, "name", "Alice".into()).await?;
    store.put(&ctx, &agent_b, "name", "Bob".into()).await?;

    assert_eq!(
        store.get(&ctx, &agent_a, "name").await?.as_deref(),
        Some("Alice")
    );
    assert_eq!(
        store.get(&ctx, &agent_b, "name").await?.as_deref(),
        Some("Bob")
    );
    Ok(())
}

#[tokio::test]
async fn memory_store_list_with_prefix_filters_correctly() -> Result<()> {
    let store = InMemoryStore::<i32>::new();
    let ns = Namespace::new("t");
    let ctx = ExecutionContext::new();
    store.put(&ctx, &ns, "user.alice", 1).await?;
    store.put(&ctx, &ns, "user.bob", 2).await?;
    store.put(&ctx, &ns, "agent.helper", 3).await?;

    let mut user_keys = store.list(&ctx, &ns, Some("user.")).await?;
    user_keys.sort();
    assert_eq!(
        user_keys,
        vec!["user.alice".to_owned(), "user.bob".to_owned()]
    );

    let all = store.list(&ctx, &ns, None).await?;
    assert_eq!(all.len(), 3);
    Ok(())
}

#[tokio::test]
async fn memory_store_delete_is_idempotent() -> Result<()> {
    let store = InMemoryStore::<i32>::new();
    let ns = Namespace::new("t");
    let ctx = ExecutionContext::new();
    store.delete(&ctx, &ns, "missing").await?; // no panic
    Ok(())
}

#[tokio::test]
async fn memory_store_dyn_dispatch_works_via_arc_store() -> Result<()> {
    let store: Arc<dyn Store<String>> = Arc::new(InMemoryStore::<String>::new());
    let ns = Namespace::new("t");
    let ctx = ExecutionContext::new();
    store.put(&ctx, &ns, "k", "v".into()).await?;
    assert_eq!(store.get(&ctx, &ns, "k").await?.as_deref(), Some("v"));
    Ok(())
}

// ── BufferMemory ───────────────────────────────────────────────────────────

#[tokio::test]
async fn buffer_memory_appends_and_returns_messages() -> Result<()> {
    let store: Arc<dyn Store<Vec<Message>>> = Arc::new(InMemoryStore::<Vec<Message>>::new());
    let buf = BufferMemory::new(store, Namespace::new("tenant"), 10);
    let ctx = ExecutionContext::new();

    buf.append(&ctx, Message::user("hi")).await?;
    buf.append(&ctx, Message::assistant("hello")).await?;

    let messages = buf.messages(&ctx).await?;
    assert_eq!(messages.len(), 2);
    Ok(())
}

#[tokio::test]
async fn buffer_memory_drops_oldest_when_over_capacity() -> Result<()> {
    let store: Arc<dyn Store<Vec<Message>>> = Arc::new(InMemoryStore::<Vec<Message>>::new());
    let buf = BufferMemory::new(store, Namespace::new("tenant"), 3);
    let ctx = ExecutionContext::new();

    for i in 0..5 {
        buf.append(&ctx, Message::user(&format!("turn {i}")))
            .await?;
    }

    let messages = buf.messages(&ctx).await?;
    assert_eq!(messages.len(), 3);
    // First two ("turn 0", "turn 1") should be dropped.
    let texts: Vec<String> = messages
        .iter()
        .filter_map(|m| match m.content.first() {
            Some(entelix_core::ir::ContentPart::Text { text, .. }) => Some(text.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(texts, vec!["turn 2", "turn 3", "turn 4"]);
    Ok(())
}

#[tokio::test]
async fn buffer_memory_clear_resets_to_empty() -> Result<()> {
    let store: Arc<dyn Store<Vec<Message>>> = Arc::new(InMemoryStore::<Vec<Message>>::new());
    let buf = BufferMemory::new(store, Namespace::new("tenant"), 10);
    let ctx = ExecutionContext::new();

    buf.append(&ctx, Message::user("hi")).await?;
    assert_eq!(buf.messages(&ctx).await?.len(), 1);
    buf.clear(&ctx).await?;
    assert_eq!(buf.messages(&ctx).await?.len(), 0);
    Ok(())
}

#[tokio::test]
async fn buffer_memory_namespaces_are_isolated() -> Result<()> {
    let store: Arc<dyn Store<Vec<Message>>> = Arc::new(InMemoryStore::<Vec<Message>>::new());
    let buf_a = BufferMemory::new(store.clone(), Namespace::new("tenant").with_scope("A"), 10);
    let buf_b = BufferMemory::new(store, Namespace::new("tenant").with_scope("B"), 10);
    let ctx = ExecutionContext::new();

    buf_a.append(&ctx, Message::user("from A")).await?;
    buf_b.append(&ctx, Message::user("from B")).await?;

    assert_eq!(buf_a.messages(&ctx).await?.len(), 1);
    assert_eq!(buf_b.messages(&ctx).await?.len(), 1);
    Ok(())
}

// ── Document smoke ─────────────────────────────────────────────────────────

#[test]
fn document_builders_set_fields() {
    let d = Document::new("hello").with_metadata(serde_json::json!({"src": "test"}));
    assert_eq!(d.content, "hello");
    assert_eq!(d.metadata["src"], "test");
    assert!(d.score.is_none());
}
