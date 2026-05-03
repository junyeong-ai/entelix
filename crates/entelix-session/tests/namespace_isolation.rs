//! `InMemorySessionLog` cross-tenant isolation — invariant 11 / F2
//! reference test. Mirrors `entelix-graph::tests::checkpoint::
//! checkpointer_partitions_per_tenant` and the persistence-layer
//! `postgres_namespace_collision.rs` shape so the in-memory backend
//! and the durable backends share one cross-tenant contract.
//!
//! The trait doc on `SessionLog` claims:
//!
//! > Tenant-scoped: every method takes `tenant_id` (invariant 11)
//! > so cross-tenant reads / writes are structurally impossible.
//!
//! These tests pin that claim against the reference impl. A
//! regression that, say, keyed `inner` on `thread_id` alone would
//! fail here.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::items_after_statements
)]

use chrono::Utc;
use entelix_core::ThreadKey;
use entelix_core::ir::ContentPart;
use entelix_session::{GraphEvent, InMemorySessionLog, SessionLog};

fn user(text: &str) -> GraphEvent {
    GraphEvent::UserMessage {
        content: vec![ContentPart::text(text)],
        timestamp: Utc::now(),
    }
}

#[tokio::test]
async fn append_and_load_isolate_distinct_tenants_under_identical_thread() {
    // Same thread_id under two different tenants must produce two
    // independent histories — invariant 11 enforced by ThreadKey.
    let log = InMemorySessionLog::new();
    let alpha = ThreadKey::new("alpha", "conv-1");
    let bravo = ThreadKey::new("bravo", "conv-1");

    let _ = log
        .append(&alpha, &[user("alpha-1"), user("alpha-2")])
        .await
        .unwrap();
    let _ = log.append(&bravo, &[user("bravo-1")]).await.unwrap();

    let alpha_events = log.load_since(&alpha, 0).await.unwrap();
    let bravo_events = log.load_since(&bravo, 0).await.unwrap();
    assert_eq!(alpha_events.len(), 2, "alpha must see only its 2 events");
    assert_eq!(bravo_events.len(), 1, "bravo must see only its 1 event");

    // Spot-check the bodies — content correlates with tenant.
    fn first_text(e: &GraphEvent) -> &str {
        match e {
            GraphEvent::UserMessage { content, .. } => match &content[0] {
                ContentPart::Text { text, .. } => text.as_str(),
                _ => "",
            },
            _ => "",
        }
    }
    assert_eq!(first_text(&alpha_events[0]), "alpha-1");
    assert_eq!(first_text(&bravo_events[0]), "bravo-1");
}

#[tokio::test]
async fn archive_before_does_not_cross_tenants() {
    // Archiving up to watermark N for tenant alpha must not touch
    // bravo's events even when both share a thread_id and ordinals.
    let log = InMemorySessionLog::new();
    let alpha = ThreadKey::new("alpha", "conv-1");
    let bravo = ThreadKey::new("bravo", "conv-1");

    let _ = log
        .append(&alpha, &[user("a1"), user("a2"), user("a3")])
        .await
        .unwrap();
    let _ = log
        .append(&bravo, &[user("b1"), user("b2"), user("b3")])
        .await
        .unwrap();

    let archived = log.archive_before(&alpha, 2).await.unwrap();
    assert_eq!(archived, 2, "alpha must report 2 events archived");

    // Alpha sees only ordinals > 2.
    let alpha_remaining = log.load_since(&alpha, 0).await.unwrap();
    assert_eq!(alpha_remaining.len(), 1);

    // Bravo sees all 3 events untouched — the archive call did not
    // bleed across the tenant boundary.
    let bravo_remaining = log.load_since(&bravo, 0).await.unwrap();
    assert_eq!(bravo_remaining.len(), 3, "bravo's events must be untouched");
}

#[tokio::test]
async fn load_since_unknown_tenant_returns_empty_not_other_tenants_data() {
    // A tenant that has never appended must not accidentally see
    // any other tenant's events. The HashMap miss path is the
    // guard — a regression that fell back to "global" events on
    // miss would surface here.
    let log = InMemorySessionLog::new();
    let alpha = ThreadKey::new("alpha", "conv-1");
    let _ = log.append(&alpha, &[user("alpha-only")]).await.unwrap();

    let unknown = ThreadKey::new("ghost", "conv-1");
    let events = log.load_since(&unknown, 0).await.unwrap();
    assert!(
        events.is_empty(),
        "unknown tenant must see empty log, not alpha's events; got {events:?}"
    );
}

#[tokio::test]
async fn distinct_threads_within_one_tenant_are_also_isolated() {
    // Sanity check beyond the tenant axis: same tenant, two
    // thread_ids — also independent histories. Confirms ThreadKey
    // partitions on the full (tenant, thread) tuple, not just the
    // tenant prefix.
    let log = InMemorySessionLog::new();
    let conv1 = ThreadKey::new("alpha", "conv-1");
    let conv2 = ThreadKey::new("alpha", "conv-2");

    let _ = log.append(&conv1, &[user("c1-only")]).await.unwrap();
    let _ = log
        .append(&conv2, &[user("c2-only"), user("c2-also")])
        .await
        .unwrap();

    let conv1_events = log.load_since(&conv1, 0).await.unwrap();
    let conv2_events = log.load_since(&conv2, 0).await.unwrap();
    assert_eq!(conv1_events.len(), 1);
    assert_eq!(conv2_events.len(), 2);
}
