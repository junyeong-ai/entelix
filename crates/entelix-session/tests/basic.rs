//! `SessionGraph` + `GraphEvent` basic tests.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::items_after_statements
)]

use chrono::Utc;
use entelix_core::ir::{ContentPart, ModelWarning, Role, ToolResultContent, Usage};
use entelix_session::{GraphEvent, SessionGraph};

fn user(text: &str) -> GraphEvent {
    GraphEvent::UserMessage {
        content: vec![ContentPart::text(text)],
        timestamp: Utc::now(),
    }
}

fn assistant(text: &str) -> GraphEvent {
    GraphEvent::AssistantMessage {
        content: vec![ContentPart::text(text)],
        usage: None,
        timestamp: Utc::now(),
    }
}

#[test]
fn append_assigns_monotonic_indices() {
    let mut s = SessionGraph::new("t-1");
    let i0 = s.append(user("hi"));
    let i1 = s.append(assistant("hello"));
    assert_eq!(i0, 0);
    assert_eq!(i1, 1);
    assert_eq!(s.len(), 2);
    assert!(!s.is_empty());
}

#[test]
fn events_since_returns_tail_slice() {
    let mut s = SessionGraph::new("t-2");
    s.append(user("a"));
    s.append(user("b"));
    s.append(user("c"));
    assert_eq!(s.events_since(0).len(), 3);
    assert_eq!(s.events_since(1).len(), 2);
    assert_eq!(s.events_since(3).len(), 0);
    assert_eq!(s.events_since(99).len(), 0);
}

#[test]
fn current_branch_messages_replays_user_assistant_pairs() {
    let mut s = SessionGraph::new("t-3");
    s.append(user("hi"));
    s.append(assistant("hello"));
    s.append(user("how are you?"));
    s.append(assistant("doing well"));

    let messages = s.current_branch_messages();
    assert_eq!(messages.len(), 4);
    assert!(matches!(messages[0].role, Role::User));
    assert!(matches!(messages[1].role, Role::Assistant));
    assert!(matches!(messages[2].role, Role::User));
    assert!(matches!(messages[3].role, Role::Assistant));
}

#[test]
fn current_branch_messages_includes_tool_results_as_tool_role() {
    let mut s = SessionGraph::new("t-tools");
    s.append(user("calc 2+2"));
    s.append(GraphEvent::AssistantMessage {
        content: vec![ContentPart::ToolUse {
            id: "u1".into(),
            name: "calc".into(),
            input: serde_json::json!({"expr": "2+2"}),
        }],
        usage: None,
        timestamp: Utc::now(),
    });
    s.append(GraphEvent::ToolCall {
        id: "u1".into(),
        name: "calc".into(),
        input: serde_json::json!({"expr": "2+2"}),
        timestamp: Utc::now(),
    });
    s.append(GraphEvent::ToolResult {
        tool_use_id: "u1".into(),
        name: "calc".into(),
        content: ToolResultContent::Text("4".into()),
        is_error: false,
        timestamp: Utc::now(),
    });

    let messages = s.current_branch_messages();
    assert_eq!(messages.len(), 3); // user, assistant(tool_use), tool(tool_result)
    assert!(matches!(messages[2].role, Role::Tool));
}

#[test]
fn current_branch_messages_skips_branch_and_warning_and_checkpoint_events() {
    let mut s = SessionGraph::new("t-skip");
    s.append(user("hi"));
    s.append(GraphEvent::Warning {
        warning: ModelWarning::LossyEncode {
            field: "x".into(),
            detail: "y".into(),
        },
        timestamp: Utc::now(),
    });
    s.append(GraphEvent::CheckpointMarker {
        checkpoint_id: "cp-1".into(),
        thread_id: "t-skip".into(),
        timestamp: Utc::now(),
    });
    s.append(GraphEvent::BranchCreated {
        branch_id: "b-1".into(),
        parent_event: 0,
        timestamp: Utc::now(),
    });
    s.append(assistant("ok"));

    let messages = s.current_branch_messages();
    // Only user + assistant produce messages; the other 3 events are
    // metadata.
    assert_eq!(messages.len(), 2);
}

#[test]
fn fork_copies_events_up_to_branch_point_and_records_branch_event() {
    let mut s = SessionGraph::new("t-parent");
    s.append(user("hi"));
    s.append(assistant("hello"));
    s.append(user("turn 2"));

    let child = s.fork(1, "t-child").unwrap();

    // Parent now has the original 3 events + 1 BranchCreated marker.
    assert_eq!(s.len(), 4);
    let last = s.events.last().unwrap();
    matches!(
        last,
        GraphEvent::BranchCreated {
            parent_event: 1,
            ..
        }
    );

    // Child has events [..=1] = first 2.
    assert_eq!(child.thread_id, "t-child");
    assert_eq!(child.len(), 2);
    assert_eq!(child.archival_watermark(), 0);
}

#[test]
fn fork_out_of_range_returns_none() {
    let mut s = SessionGraph::new("t-fork-oor");
    s.append(user("hi"));
    assert!(s.fork(99, "t-child").is_none());
    assert_eq!(s.len(), 1); // no branch event appended
}

#[test]
fn archival_watermark_is_monotonic() {
    let mut s = SessionGraph::new("t-arch");
    for i in 0..5 {
        s.append(user(&format!("turn {i}")));
    }
    assert_eq!(s.archival_watermark(), 0);

    s.archive_before(2);
    assert_eq!(s.archival_watermark(), 2);

    // Lowering is rejected.
    s.archive_before(1);
    assert_eq!(s.archival_watermark(), 2);

    // Above len() is rejected.
    s.archive_before(99);
    assert_eq!(s.archival_watermark(), 2);

    // Same value (not strictly higher) is rejected.
    s.archive_before(2);
    assert_eq!(s.archival_watermark(), 2);

    // Bumping forward works.
    s.archive_before(4);
    assert_eq!(s.archival_watermark(), 4);
}

#[test]
fn branch_events_iterator_filters_correctly() {
    let mut s = SessionGraph::new("t-bev");
    s.append(user("hi"));
    s.append(assistant("hello"));
    let _ = s.fork(0, "child-A");
    let _ = s.fork(1, "child-B");

    let branches: Vec<_> = s.branch_events().collect();
    assert_eq!(branches.len(), 2);
    assert_eq!(branches[0].0, "child-A");
    assert_eq!(branches[0].1, 0);
    assert_eq!(branches[1].0, "child-B");
    assert_eq!(branches[1].1, 1);
}

#[test]
fn replay_into_folds_every_event_in_order() {
    let mut s = SessionGraph::new("t-replay");
    s.append(user("a"));
    s.append(assistant("b"));
    s.append(user("c"));
    s.append(assistant("d"));

    // Toy state: collect just the user-message texts, oldest-first.
    let texts: Vec<String> = s.replay_into(Vec::new(), |acc, ev| {
        if let GraphEvent::UserMessage { content, .. } = ev
            && let Some(ContentPart::Text { text, .. }) = content.first()
        {
            acc.push(text.clone());
        }
    });
    assert_eq!(texts, vec!["a".to_owned(), "c".to_owned()]);
}

#[test]
fn replay_into_observes_thinking_rate_limit_cancelled_error_interrupt_variants() {
    use entelix_core::rate_limit::RateLimitSnapshot;
    let mut s = SessionGraph::new("t-new-variants");
    s.append(GraphEvent::ThinkingDelta {
        text: "let me think".into(),
        signature: Some("sig-1".into()),
        timestamp: Utc::now(),
    });
    s.append(GraphEvent::RateLimit {
        snapshot: RateLimitSnapshot::default(),
        timestamp: Utc::now(),
    });
    s.append(GraphEvent::Interrupt {
        payload: serde_json::json!({"reason": "human review"}),
        timestamp: Utc::now(),
    });
    s.append(GraphEvent::Cancelled {
        reason: "deadline elapsed".into(),
        timestamp: Utc::now(),
    });
    s.append(GraphEvent::Error {
        class: "provider".into(),
        message: "503 from upstream".into(),
        timestamp: Utc::now(),
    });
    assert_eq!(s.len(), 5);
    // Each variant must be reachable through `timestamp()`.
    for event in s.events_since(0) {
        let _ts = event.timestamp();
    }

    // Replay aggregates a count of every new variant.
    #[derive(Default)]
    struct Counts {
        thinking: usize,
        rate_limit: usize,
        interrupt: usize,
        cancelled: usize,
        error: usize,
    }
    let counts = s.replay_into(Counts::default(), |c, ev| match ev {
        GraphEvent::ThinkingDelta { .. } => c.thinking += 1,
        GraphEvent::RateLimit { .. } => c.rate_limit += 1,
        GraphEvent::Interrupt { .. } => c.interrupt += 1,
        GraphEvent::Cancelled { .. } => c.cancelled += 1,
        GraphEvent::Error { .. } => c.error += 1,
        _ => {}
    });
    assert_eq!(counts.thinking, 1);
    assert_eq!(counts.rate_limit, 1);
    assert_eq!(counts.interrupt, 1);
    assert_eq!(counts.cancelled, 1);
    assert_eq!(counts.error, 1);
}

#[test]
fn graph_event_serde_roundtrip() {
    let event = GraphEvent::AssistantMessage {
        content: vec![ContentPart::text("hi")],
        usage: Some(Usage::new(10, 5)),
        timestamp: Utc::now(),
    };
    let json = serde_json::to_string(&event).unwrap();
    let back: GraphEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(back, event);
}

#[test]
fn timestamp_accessor_reaches_every_variant() {
    let warning_event = GraphEvent::Warning {
        warning: ModelWarning::UnknownStopReason {
            raw: "weird".into(),
        },
        timestamp: Utc::now(),
    };
    let _ts = warning_event.timestamp(); // smoke — should not panic
    let user_event = user("hi");
    let _ts2 = user_event.timestamp();
}

#[test]
fn session_graph_serde_roundtrip() {
    let mut s = SessionGraph::new("t-serde");
    s.append(user("hi"));
    s.append(assistant("hello"));
    s.archive_before(1);

    let json = serde_json::to_string(&s).unwrap();
    let back: SessionGraph = serde_json::from_str(&json).unwrap();
    assert_eq!(back.thread_id, s.thread_id);
    assert_eq!(back.len(), s.len());
    assert_eq!(back.archival_watermark(), s.archival_watermark());
}

#[tokio::test]
async fn memory_session_log_load_since_honors_archival_watermark() {
    // Cross-backend behaviour test: Postgres deletes archived rows
    // and Redis trims them. The in-memory backend retains them in
    // its Vec for replay tooling, but `load_since` MUST hide the
    // archived range so callers see the same event surface across
    // all three backends. Without this, a caller archiving and
    // then reading from cursor 0 sees archived events on
    // InMemorySessionLog but not on Postgres / Redis — silently
    // breaking cross-backend integration tests.
    use entelix_core::ThreadKey;
    use entelix_session::{InMemorySessionLog, SessionLog};

    let log = InMemorySessionLog::new();
    let key = ThreadKey::new(TenantId::new("t"), "thread-1");

    let events: Vec<GraphEvent> = (0..7).map(|i| user(&format!("msg {i}"))).collect();
    let head = log.append(&key, &events).await.unwrap();
    assert_eq!(head, 7, "append should return ordinal 7 for 7 events");

    // Archive the first 3 events. Postgres deletes seq <= 3;
    // Redis LTRIMs the first 3; in-memory bumps the watermark.
    let archived = log.archive_before(&key, 3).await.unwrap();
    assert!(archived >= 1, "archive must report progress");

    // load_since(0) — caller asking for everything available.
    // Postgres returns events with seq > 0, but seq 1..=3 were
    // deleted, so events 4..=7 (4 events). Redis: same. In-memory
    // must match.
    let live = log.load_since(&key, 0).await.unwrap();
    assert_eq!(
        live.len(),
        4,
        "load_since(0) after archive(3) of 7 events must return 4 (ordinals 4..=7), \
         not 7 — the archival watermark applies regardless of the cursor value"
    );

    // load_since(5) — caller resuming mid-stream. Both backends
    // return events 6..=7 (2 events). In-memory must match.
    let mid = log.load_since(&key, 5).await.unwrap();
    assert_eq!(mid.len(), 2, "load_since(5) must return ordinals 6 and 7");

    // load_since(7) — at head, nothing newer.
    let none = log.load_since(&key, 7).await.unwrap();
    assert!(none.is_empty(), "load_since(7) at head must return empty");
}

#[tokio::test]
async fn memory_session_log_archive_before_is_monotonic() {
    use entelix_core::ThreadKey;
    use entelix_session::{InMemorySessionLog, SessionLog};

    let log = InMemorySessionLog::new();
    let key = ThreadKey::new(TenantId::new("t"), "thread-2");
    log.append(
        &key,
        &(0..5).map(|i| user(&format!("m{i}"))).collect::<Vec<_>>(),
    )
    .await
    .unwrap();

    let first = log.archive_before(&key, 3).await.unwrap();
    assert!(first >= 1);
    // Lowering the watermark must be a no-op.
    let second = log.archive_before(&key, 2).await.unwrap();
    assert_eq!(
        second, 0,
        "archive_before below current watermark must be a no-op"
    );
    // load_since must still hide archived events after the no-op.
    let live = log.load_since(&key, 0).await.unwrap();
    assert_eq!(live.len(), 2, "watermark must not regress to 2");
}
