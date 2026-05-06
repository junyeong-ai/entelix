//! External `Compactor` implementations — proves the trait is
//! genuinely operator-extensible.
//!
//! Slice 109 sealed the `tool_call` / `tool_result` pair invariant
//! at the type level: only the internal grouping code can build a
//! [`ToolPair`]. But the seal initially extended too far —
//! [`CompactedHistory`]'s `turns` field was private with no public
//! constructor, so an external `Compactor` had no way to return a
//! value. ADR-0095 added [`CompactedHistory::group`] +
//! [`CompactedHistory::from_turns`] to restore extensibility while
//! keeping the pair invariant sealed at the [`ToolPair`] level.
//!
//! These regressions pin the contract:
//!
//! 1. An out-of-crate `Compactor` impl compiles and runs.
//! 2. It can filter / truncate turns and rebuild via `from_turns`.
//! 3. The `ToolPair`s the operator obtained from `group(...)`
//!    survive the rebuild — the pair invariant holds end-to-end.

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use chrono::Utc;
use entelix_core::Result;
use entelix_core::ir::{ContentPart, ToolResultContent};
use entelix_session::{CompactedHistory, Compactor, GraphEvent, Turn};

/// First-N compactor — keeps only the *first* `n` turns instead of
/// the last `n`. Demonstrates that an external strategy can
/// transform the grouped form and rebuild via `from_turns`.
struct FirstNCompactor {
    n: usize,
}

impl Compactor for FirstNCompactor {
    fn compact(&self, events: &[GraphEvent], _budget_chars: usize) -> Result<CompactedHistory> {
        let mut turns = CompactedHistory::group(events)?.turns().to_vec();
        turns.truncate(self.n);
        Ok(CompactedHistory::from_turns(turns))
    }
}

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

fn assistant_with_tool_use(text: &str, call_id: &str, name: &str) -> GraphEvent {
    GraphEvent::AssistantMessage {
        content: vec![
            ContentPart::text(text),
            ContentPart::ToolUse {
                id: call_id.to_owned(),
                name: name.to_owned(),
                input: serde_json::json!({"q": "x"}),
            },
        ],
        usage: None,
        timestamp: Utc::now(),
    }
}

fn tool_call(id: &str, name: &str) -> GraphEvent {
    GraphEvent::ToolCall {
        id: id.to_owned(),
        name: name.to_owned(),
        input: serde_json::json!({"q": "x"}),
        timestamp: Utc::now(),
    }
}

fn tool_result(id: &str, name: &str, text: &str) -> GraphEvent {
    GraphEvent::ToolResult {
        tool_use_id: id.to_owned(),
        name: name.to_owned(),
        content: ToolResultContent::Text(text.to_owned()),
        is_error: false,
        timestamp: Utc::now(),
    }
}

#[test]
fn external_compactor_can_construct_compacted_history() {
    let events = vec![
        user("first"),
        assistant("first reply"),
        user("second"),
        assistant("second reply"),
    ];

    let history = FirstNCompactor { n: 2 }.compact(&events, 0).unwrap();
    assert_eq!(history.len(), 2);
    let turns = history.turns();
    assert!(matches!(&turns[0], Turn::User { .. }));
    assert!(matches!(&turns[1], Turn::Assistant { .. }));
}

#[test]
fn external_compactor_passes_tool_pairs_through_unchanged() {
    let events = vec![
        user("query"),
        assistant_with_tool_use("checking", "call_1", "search"),
        tool_call("call_1", "search"),
        tool_result("call_1", "search", "hits"),
        assistant("done"),
    ];

    let history = FirstNCompactor { n: 2 }.compact(&events, 0).unwrap();
    assert_eq!(history.len(), 2);
    let turns = history.turns();
    assert!(matches!(&turns[0], Turn::User { .. }));
    let Turn::Assistant { tools, .. } = &turns[1] else {
        panic!("expected Turn::Assistant at index 1");
    };
    assert_eq!(tools.len(), 1, "the round-trip must survive the rebuild");
    assert_eq!(tools[0].id(), "call_1");
    assert_eq!(tools[0].name(), "search");
}
