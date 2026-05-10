//! `SessionGraph` — append-only durable audit log for one conversation
//! thread (invariant 1: session is event `SSoT`).
//!
//! Distinct from the runtime `EventBus` (`entelix-core::events`) per F1
//! mitigation: `EventBus` is fire-and-forget broadcast for hooks /
//! observability, `SessionGraph` is the durable replay source.

use chrono::Utc;
use entelix_core::ir::{ContentPart, Message, Role};
use serde::{Deserialize, Serialize};

use crate::event::GraphEvent;

/// Append-only event log for a single conversation thread.
///
/// `events` is the only first-class data — every higher-level view
/// (current branch messages, checkpoint markers, warning summaries) is
/// derived. Entries before `archival_watermark` may have been moved to
/// cold storage; consumers should treat indices `< archival_watermark`
/// as opaque.
///
/// `#[non_exhaustive]` so internal bookkeeping fields (e.g.
/// `schema_version`, archival metadata) can be added without
/// breaking downstream `SessionLog` impls. Construct via
/// `SessionGraph::new(thread_id)`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SessionGraph {
    /// Conversation identifier this log belongs to.
    pub thread_id: String,
    /// All events in append order.
    pub events: Vec<GraphEvent>,
    /// Index above which events are guaranteed to still live in `events`.
    /// Below the watermark, entries may have been pruned.
    archival_watermark: usize,
}

impl SessionGraph {
    /// Empty session bound to `thread_id`.
    pub fn new(thread_id: impl Into<String>) -> Self {
        Self {
            thread_id: thread_id.into(),
            events: Vec::new(),
            archival_watermark: 0,
        }
    }

    /// Append one event to the log. Returns the index assigned to it.
    pub fn append(&mut self, event: GraphEvent) -> usize {
        self.events.push(event);
        self.events.len().saturating_sub(1)
    }

    /// Number of events currently in memory (excludes archived ranges).
    pub const fn len(&self) -> usize {
        self.events.len()
    }

    /// True when no events have been appended.
    pub const fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Borrow the slice of events at indices `>= cursor`.
    pub fn events_since(&self, cursor: usize) -> &[GraphEvent] {
        self.events.get(cursor..).unwrap_or(&[])
    }

    /// Derive a value of arbitrary type `S` by folding every event in
    /// the log through `reducer`, oldest-first. The closure is called
    /// once per event with `(&mut state, &GraphEvent)` so it can both
    /// inspect and mutate the accumulator.
    ///
    /// This is the closure form of "the audit log is the source of
    /// truth": every domain that needs a derived view (message
    /// transcript, error count per turn, custom analytics) walks the
    /// log directly through this method rather than maintaining a
    /// parallel projection that could diverge.
    pub fn replay_into<S, R>(&self, initial: S, mut reducer: R) -> S
    where
        R: FnMut(&mut S, &GraphEvent),
    {
        let mut state = initial;
        for event in &self.events {
            reducer(&mut state, event);
        }
        state
    }

    /// Render the conversation as a `Vec<Message>` suitable for
    /// `ChatModel::complete`. Only `UserMessage` and `AssistantMessage`
    /// events contribute; tool events live inside the assistant's content
    /// blocks, which the codec handles separately.
    pub fn current_branch_messages(&self) -> Vec<Message> {
        let mut out = Vec::new();
        for event in &self.events {
            match event {
                GraphEvent::UserMessage { content, .. } => {
                    out.push(Message::new(Role::User, content.clone()));
                }
                GraphEvent::AssistantMessage { content, .. } => {
                    out.push(Message::new(Role::Assistant, content.clone()));
                }
                GraphEvent::ToolResult {
                    tool_use_id,
                    name,
                    content,
                    is_error,
                    ..
                } => out.push(Message::new(
                    Role::Tool,
                    vec![ContentPart::ToolResult {
                        tool_use_id: tool_use_id.clone(),
                        name: name.clone(),
                        content: content.clone(),
                        is_error: *is_error,
                        cache_control: None,
                        provider_echoes: Vec::new(),
                    }],
                )),
                _ => {}
            }
        }
        out
    }

    /// Fork: produce a fresh session whose events are a copy of this
    /// session's events at indices `0..=branch_at`, bound to `new_thread_id`.
    /// A `BranchCreated` event is appended **to the parent** to record the
    /// fork point.
    ///
    /// Returns `None` if `branch_at` is out of range.
    pub fn fork(&mut self, branch_at: usize, new_thread_id: impl Into<String>) -> Option<Self> {
        let cloned_events = match self.events.get(..=branch_at) {
            Some(slice) => slice.to_vec(),
            None => return None,
        };
        let new_thread_id = new_thread_id.into();
        self.append(GraphEvent::BranchCreated {
            branch_id: new_thread_id.clone(),
            parent_event: branch_at,
            timestamp: Utc::now(),
        });
        Some(Self {
            thread_id: new_thread_id,
            events: cloned_events,
            archival_watermark: 0,
        })
    }

    /// Mark events at indices `< watermark` as archived. Does not actually
    /// drop them from `events` — a persistence backend may purge them
    /// during cold-storage migration. Watermarks are monotonic and
    /// silently ignore non-advancing values: a `watermark` ≤ the
    /// current archival point or beyond `events.len()` is a no-op
    /// without error, mirroring [`crate::SessionLog::archive_before`].
    pub const fn archive_before(&mut self, watermark: usize) {
        if watermark > self.archival_watermark && watermark <= self.events.len() {
            self.archival_watermark = watermark;
        }
    }

    /// Effective archival cut-off.
    pub const fn archival_watermark(&self) -> usize {
        self.archival_watermark
    }

    /// Convenience iterator over `BranchCreated` events for tooling that
    /// renders branch trees.
    pub fn branch_events(&self) -> impl Iterator<Item = (&str, usize)> {
        self.events.iter().filter_map(|e| match e {
            GraphEvent::BranchCreated {
                branch_id,
                parent_event,
                ..
            } => Some((branch_id.as_str(), *parent_event)),
            _ => None,
        })
    }
}
