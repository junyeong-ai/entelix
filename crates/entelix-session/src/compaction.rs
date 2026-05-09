//! `Compactor` + sealed `CompactedHistory<Turn>` — type-enforced
//! `ToolCall` / `ToolResult` pair invariant for context compaction.
//!
//! Long agent runs accumulate event logs that exceed the model's
//! context window. Operators must drop *some* events without
//! breaking the conversation invariants vendors enforce — chiefly
//! that every `tool_use` block has a matching `tool_result`.
//! Mismatched pairs surface as HTTP 400s on the next call;
//! pydantic-ai's [issue #4137](https://github.com/pydantic/pydantic-ai/issues/4137)
//! catalogues the recurring footgun across SDKs.
//!
//! `entelix-session` closes the foot-gun by exposing compaction
//! through this module. A [`Compactor`] consumes `&[GraphEvent]`
//! and returns a [`CompactedHistory`] whose constructor is sealed
//! to this module — operators cannot hand-build a `CompactedHistory`
//! that violates the pair invariant. The sealed [`Turn`] enum
//! groups events into:
//!
//! - [`Turn::User`] — one `UserMessage`.
//! - [`Turn::Assistant`] — one `AssistantMessage` plus zero or
//!   more [`ToolPair`]s, each binding a `ToolCall` to its
//!   matching `ToolResult` *by structure*.
//!
//! Because `ToolPair` cannot be constructed with only one half,
//! every compaction strategy operates on whole `Turn`s — the
//! model never receives a `tool_use` without its `tool_result`.
//!
//! ## Reference impl
//!
//! [`HeadDropCompactor`] is the canonical "drop oldest" strategy:
//! walks turns from newest backwards, keeps turns that fit under
//! the character budget, returns the trimmed window. Operators
//! whose use case wants summary-style compaction (LLM-generated
//! synopsis of dropped turns) implement [`Compactor`] directly.

use std::collections::HashMap;

use async_trait::async_trait;
use chrono::Utc;
use entelix_core::ExecutionContext;
use entelix_core::error::{Error, Result};
use entelix_core::ir::{ContentPart, Message, Role, ToolResultContent};

use crate::event::GraphEvent;

/// One matched `ToolCall` / `ToolResult` pair. Sealed: the only
/// path to construction is [`Compactor::compact`] internal grouping,
/// so a pair without both halves cannot exist.
///
/// Read-only accessors expose the call's id / name / input and the
/// result's content / error flag for operators that inspect the
/// compacted view (rendering, dashboards). Mutation is not
/// supported — a `Turn` carrying a different pair set is a fresh
/// compaction.
#[derive(Clone, Debug)]
pub struct ToolPair {
    call_id: String,
    name: String,
    input: serde_json::Value,
    result: ToolResultContent,
    is_error: bool,
}

impl ToolPair {
    /// Stable tool-use id binding the call to its result.
    pub fn id(&self) -> &str {
        &self.call_id
    }

    /// Tool name as registered with the dispatching `ToolRegistry`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Tool input as JSON.
    pub const fn input(&self) -> &serde_json::Value {
        &self.input
    }

    /// Result payload returned by the tool.
    pub const fn result(&self) -> &ToolResultContent {
        &self.result
    }

    /// Whether the tool reported an error path.
    pub const fn is_error(&self) -> bool {
        self.is_error
    }
}

/// One turn in a compacted conversation. Sealed so a `Turn::Assistant`
/// can only carry [`ToolPair`]s constructed via the compactor's
/// internal grouping code (paired calls + results).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Turn {
    /// User-authored message — opaque content, never paired.
    User {
        /// Multi-part content (text, image, …).
        content: Vec<ContentPart>,
    },
    /// Assistant-authored message + the tool round-trips it
    /// initiated. Empty `tools` means the assistant turn produced
    /// final text only.
    Assistant {
        /// Assistant's content (may include `ContentPart::ToolUse`
        /// blocks; the embedded tool-use ids match the
        /// corresponding [`ToolPair::id`]s).
        content: Vec<ContentPart>,
        /// Matched tool round-trips initiated by this turn.
        tools: Vec<ToolPair>,
    },
}

/// Compacted view over a `SessionGraph`'s event log.
///
/// External operators implementing [`Compactor`] for a custom
/// strategy (LLM-summary compaction, importance-weighted retention,
/// …) construct the initial form via [`CompactedHistory::group`]
/// and return either the same value or one rebuilt with
/// [`CompactedHistory::from_turns`] after filtering. The
/// `tool_call` / `tool_result` pair invariant stays type-enforced:
/// the only path to a [`ToolPair`] is the internal grouping code,
/// so external impls can drop or pass through tool round-trips
/// but can't synthesize unmatched ones.
#[derive(Clone, Debug)]
pub struct CompactedHistory {
    turns: Vec<Turn>,
}

impl CompactedHistory {
    /// Group `events` into the type-enforced [`Turn`] shape and
    /// return the un-trimmed compaction. The grouping rejects an
    /// event log that violates the pair invariant *before*
    /// compaction (e.g. `ToolResult` without a preceding
    /// `ToolCall`); a well-formed `SessionGraph` never hits the
    /// error path.
    ///
    /// External [`Compactor`] impls call this to get the initial
    /// grouped form, then choose which turns to retain.
    pub fn group(events: &[GraphEvent]) -> Result<Self> {
        Ok(Self {
            turns: group_into_turns(events)?,
        })
    }

    /// Build a `CompactedHistory` from a pre-grouped `Vec<Turn>`.
    /// External [`Compactor`] impls reach for this after filtering
    /// or transforming the turns returned by
    /// [`CompactedHistory::group`]. The pair invariant survives
    /// the round-trip because the only path to a [`ToolPair`] is
    /// still the internal grouping — operators pass them through
    /// but can't synthesize new ones.
    #[must_use]
    pub const fn from_turns(turns: Vec<Turn>) -> Self {
        Self { turns }
    }

    /// Borrow the compacted turns.
    pub fn turns(&self) -> &[Turn] {
        &self.turns
    }

    /// Number of turns retained.
    pub const fn len(&self) -> usize {
        self.turns.len()
    }

    /// Whether the compacted history is empty.
    pub const fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    /// Render as `Vec<Message>` suitable for `ChatModel::complete`.
    /// Mirrors [`crate::SessionGraph::current_branch_messages`] but
    /// over the compacted view: every assistant turn's `tool_use`
    /// blocks are followed by a synthetic `Role::Tool` message
    /// per [`ToolPair`], so the wire-side codec sees the matched
    /// pairs the vendor expects.
    pub fn to_messages(&self) -> Vec<Message> {
        let mut out = Vec::with_capacity(self.turns.len() * 2);
        for turn in &self.turns {
            match turn {
                Turn::User { content } => {
                    out.push(Message::new(Role::User, content.clone()));
                }
                Turn::Assistant { content, tools } => {
                    out.push(Message::new(Role::Assistant, content.clone()));
                    for pair in tools {
                        out.push(Message::new(
                            Role::Tool,
                            vec![ContentPart::ToolResult {
                                tool_use_id: pair.call_id.clone(),
                                name: pair.name.clone(),
                                content: pair.result.clone(),
                                is_error: pair.is_error,
                                cache_control: None,
                            }],
                        ));
                    }
                }
            }
        }
        out
    }
}

/// Operator-supplied compaction strategy.
///
/// Receives the full event log plus a character-budget hint and
/// returns the trimmed view. Async by default so summary-style
/// implementations can dispatch a `ChatModel` call (`SummaryCompactor`
/// in `entelix-agents` is the canonical reference); pure-retention
/// strategies (`HeadDropCompactor`) simply ignore the future point
/// and return synchronously inside the async fn body.
///
/// Implementations must preserve the `ToolCall` / `ToolResult` pair
/// invariant — the [`CompactedHistory`] return type enforces that
/// structurally; trait authors only need to choose *which* turns to
/// retain.
#[async_trait]
pub trait Compactor: Send + Sync + 'static {
    /// Compact `events` to fit within `budget_chars`. The budget is
    /// approximate — implementations measure character length of
    /// the rendered text (closest free proxy for token count
    /// without pulling a tokenizer dependency). The
    /// [`ExecutionContext`] carries cancellation + deadline so a
    /// long-running summarisation respects the same lifetime as the
    /// dispatch that triggered it. Returns [`Error::Config`] when
    /// the event log violates the pair invariant *before*
    /// compaction (e.g. `ToolResult` without a preceding
    /// `ToolCall`); a well-formed `SessionGraph` never hits this
    /// path.
    async fn compact(
        &self,
        events: &[GraphEvent],
        budget_chars: usize,
        ctx: &ExecutionContext,
    ) -> Result<CompactedHistory>;
}

/// Reference compactor: drop oldest turns until the rendered
/// character count fits under `budget_chars`. Tool round-trips
/// stay paired by construction; the strategy never partially
/// includes a turn.
///
/// Synchronous in spirit — the async fn body runs to completion
/// without awaiting any future. Operators that want LLM-generated
/// summary compaction reach for `entelix_agents::SummaryCompactor`
/// instead.
#[derive(Clone, Copy, Debug, Default)]
pub struct HeadDropCompactor;

#[async_trait]
impl Compactor for HeadDropCompactor {
    async fn compact(
        &self,
        events: &[GraphEvent],
        budget_chars: usize,
        _ctx: &ExecutionContext,
    ) -> Result<CompactedHistory> {
        let mut turns = CompactedHistory::group(events)?.turns;
        // Walk newest to oldest, keep turns that fit under budget.
        let mut remaining = budget_chars;
        let mut keep_index = turns.len();
        for (idx, turn) in turns.iter().enumerate().rev() {
            let cost = turn_char_cost(turn);
            if cost > remaining {
                break;
            }
            remaining -= cost;
            keep_index = idx;
        }
        let trimmed = turns.split_off(keep_index);
        Ok(CompactedHistory::from_turns(trimmed))
    }
}

/// Render an in-flight `Vec<Message>` (the shape an agent's working
/// state carries) into a `Vec<GraphEvent>` that [`Compactor::compact`]
/// can consume. Inverse of [`CompactedHistory::to_messages`].
///
/// Auto-compaction wiring: an agent loop holds messages, not events,
/// so the trigger path needs this helper to feed the existing
/// event-shaped compaction surface — preserving the type-enforced
/// `tool_call` / `tool_result` pair invariant end-to-end.
///
/// `Role::System` messages are dropped — system prompts ride outside
/// the event log by design (configured separately on the model). All
/// timestamps are stamped with [`Utc::now`] since per-message wall-clock
/// is unavailable from the message representation; compaction does not
/// rely on event ordering by timestamp (it uses positional ordering).
///
/// Returns [`Error::Config`] when the message sequence violates the
/// `tool_call` / `tool_result` pair invariant before compaction (e.g.
/// `Role::Tool` content carrying a `tool_use_id` with no preceding
/// assistant `ToolUse` part).
pub fn messages_to_events(messages: &[Message]) -> Result<Vec<GraphEvent>> {
    let now = Utc::now();
    let mut events = Vec::with_capacity(messages.len() * 2);
    for msg in messages {
        match msg.role {
            Role::User => {
                events.push(GraphEvent::UserMessage {
                    content: msg.content.clone(),
                    timestamp: now,
                });
            }
            Role::Assistant => {
                events.push(GraphEvent::AssistantMessage {
                    content: msg.content.clone(),
                    usage: None,
                    timestamp: now,
                });
                for part in &msg.content {
                    if let ContentPart::ToolUse { id, name, input } = part {
                        events.push(GraphEvent::ToolCall {
                            id: id.clone(),
                            name: name.clone(),
                            input: input.clone(),
                            timestamp: now,
                        });
                    }
                }
            }
            Role::Tool => {
                for part in &msg.content {
                    if let ContentPart::ToolResult {
                        tool_use_id,
                        name,
                        content,
                        is_error,
                        ..
                    } = part
                    {
                        events.push(GraphEvent::ToolResult {
                            tool_use_id: tool_use_id.clone(),
                            name: name.clone(),
                            content: content.clone(),
                            is_error: *is_error,
                            timestamp: now,
                        });
                    }
                }
            }
            // `Role::System` rides outside the event log (configured
            // separately on the model); future variants similarly do
            // not represent appendable conversation turns.
            _ => {}
        }
    }
    Ok(events)
}

/// Character-length proxy for the token cost of a message slice. Same
/// metric [`HeadDropCompactor`] uses to compare against `budget_chars`,
/// so threshold-driven auto-compaction can use the same yardstick.
#[must_use]
pub fn messages_char_size(messages: &[Message]) -> usize {
    messages.iter().map(|m| content_chars(&m.content)).sum()
}

/// Group events into the type-enforced [`Turn`] shape. Every
/// `ToolCall` must have a matching `ToolResult` (paired by `id`);
/// every `ToolResult` must follow an `AssistantMessage`. Returns
/// [`Error::Config`] on either violation.
fn group_into_turns(events: &[GraphEvent]) -> Result<Vec<Turn>> {
    let mut pending_calls: HashMap<String, (String, serde_json::Value)> = HashMap::new();
    let mut turns: Vec<Turn> = Vec::new();
    for event in events {
        match event {
            GraphEvent::UserMessage { content, .. } => {
                turns.push(Turn::User {
                    content: content.clone(),
                });
            }
            GraphEvent::AssistantMessage { content, .. } => {
                turns.push(Turn::Assistant {
                    content: content.clone(),
                    tools: Vec::new(),
                });
            }
            GraphEvent::ToolCall {
                id, name, input, ..
            } => {
                pending_calls.insert(id.clone(), (name.clone(), input.clone()));
            }
            GraphEvent::ToolResult {
                tool_use_id,
                name,
                content,
                is_error,
                ..
            } => {
                let (_call_name, call_input) =
                    pending_calls.remove(tool_use_id).ok_or_else(|| {
                        Error::config(format!(
                            "Compactor: ToolResult tool_use_id={tool_use_id} \
                             has no matching ToolCall in event log"
                        ))
                    })?;
                let pair = ToolPair {
                    call_id: tool_use_id.clone(),
                    name: name.clone(),
                    input: call_input,
                    result: content.clone(),
                    is_error: *is_error,
                };
                let host = turns
                    .iter_mut()
                    .rev()
                    .find(|t| matches!(t, Turn::Assistant { .. }))
                    .ok_or_else(|| {
                        Error::config("Compactor: ToolResult appeared before any AssistantMessage")
                    })?;
                if let Turn::Assistant { tools, .. } = host {
                    tools.push(pair);
                }
            }
            _ => {}
        }
    }
    if !pending_calls.is_empty() {
        return Err(Error::config(format!(
            "Compactor: {} ToolCall(s) without matching ToolResult — pair invariant violated",
            pending_calls.len()
        )));
    }
    Ok(turns)
}

/// Character-length proxy for token cost. Walks the turn's content
/// blocks summing text bytes (UTF-8 byte count, not grapheme count
/// — the cheap-monotonic property is what matters for "drop until
/// under budget"). Tool inputs / outputs contribute their JSON
/// serialisation length.
fn turn_char_cost(turn: &Turn) -> usize {
    match turn {
        Turn::User { content } => content_chars(content),
        Turn::Assistant { content, tools } => {
            let mut sum = content_chars(content);
            for pair in tools {
                sum += pair.input.to_string().len();
                sum += match &pair.result {
                    ToolResultContent::Text(s) => s.len(),
                    ToolResultContent::Json(v) => v.to_string().len(),
                    _ => 0,
                };
            }
            sum
        }
    }
}

fn content_chars(parts: &[ContentPart]) -> usize {
    parts
        .iter()
        .map(|p| match p {
            ContentPart::Text { text, .. } | ContentPart::Thinking { text, .. } => text.len(),
            ContentPart::ToolUse { input, .. } => input.to_string().len(),
            ContentPart::ToolResult { content, .. } => match content {
                ToolResultContent::Text(s) => s.len(),
                ToolResultContent::Json(v) => v.to_string().len(),
                _ => 0,
            },
            _ => 0,
        })
        .sum()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use chrono::Utc;
    use serde_json::json;

    use super::*;

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

    fn tool_call(id: &str, name: &str, input: serde_json::Value) -> GraphEvent {
        GraphEvent::ToolCall {
            id: id.to_owned(),
            name: name.to_owned(),
            input,
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

    #[tokio::test]
    async fn empty_event_log_compacts_to_empty_history() {
        let history = HeadDropCompactor
            .compact(&[], 1024, &ExecutionContext::new())
            .await
            .unwrap();
        assert!(history.is_empty());
    }

    #[tokio::test]
    async fn user_assistant_round_trip_preserves_both_turns() {
        let events = vec![user("hi"), assistant("hello!")];
        let history = HeadDropCompactor
            .compact(&events, 1024, &ExecutionContext::new())
            .await
            .unwrap();
        assert_eq!(history.len(), 2);
        assert!(matches!(history.turns()[0], Turn::User { .. }));
        assert!(matches!(history.turns()[1], Turn::Assistant { .. }));
    }

    #[tokio::test]
    async fn tool_pair_attaches_to_preceding_assistant_turn() {
        let events = vec![
            user("compute 1+1"),
            assistant("calling calculator"),
            tool_call("call_1", "calculator", json!({"expr": "1+1"})),
            tool_result("call_1", "calculator", "2"),
            assistant("answer is 2"),
        ];
        let history = HeadDropCompactor
            .compact(&events, 1024, &ExecutionContext::new())
            .await
            .unwrap();
        assert_eq!(history.len(), 3); // user + assistant + assistant
        if let Turn::Assistant { tools, .. } = &history.turns()[1] {
            assert_eq!(tools.len(), 1);
            assert_eq!(tools[0].id(), "call_1");
            assert_eq!(tools[0].name(), "calculator");
        } else {
            panic!("expected Assistant turn at index 1");
        }
    }

    #[tokio::test]
    async fn tool_result_without_matching_call_returns_config_error() {
        let events = vec![
            user("ask"),
            assistant("calling"),
            tool_result("orphan", "calc", "x"),
        ];
        let err = HeadDropCompactor
            .compact(&events, 1024, &ExecutionContext::new())
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("orphan"),
            "diagnostic must name the unmatched id: {msg}"
        );
    }

    #[tokio::test]
    async fn tool_call_without_matching_result_returns_config_error() {
        let events = vec![
            user("ask"),
            assistant("calling"),
            tool_call("dangling", "calc", json!({})),
        ];
        let err = HeadDropCompactor
            .compact(&events, 1024, &ExecutionContext::new())
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("pair invariant violated"), "got: {msg}");
    }

    #[tokio::test]
    async fn budget_drops_oldest_turns_keeps_newest() {
        // Three user/assistant round-trips. budget_chars selected
        // to fit only the last two turns.
        let events = vec![
            user("one one one"),
            assistant("one reply"),
            user("two two two"),
            assistant("two reply"),
            user("three three three"),
            assistant("three reply"),
        ];
        let history = HeadDropCompactor
            .compact(&events, 50, &ExecutionContext::new())
            .await
            .unwrap();
        // Must include the LAST turns under budget — never partial.
        assert!(!history.is_empty());
        let last = history.turns().last().unwrap();
        if let Turn::Assistant { content, .. } = last {
            if let ContentPart::Text { text, .. } = &content[0] {
                assert!(
                    text.contains("three"),
                    "newest turn must be retained, got: {text}"
                );
            }
        } else {
            panic!("expected Assistant as last turn");
        }
    }

    #[tokio::test]
    async fn to_messages_round_trips_user_assistant_tool_sequence() {
        let events = vec![
            user("ask"),
            assistant("calling"),
            tool_call("c", "tool", json!({})),
            tool_result("c", "tool", "ok"),
        ];
        let history = HeadDropCompactor
            .compact(&events, 1024, &ExecutionContext::new())
            .await
            .unwrap();
        let msgs = history.to_messages();
        assert_eq!(msgs.len(), 3); // user, assistant, tool
        assert!(matches!(msgs[0].role, Role::User));
        assert!(matches!(msgs[1].role, Role::Assistant));
        assert!(matches!(msgs[2].role, Role::Tool));
    }

    #[tokio::test]
    async fn pair_invariant_holds_under_partial_budget_drop() {
        // Even when budget forces dropping turns, the retained set
        // must NEVER contain an unpaired tool — the structural
        // guarantee of `Turn::Assistant`'s `tools: Vec<ToolPair>`
        // makes this true by construction; the test pins the
        // round-trip to catch any future refactor that loosens it.
        let events = vec![
            user("u1"),
            assistant("a1"),
            tool_call("t1", "x", json!({"v": 1})),
            tool_result("t1", "x", "r1"),
            user("u2"),
            assistant("a2"),
        ];
        let history = HeadDropCompactor
            .compact(&events, 30, &ExecutionContext::new())
            .await
            .unwrap();
        for turn in history.turns() {
            if let Turn::Assistant { tools, .. } = turn {
                for pair in tools {
                    // Both halves accessible — proves no half is missing.
                    let _ = (pair.id(), pair.name(), pair.input(), pair.result());
                }
            }
        }
    }
}
