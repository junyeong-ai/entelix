//! Shared agent state types used across the recipes.
//!
//! Each recipe carries its messages list around as `Vec<Message>` (the
//! shape `ChatModel` consumes). The agent-state structs below add a
//! handful of bookkeeping fields per recipe.

use entelix_core::ir::{ContentPart, Message, Role};

/// Extract the concatenated [`ContentPart::Text`] body of the most
/// recent [`Role::Assistant`] message in `messages`. Returns `None`
/// when no assistant message exists or every assistant message
/// contains only non-text content (tool calls only, reasoning only).
///
/// Reasoning blocks (`ContentPart::Reasoning`) are excluded by design
/// — they're a separate variant from `Text` so recipes can surface
/// them independently; the "assistant text" accessor returns the
/// user-facing reply text only.
fn last_assistant_text(messages: &[Message]) -> Option<String> {
    let assistant = messages.iter().rev().find(|m| m.role == Role::Assistant)?;
    let mut buf = String::new();
    for part in &assistant.content {
        if let ContentPart::Text { text, .. } = part {
            buf.push_str(text);
        }
    }
    if buf.is_empty() { None } else { Some(buf) }
}

/// State for [`crate::create_chat_agent`] and the simplest single-turn
/// recipes — just the conversation so far.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChatState {
    /// The conversation, oldest first. The agent appends one assistant
    /// message per invocation.
    pub messages: Vec<Message>,
}

impl ChatState {
    /// Build a state with a single user message.
    pub fn from_user(text: impl Into<String>) -> Self {
        Self {
            messages: vec![Message::user(text)],
        }
    }

    /// Concatenated user-facing text of the most recent assistant
    /// message. `None` when no assistant message exists or only
    /// non-text content was emitted (tool calls only).
    #[must_use]
    pub fn last_assistant_text(&self) -> Option<String> {
        last_assistant_text(&self.messages)
    }
}

/// State for [`crate::create_react_agent`] — messages plus a step count
/// to make traces easier to inspect.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReActState {
    /// The conversation, oldest first. Tool results are appended as
    /// `Role::Tool` messages between assistant turns.
    pub messages: Vec<Message>,
    /// Number of (model + tool) round trips taken so far.
    pub steps: usize,
}

impl ReActState {
    /// Build a state with a single user message.
    pub fn from_user(text: impl Into<String>) -> Self {
        Self {
            messages: vec![Message::user(text)],
            steps: 0,
        }
    }

    /// Concatenated user-facing text of the most recent assistant
    /// message. `None` when no assistant message exists or only
    /// non-text content was emitted (tool calls only).
    #[must_use]
    pub fn last_assistant_text(&self) -> Option<String> {
        last_assistant_text(&self.messages)
    }
}

/// State for [`crate::create_supervisor_agent`] — messages plus the
/// last-active sub-agent identifier for traceability.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SupervisorState {
    /// The conversation, oldest first.
    pub messages: Vec<Message>,
    /// Name of the agent whose output most recently appended to
    /// `messages`. `None` until the supervisor first dispatches.
    pub last_speaker: Option<String>,
    /// Routing decision set by the supervisor's last router call.
    /// Reset to `None` when a sub-agent dispatch returns and the
    /// supervisor is about to make the next decision.
    pub next_speaker: Option<crate::supervisor::SupervisorDecision>,
}

impl SupervisorState {
    /// Build a state with a single user message.
    pub fn from_user(text: impl Into<String>) -> Self {
        Self {
            messages: vec![Message::user(text)],
            last_speaker: None,
            next_speaker: None,
        }
    }

    /// Concatenated user-facing text of the most recent assistant
    /// message. `None` when no assistant message exists or only
    /// non-text content was emitted (tool calls only).
    #[must_use]
    pub fn last_assistant_text(&self) -> Option<String> {
        last_assistant_text(&self.messages)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn assistant(parts: Vec<ContentPart>) -> Message {
        Message::new(Role::Assistant, parts)
    }

    fn text(s: &str) -> ContentPart {
        ContentPart::text(s)
    }

    #[test]
    fn returns_none_when_no_assistant_message_exists() {
        let state = ChatState::from_user("hi");
        assert_eq!(state.last_assistant_text(), None);
    }

    #[test]
    fn concatenates_every_text_part_of_the_last_assistant_message() {
        // Multi-part assistant message — every Text part contributes.
        let mut state = ChatState::from_user("hi");
        state
            .messages
            .push(assistant(vec![text("first"), text(" "), text("second")]));
        assert_eq!(state.last_assistant_text(), Some("first second".to_owned()));
    }

    #[test]
    fn skips_non_text_content_parts() {
        // Tool-use blocks interleaved with text — only Text parts
        // accumulate, preserving in-order concatenation.
        let mut state = ReActState::from_user("ask");
        let tool_use = ContentPart::ToolUse {
            id: "tu1".into(),
            name: "calc".into(),
            input: serde_json::json!({}),
            provider_echoes: Vec::new(),
        };
        state
            .messages
            .push(assistant(vec![text("before"), tool_use, text("after")]));
        assert_eq!(state.last_assistant_text(), Some("beforeafter".to_owned()));
    }

    #[test]
    fn returns_none_when_last_assistant_message_has_no_text() {
        // Assistant turn with only tool-use blocks — no user-facing
        // text to surface.
        let mut state = SupervisorState::from_user("ask");
        let tool_use = ContentPart::ToolUse {
            id: "tu1".into(),
            name: "calc".into(),
            input: serde_json::json!({}),
            provider_echoes: Vec::new(),
        };
        state.messages.push(assistant(vec![tool_use]));
        assert_eq!(state.last_assistant_text(), None);
    }

    #[test]
    fn returns_text_from_most_recent_assistant_skipping_earlier_turns() {
        // Multiple assistant turns; the helper returns the LAST one's
        // text, not the first.
        let mut state = ChatState::from_user("hi");
        state.messages.push(assistant(vec![text("old")]));
        state.messages.push(Message::user("follow-up"));
        state.messages.push(assistant(vec![text("new")]));
        assert_eq!(state.last_assistant_text(), Some("new".to_owned()));
    }
}
