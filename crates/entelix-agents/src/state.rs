//! Shared agent state types used across the recipes.
//!
//! Each recipe carries its messages list around as `Vec<Message>` (the
//! shape `ChatModel` consumes). The agent-state structs below add a
//! handful of bookkeeping fields per recipe.

use entelix_core::ir::Message;

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
}
