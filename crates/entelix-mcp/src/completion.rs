//! `McpCompletion*` — types for `completion/complete`.
//! Spec reference: 2024-11-05 revision §"Completion".
//!
//! The completion endpoint lets the agent ask the server "what
//! values is `argument` likely to take" given a partial input. The
//! `reference` selects which argument-bearing surface (a prompt or a
//! resource template) is being completed.

use serde::{Deserialize, Serialize};

/// Reference target for a completion query.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum McpCompletionReference {
    /// A resource (template) — the server completes for one of its
    /// `{placeholder}` fields.
    #[serde(rename = "ref/resource")]
    Resource {
        /// Resource URI (may include `{placeholder}` segments).
        uri: String,
    },
    /// A prompt — the server completes for one of its declared
    /// arguments.
    #[serde(rename = "ref/prompt")]
    Prompt {
        /// Prompt name as published by `prompts/list`.
        name: String,
    },
}

/// Argument being completed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpCompletionArgument {
    /// Argument name as declared on the reference target.
    pub name: String,
    /// Partial value typed so far.
    pub value: String,
}

/// Response to a `completion/complete`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpCompletionResult {
    /// Suggested completion values, ordered server-side by relevance.
    /// Capped at 100 entries per spec; the server is free to return
    /// fewer.
    #[serde(default)]
    pub values: Vec<String>,
    /// Total candidate count when known. `None` when the server
    /// either did not compute a total or chose not to disclose it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total: Option<u32>,
    /// `true` when more candidates exist beyond `values` — the
    /// server expects the client to refine the partial value before
    /// asking again.
    #[serde(rename = "hasMore", default)]
    pub has_more: bool,
}
