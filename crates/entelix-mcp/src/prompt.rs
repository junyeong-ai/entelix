//! `McpPrompt` + related types for `prompts/list` and `prompts/get`.
//! Spec reference: 2024-11-05 revision §"Prompts".

use serde::{Deserialize, Serialize};

/// One prompt template published by an MCP server.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpPrompt {
    /// Stable prompt identifier — passed back as `prompts/get.params.name`.
    pub name: String,
    /// Human-readable description shown to the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Argument descriptors. Empty when the prompt takes no arguments.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub arguments: Vec<McpPromptArgument>,
    /// Forward-compatibility slot for fields not yet modelled.
    #[serde(flatten)]
    pub extras: serde_json::Map<String, serde_json::Value>,
}

/// One argument expected by a prompt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpPromptArgument {
    /// Argument name — used as the key in `prompts/get.params.arguments`.
    pub name: String,
    /// Human-readable description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Whether the server requires this argument.
    #[serde(default)]
    pub required: bool,
}

/// Result of a `prompts/get` invocation — a description plus an
/// ordered transcript of messages the agent should add to its
/// context.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpPromptInvocation {
    /// Operator-supplied summary of the bound prompt. Useful for
    /// observability / audit when the agent decides to inject the
    /// transcript into a session.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Ordered transcript. Roles are `"user"` / `"assistant"` per
    /// spec — kept as `String` rather than a closed enum so future
    /// MCP revisions adding new roles don't break decode.
    pub messages: Vec<McpPromptMessage>,
}

/// One message in a `prompts/get` transcript.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpPromptMessage {
    /// Role tag (`"user"` / `"assistant"` / future variants).
    pub role: String,
    /// Content block. MCP defines a single content per message at
    /// the 2024-11-05 revision; future revisions may broaden this
    /// to a list — that change rides through as a wrapper variant.
    pub content: McpPromptContent,
}

/// Content inside a [`McpPromptMessage`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum McpPromptContent {
    /// Plain text.
    Text {
        /// The text payload.
        text: String,
    },
    /// Inline image, base64-encoded.
    Image {
        /// Base64-encoded image bytes.
        data: String,
        /// IANA media type (e.g. `image/png`).
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    /// Reference to a resource exposed by the same server. Useful
    /// when a prompt wants to embed file content without forcing the
    /// agent to round-trip through `resources/read` itself.
    Resource {
        /// The embedded resource block.
        resource: McpPromptResourceRef,
    },
}

/// Resource block embedded inside a [`McpPromptContent::Resource`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpPromptResourceRef {
    /// Resource URI.
    pub uri: String,
    /// IANA media type when known.
    #[serde(rename = "mimeType", default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Inline text payload, when the server inlined the resource's
    /// content rather than asking the client to fetch it via
    /// `resources/read`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Inline base64-encoded blob payload — same caveat as `text`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blob: Option<String>,
}
