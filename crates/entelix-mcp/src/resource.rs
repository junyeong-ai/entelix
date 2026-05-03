//! `McpResource` + `McpResourceContent` — what an MCP server
//! advertises in `resources/list` and returns from `resources/read`.
//! Spec reference: 2024-11-05 revision §"Resources".

use serde::{Deserialize, Serialize};

/// One resource published by an MCP server.
///
/// Resources are addressable by `uri`; `mime_type` is operator-supplied
/// metadata that downstream agents may use to choose a renderer or
/// parsing strategy. Future protocol bumps that introduce new fields
/// ride through `extras`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpResource {
    /// Stable resource identifier.
    pub uri: String,
    /// Human-readable display name.
    #[serde(default)]
    pub name: String,
    /// Operator-supplied description shown to the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// IANA media type when known (e.g. `application/json`).
    #[serde(rename = "mimeType", default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Any additional fields the server returned that we don't yet
    /// model. Preserved for forward compatibility.
    #[serde(flatten)]
    pub extras: serde_json::Map<String, serde_json::Value>,
}

/// One block of content returned by `resources/read`. The MCP spec
/// allows a single `read` to return multiple blocks (e.g. a file
/// plus its metadata).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(untagged)]
pub enum McpResourceContent {
    /// Text payload (e.g. JSON, Markdown, source code).
    Text {
        /// Resource URI this block belongs to. Set on multi-block
        /// reads so the consumer can re-correlate blocks to their
        /// originating resource.
        uri: String,
        /// IANA media type when the server knows it.
        #[serde(rename = "mimeType", default, skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        /// The text payload.
        text: String,
    },
    /// Binary payload, base64-encoded on the wire.
    Blob {
        /// Resource URI this block belongs to.
        uri: String,
        /// IANA media type when the server knows it.
        #[serde(rename = "mimeType", default, skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        /// Base64-encoded payload (per MCP spec — not pre-decoded so
        /// the consumer chooses when to allocate the decoded bytes).
        blob: String,
    },
}
