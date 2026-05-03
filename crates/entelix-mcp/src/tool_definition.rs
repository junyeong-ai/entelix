//! `McpToolDefinition` — what an MCP server advertises in
//! `tools/list`. Mirror of the JSON shape spec'd by Anthropic's MCP
//! 2024-11-05 protocol revision.

use serde::{Deserialize, Serialize};

/// One tool published by an MCP server.
///
/// Fields not yet used by entelix (e.g. annotations, mime types) are
/// preserved through `extras` so future protocol bumps don't strip
/// information from in-flight payloads.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpToolDefinition {
    /// Stable tool identifier — passed back as `tools/call.params.name`.
    pub name: String,
    /// Human-readable description shown to the model.
    #[serde(default)]
    pub description: String,
    /// JSON Schema describing the tool's argument shape.
    #[serde(rename = "inputSchema", default)]
    pub input_schema: serde_json::Value,
    /// Any additional fields the server returned that we don't yet
    /// model in IR. Preserved for diagnostics and forward
    /// compatibility.
    #[serde(flatten)]
    pub extras: serde_json::Map<String, serde_json::Value>,
}
