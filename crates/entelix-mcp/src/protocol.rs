//! JSON-RPC 2.0 envelope + MCP-specific request/response payloads.
//!
//! Spec reference: <https://modelcontextprotocol.io/specification/>
//! (2024-11-05 revision). The protocol version is hard-coded; future
//! bumps land alongside a new revision constant.

// `pub(crate)` items are crate-internal; the workspace uses `unreachable_pub`
// (rust) so we can't drop the qualifier. The clippy nursery lint
// `redundant_pub_crate` disagrees — we side with the rust lint.
#![allow(clippy::redundant_pub_crate)]

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// MCP protocol revision this client speaks.
pub const PROTOCOL_VERSION: &str = "2024-11-05";

/// JSON-RPC envelope sent to the server.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct JsonRpcRequest<'a, P>
where
    P: Serialize,
{
    pub jsonrpc: &'static str,
    pub id: u64,
    pub method: &'a str,
    pub params: P,
}

impl<'a, P: Serialize> JsonRpcRequest<'a, P> {
    pub(crate) const fn new(id: u64, method: &'a str, params: P) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            method,
            params,
        }
    }
}

/// JSON-RPC envelope received from the server. `result` and `error`
/// are mutually exclusive per spec — we model that with `Option`s
/// and verify at decode time. `jsonrpc` and `id` carry no decision
/// information for the client and are silently ignored by serde.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct JsonRpcResponse {
    #[serde(default)]
    pub result: Option<Value>,
    #[serde(default)]
    pub error: Option<JsonRpcError>,
}

/// Error returned in a `JsonRpcResponse::error` slot.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct JsonRpcError {
    pub code: i64,
    pub message: String,
}

/// Notification (no `id`, no response).
#[derive(Clone, Debug, Serialize)]
pub(crate) struct JsonRpcNotification<'a, P>
where
    P: Serialize,
{
    pub jsonrpc: &'static str,
    pub method: &'a str,
    pub params: P,
}

impl<'a, P: Serialize> JsonRpcNotification<'a, P> {
    pub(crate) const fn new(method: &'a str, params: P) -> Self {
        Self {
            jsonrpc: "2.0",
            method,
            params,
        }
    }
}

// ── MCP-specific payloads ──────────────────────────────────────────

/// Body of an `initialize` request.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct InitializeParams<'a> {
    #[serde(rename = "protocolVersion")]
    pub protocol_version: &'static str,
    pub capabilities: ClientCapabilities,
    #[serde(rename = "clientInfo")]
    pub client_info: ClientInfo<'a>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct ClientCapabilities {
    /// Roots advertisement — present iff the operator wired a
    /// `RootsProvider`. Servers gate `roots/list` requests on this
    /// field, so omitting it suppresses server-initiated roots
    /// traffic entirely.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roots: Option<RootsCapability>,
    /// Elicitation advertisement — present iff the operator
    /// wired an `ElicitationProvider`. Servers gate
    /// `elicitation/create` requests on this field; omitting it
    /// suppresses server-initiated elicitation traffic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elicitation: Option<ElicitationCapability>,
    /// Sampling advertisement — present iff the operator wired
    /// a `SamplingProvider`. Servers gate
    /// `sampling/createMessage` requests on this field;
    /// omitting it suppresses server-initiated sampling
    /// traffic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling: Option<SamplingCapability>,
}

/// Body of the `roots` slot under [`ClientCapabilities`]. We always
/// advertise `listChanged: true` when roots are advertised at all
/// — the client implements `notifications/roots/list_changed`
/// unconditionally.
#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct RootsCapability {
    #[serde(rename = "listChanged")]
    pub list_changed: bool,
}

/// Body of the `elicitation` slot under [`ClientCapabilities`].
/// The spec defines no sub-fields — presence alone signals
/// support. Serialises to `{}` so the wire envelope stays
/// well-formed.
#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct ElicitationCapability {}

/// Body of the `sampling` slot under [`ClientCapabilities`].
/// Same shape as `ElicitationCapability` — empty struct
/// signalling support, serialises to `{}`.
#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct SamplingCapability {}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ClientInfo<'a> {
    pub name: &'a str,
    pub version: &'a str,
}

/// Body of a `tools/call` request.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct ToolsCallParams<'a> {
    pub name: &'a str,
    pub arguments: Value,
}

/// Result of a `tools/call`.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct ToolsCallResult {
    #[serde(default)]
    pub content: Vec<ToolContent>,
    #[serde(rename = "isError", default)]
    pub is_error: bool,
}

/// One content block in a `tools/call` result.
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ToolContent {
    Text {
        text: String,
    },
    #[serde(other)]
    Other,
}

/// Result of a `tools/list`.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct ToolsListResult {
    #[serde(default)]
    pub tools: Vec<crate::tool_definition::McpToolDefinition>,
}

/// Result of a `resources/list`.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct ResourcesListResult {
    #[serde(default)]
    pub resources: Vec<crate::resource::McpResource>,
}

/// Body of a `resources/read` request.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct ResourcesReadParams<'a> {
    pub uri: &'a str,
}

/// Result of a `resources/read`.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct ResourcesReadResult {
    #[serde(default)]
    pub contents: Vec<crate::resource::McpResourceContent>,
}

/// Result of a `prompts/list`.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct PromptsListResult {
    #[serde(default)]
    pub prompts: Vec<crate::prompt::McpPrompt>,
}

/// Body of a `prompts/get` request.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct PromptsGetParams<'a> {
    pub name: &'a str,
    /// Server expects a flat string→string map. `BTreeMap` keeps
    /// the wire ordering deterministic for replay parity.
    pub arguments: std::collections::BTreeMap<String, String>,
}

/// Body of a `completion/complete` request. The MCP spec writes
/// `ref` (a Rust keyword) so the wire field is renamed.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct CompleteParams {
    #[serde(rename = "ref")]
    pub reference: crate::completion::McpCompletionReference,
    pub argument: crate::completion::McpCompletionArgument,
}

/// Result envelope for `completion/complete`. The wire shape nests
/// the values under a `completion` key; we unwrap inside the client
/// so callers get [`crate::completion::McpCompletionResult`] directly.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct CompleteResult {
    pub completion: crate::completion::McpCompletionResult,
}

/// Server-initiated JSON-RPC request, received over the
/// streamable-http SSE channel. The id is server-side and may be a
/// number or a string per JSON-RPC; we deserialize as `Value` and
/// echo it back verbatim on the response. The dispatcher matches
/// on `method` and routes to the configured handler
/// ([`crate::RootsProvider`] for `roots/list`).
///
/// `params` is parsed but unused for `roots/list` (the spec
/// request payload is empty); future server-initiated methods
/// (sampling, elicitation) will consume it. We keep the field
/// captured so the dispatcher's `JsonRpcServerRequest`-shaped
/// match arms see the full request. `params` is consumed by
/// the `elicitation/create` dispatcher (and future
/// server-initiated methods that carry typed parameters);
/// `roots/list` ignores it per spec.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct JsonRpcServerRequest {
    pub id: Value,
    pub method: String,
    #[serde(default)]
    pub params: Option<Value>,
}

/// Result of `roots/list` (client-side response shape). Mirrors the
/// public [`crate::McpRoot`] but lives in `protocol` so the wire
/// shape is governed alongside other JSON-RPC types.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct RootsListResult {
    pub roots: Vec<crate::roots::McpRoot>,
}
