//! MCP-layer errors. Surface to callers as
//! `entelix_core::Error::Provider` (or `Config`) at the
//! `McpManager` boundary.

use thiserror::Error;

use entelix_core::error::Error;

/// Result alias used inside `entelix-mcp`.
pub type McpResult<T> = std::result::Result<T, McpError>;

/// MCP-layer failures.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum McpError {
    /// Network / HTTP failure reaching the MCP server. Carries the
    /// underlying transport error (`reqwest::Error`, `std::io::Error`,
    /// …) as the source so callers see the full chain via
    /// `std::error::Error::source`.
    #[error("network failure: {message}")]
    Network {
        /// Human-readable summary of the failure.
        message: String,
        /// Underlying transport error.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
    },

    /// JSON-RPC level error returned by the server.
    #[error("MCP server returned error code {code}: {message}")]
    JsonRpc {
        /// JSON-RPC error code.
        code: i64,
        /// Server-supplied message.
        message: String,
    },

    /// Server returned a malformed JSON-RPC envelope (missing `result`
    /// and `error`, type mismatch, etc.). Carries the underlying
    /// parse error as the source when one is available.
    #[error("malformed JSON-RPC response: {message}")]
    MalformedResponse {
        /// Human-readable summary of the malformed shape.
        message: String,
        /// Underlying decode error, if any.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
    },

    /// The MCP `(tenant_id, server)` pair has not been configured.
    #[error("MCP server '{server}' not registered for tenant '{tenant_id}'")]
    UnknownServer {
        /// Tenant scope.
        tenant_id: String,
        /// Server name.
        server: String,
    },

    /// Configuration error — malformed URL, missing field, etc.
    #[error("configuration error: {0}")]
    Config(String),

    /// Reconnect attempts exhausted.
    #[error("MCP server '{server}' unreachable after {attempts} attempts")]
    Unreachable {
        /// Server name.
        server: String,
        /// Number of attempts taken.
        attempts: u32,
    },

    /// Listener-side resource bound exceeded — e.g. an SSE frame
    /// grew past `McpServerConfig::max_frame_bytes` without a
    /// terminator (signalling a hostile or malfunctioning peer).
    /// Semantically distinct from [`McpError::MalformedResponse`],
    /// which is "vendor sent garbage on the wire" — this variant is
    /// "vendor exceeded the operator-tunable bound the SDK ships
    /// against DoS-class behaviour".
    #[error("MCP listener resource bound exceeded ({kind}): {message}")]
    ResourceBounded {
        /// Which bound was crossed.
        kind: ResourceBoundKind,
        /// Operator-actionable diagnostic.
        message: String,
    },

    /// JSON encode/decode failure.
    #[error(transparent)]
    Serde(#[from] serde_json::Error),
}

/// Names the resource bound that fired in [`McpError::ResourceBounded`].
/// `#[non_exhaustive]` so future bounds (per-method quota, total
/// session bytes, …) can ship without breaking match arms.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ResourceBoundKind {
    /// Single SSE frame exceeded `max_frame_bytes` without a `\n\n`
    /// terminator. Listener closed the connection.
    FrameSize,
}

impl std::fmt::Display for ResourceBoundKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FrameSize => f.write_str("frame_size"),
        }
    }
}

impl McpError {
    /// Wrap any transport-layer error as a [`McpError::Network`],
    /// preserving the source chain.
    pub fn network<E>(source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Network {
            message: source.to_string(),
            source: Some(Box::new(source)),
        }
    }

    /// Construct a [`McpError::Network`] from a bare message (no
    /// underlying error to chain — typically a non-2xx HTTP status).
    pub fn network_status(message: impl Into<String>) -> Self {
        Self::Network {
            message: message.into(),
            source: None,
        }
    }

    /// Wrap a decode error as a [`McpError::MalformedResponse`],
    /// preserving the source chain.
    pub fn malformed<E>(source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::MalformedResponse {
            message: source.to_string(),
            source: Some(Box::new(source)),
        }
    }

    /// Construct a [`McpError::MalformedResponse`] from a bare message
    /// (semantic violation with no inner error to chain).
    pub fn malformed_msg(message: impl Into<String>) -> Self {
        Self::MalformedResponse {
            message: message.into(),
            source: None,
        }
    }
}

impl From<McpError> for Error {
    fn from(err: McpError) -> Self {
        match err {
            McpError::Config(msg) => Self::config(msg),
            McpError::UnknownServer { .. } => Self::invalid_request(err.to_string()),
            McpError::ResourceBounded { .. } => Self::provider_network(err.to_string()),
            other => Self::provider_network_from(other),
        }
    }
}

impl entelix_core::LlmRenderable<String> for McpError {
    /// Short, model-actionable rendering. Operator diagnostics
    /// (transport source chains, tenant identifiers, raw vendor
    /// messages, malformed-response bodies) never enter this
    /// channel — they continue to flow through `Display` / source
    /// chains / tracing.
    fn render_for_llm(&self) -> String {
        match self {
            Self::Network { .. } => "MCP transport failure".to_owned(),
            Self::JsonRpc { code, .. } => format!("MCP server returned error code {code}"),
            Self::MalformedResponse { .. } => "MCP malformed response".to_owned(),
            Self::UnknownServer { server, .. } => {
                format!("MCP server '{server}' not registered")
            }
            Self::Config(_) => "MCP misconfigured".to_owned(),
            Self::Unreachable { server, .. } => {
                format!("MCP server '{server}' unreachable")
            }
            Self::ResourceBounded { kind, .. } => {
                format!("MCP listener bound exceeded ({kind})")
            }
            Self::Serde(_) => "MCP payload could not be serialised".to_owned(),
        }
    }
}
