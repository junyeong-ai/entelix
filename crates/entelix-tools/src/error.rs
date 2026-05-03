//! Tool-layer errors. Surface to callers as
//! [`entelix_core::Error::InvalidRequest`] (input shape mismatch),
//! [`entelix_core::Error::Provider`] (HTTP / upstream failure), or
//! [`entelix_core::Error::Config`] (misconfiguration).

use thiserror::Error;

use entelix_core::error::Error;

/// Result alias used inside `entelix-tools`.
pub type ToolResult<T> = std::result::Result<T, ToolError>;

/// Tool-layer failures.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ToolError {
    /// Input did not match the tool's `input_schema`.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Host blocked by the configured allowlist.
    #[error("host '{host}' not on the allowlist")]
    HostBlocked {
        /// The rejected host.
        host: String,
    },

    /// URL scheme was not `http` or `https`.
    #[error("unsupported URL scheme '{scheme}': only http/https are allowed")]
    UnsupportedScheme {
        /// The rejected scheme.
        scheme: String,
    },

    /// HTTP method blocked by the configured method allowlist.
    #[error("method '{method}' not allowed by this tool")]
    MethodBlocked {
        /// The rejected method, uppercased.
        method: String,
    },

    /// Response body exceeded the configured cap.
    #[error("response body exceeded {limit_bytes} bytes")]
    BodyTooLarge {
        /// Configured cap.
        limit_bytes: usize,
    },

    /// Network / HTTP failure. Carries the underlying transport error
    /// as the source so callers see the full chain via
    /// `std::error::Error::source`.
    #[error("network: {message}")]
    Network {
        /// Human-readable summary.
        message: String,
        /// Underlying transport error.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
    },

    /// Configuration error (malformed allowlist pattern, etc.). Carries
    /// the parse / validation source when the misconfiguration was
    /// detected via a typed error.
    #[error("configuration: {message}")]
    Config {
        /// Human-readable summary.
        message: String,
        /// Underlying parse / validation error.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
    },

    /// Calculator parser rejected the input expression.
    #[error("calculator: {0}")]
    Calculator(String),

    /// JSON encode/decode failure.
    #[error(transparent)]
    Serde(#[from] serde_json::Error),
}

impl ToolError {
    /// Wrap any transport error as a [`ToolError::Network`] preserving
    /// the source chain.
    pub fn network<E>(source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Network {
            message: source.to_string(),
            source: Some(Box::new(source)),
        }
    }

    /// Construct a [`ToolError::Network`] from a bare message
    /// (timeout, body cap exceeded, …).
    pub fn network_msg(message: impl Into<String>) -> Self {
        Self::Network {
            message: message.into(),
            source: None,
        }
    }

    /// Wrap a parse/validation error as a [`ToolError::Config`]
    /// preserving the source chain.
    pub fn config<E>(source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Config {
            message: source.to_string(),
            source: Some(Box::new(source)),
        }
    }

    /// Construct a [`ToolError::Config`] from a bare message.
    pub fn config_msg(message: impl Into<String>) -> Self {
        Self::Config {
            message: message.into(),
            source: None,
        }
    }
}

impl From<ToolError> for Error {
    fn from(err: ToolError) -> Self {
        match err {
            ToolError::Config { .. } => Self::config(err.to_string()),
            ToolError::BodyTooLarge { .. } => Self::provider_http_from(413, err),
            ToolError::Network { .. } => Self::provider_network_from(err),
            // Everything else is a caller-side input mismatch.
            ToolError::InvalidInput(_)
            | ToolError::HostBlocked { .. }
            | ToolError::UnsupportedScheme { .. }
            | ToolError::MethodBlocked { .. }
            | ToolError::Calculator(_)
            | ToolError::Serde(_) => Self::invalid_request(err.to_string()),
        }
    }
}
