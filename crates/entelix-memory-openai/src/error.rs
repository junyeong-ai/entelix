//! Local error type. Surfaces to callers as `entelix_core::Error::Provider`
//! when crossing the public `Embedder::embed` boundary — the embedder's
//! `Result` alias `entelix_core::Result` ultimately consumes this.

use thiserror::Error;

use entelix_core::error::Error;

/// Result alias used inside `entelix-memory-openai`.
pub type OpenAiEmbedderResult<T> = std::result::Result<T, OpenAiEmbedderError>;

/// Errors that can surface from `OpenAiEmbedder`.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum OpenAiEmbedderError {
    /// Network / HTTP failure reaching the OpenAI API. Carries the
    /// underlying `reqwest::Error` chain.
    #[error("network failure: {message}")]
    Network {
        /// Human-readable summary.
        message: String,
        /// Underlying transport error.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
    },

    /// Non-2xx HTTP status from OpenAI. Body is truncated to keep
    /// log volume bounded under malicious upstream payloads.
    #[error("HTTP {status}: {body}")]
    HttpStatus {
        /// HTTP status code.
        status: u16,
        /// Truncated response body.
        body: String,
    },

    /// Server returned a malformed JSON body (missing `data`, vector
    /// dimension mismatch, etc.).
    #[error("malformed response: {0}")]
    Malformed(String),

    /// Configuration error — invalid base URL, missing credentials,
    /// dimension mismatch at build time.
    #[error("configuration error: {0}")]
    Config(String),

    /// Credential resolution failed.
    #[error(transparent)]
    Credential(Error),

    /// JSON encode/decode failure.
    #[error(transparent)]
    Serde(#[from] serde_json::Error),
}

impl OpenAiEmbedderError {
    /// Wrap any transport-layer error as a [`OpenAiEmbedderError::Network`].
    pub fn network<E>(source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Network {
            message: source.to_string(),
            source: Some(Box::new(source)),
        }
    }
}

impl From<OpenAiEmbedderError> for Error {
    fn from(err: OpenAiEmbedderError) -> Self {
        match err {
            OpenAiEmbedderError::Config(msg) => Self::config(msg),
            OpenAiEmbedderError::Credential(e) => e,
            other => Self::provider_network_from(other),
        }
    }
}
