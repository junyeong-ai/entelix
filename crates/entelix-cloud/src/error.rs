//! Crate-local error type. Surfaces to callers as
//! `entelix_core::Error::Provider` (or `Config`) at the Transport
//! boundary.

use thiserror::Error;

use entelix_core::error::Error;

/// Failures emitted by cloud-layer machinery (credential resolution,
/// signing, refreshable token plumbing, binary frame parsing).
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CloudError {
    /// Credential resolver returned an error or no credentials.
    #[error("credential resolution failed: {message}")]
    Credential {
        /// Human-readable summary.
        message: String,
        /// Underlying credential-provider error.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
    },

    /// SigV4 / AAD signing failed.
    #[error("signing failed: {message}")]
    Signing {
        /// Human-readable summary.
        message: String,
        /// Underlying signer error.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
    },

    /// Network / HTTP failure.
    #[error("network failure: {message}")]
    Network {
        /// Human-readable summary.
        message: String,
        /// Underlying transport error.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
    },

    /// Configuration error — missing region, malformed URL, etc.
    #[error("configuration error: {0}")]
    Config(String),

    /// Event-stream frame parser failure (Bedrock binary protocol).
    #[cfg(feature = "aws")]
    #[cfg_attr(docsrs, doc(cfg(feature = "aws")))]
    #[error("event-stream decode failed: {0}")]
    EventStream(#[from] crate::bedrock::event_stream::EventStreamParseError),
}

impl CloudError {
    /// Wrap a credential-provider failure preserving the source chain.
    pub fn credential<E>(source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Credential {
            message: source.to_string(),
            source: Some(Box::new(source)),
        }
    }

    /// Construct a [`CloudError::Credential`] from a bare message.
    pub fn credential_msg(message: impl Into<String>) -> Self {
        Self::Credential {
            message: message.into(),
            source: None,
        }
    }

    /// Wrap a signer failure preserving the source chain.
    pub fn signing<E>(source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Signing {
            message: source.to_string(),
            source: Some(Box::new(source)),
        }
    }

    /// Construct a [`CloudError::Signing`] from a bare message.
    pub fn signing_msg(message: impl Into<String>) -> Self {
        Self::Signing {
            message: message.into(),
            source: None,
        }
    }

    /// Wrap a transport failure preserving the source chain.
    pub fn network<E>(source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Network {
            message: source.to_string(),
            source: Some(Box::new(source)),
        }
    }

    /// Construct a [`CloudError::Network`] from a bare message.
    pub fn network_msg(message: impl Into<String>) -> Self {
        Self::Network {
            message: message.into(),
            source: None,
        }
    }
}

impl From<CloudError> for Error {
    fn from(err: CloudError) -> Self {
        match err {
            CloudError::Config(msg) => Self::config(msg),
            other => Self::provider_network_from(other),
        }
    }
}
