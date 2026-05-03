//! Local error type. Surfaces to callers as
//! `entelix_core::Error::Provider` when crossing the public
//! `VectorStore` boundary.

use thiserror::Error;

use entelix_core::error::Error;

/// Result alias used inside `entelix-memory-qdrant`.
pub type QdrantStoreResult<T> = std::result::Result<T, QdrantStoreError>;

/// Errors that can surface from `QdrantVectorStore`.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum QdrantStoreError {
    /// gRPC / transport failure reaching the Qdrant server.
    #[error("qdrant transport failure: {0}")]
    Transport(#[from] qdrant_client::QdrantError),

    /// Server returned a malformed or unexpected response shape.
    #[error("malformed qdrant response: {0}")]
    Malformed(String),

    /// Configuration error — invalid URL, dimension mismatch at
    /// build time, missing required collection name.
    #[error("configuration error: {0}")]
    Config(String),

    /// `VectorFilter` carries a JSON value the qdrant projection
    /// cannot represent (e.g. a non-numeric value supplied to a
    /// `Range` filter). The audit suggests the operator-side
    /// filter taxonomy or coerce the value before calling.
    #[error("filter projection error: {0}")]
    FilterProjection(String),
}

impl From<QdrantStoreError> for Error {
    fn from(err: QdrantStoreError) -> Self {
        match err {
            QdrantStoreError::Config(msg) | QdrantStoreError::FilterProjection(msg) => {
                Self::config(msg)
            }
            other => Self::provider_network_from(other),
        }
    }
}
