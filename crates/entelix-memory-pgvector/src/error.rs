//! Local error type. Surfaces to callers as
//! `entelix_core::Error::Provider` when crossing the public
//! `VectorStore` boundary.

use thiserror::Error;

use entelix_core::error::Error;

/// Result alias used inside `entelix-memory-pgvector`.
pub type PgVectorStoreResult<T> = std::result::Result<T, PgVectorStoreError>;

/// Errors that can surface from `PgVectorStore`.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PgVectorStoreError {
    /// Underlying sqlx / Postgres failure.
    #[error("postgres transport failure: {0}")]
    Sqlx(#[from] sqlx::Error),

    /// Server returned a row shape inconsistent with what the
    /// schema migration installed. Indicates either a stale
    /// `auto_migrate=false` deployment or external schema drift.
    #[error("malformed row shape: {0}")]
    Malformed(String),

    /// Configuration error — invalid connection string, dimension
    /// mismatch at build time, missing pool / pool URL.
    #[error("configuration error: {0}")]
    Config(String),

    /// `VectorFilter` carries a JSON value the SQL projection
    /// cannot express (e.g. a non-numeric value supplied to a
    /// `Range` filter).
    #[error("filter projection error: {0}")]
    FilterProjection(String),
}

impl From<PgVectorStoreError> for Error {
    fn from(err: PgVectorStoreError) -> Self {
        match err {
            PgVectorStoreError::Config(msg) | PgVectorStoreError::FilterProjection(msg) => {
                Self::config(msg)
            }
            other => Self::provider_network_from(other),
        }
    }
}
