//! Local error type for `PgGraphMemory`. Surfaces to callers as
//! `entelix_core::Error` when crossing the public `GraphMemory`
//! boundary.

use thiserror::Error;

use entelix_core::error::Error;

/// Result alias used inside `entelix-graphmemory-pg`.
pub type PgGraphMemoryResult<T> = std::result::Result<T, PgGraphMemoryError>;

/// Errors that can surface from `PgGraphMemory`.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PgGraphMemoryError {
    /// Underlying `sqlx` / Postgres failure.
    #[error("postgres transport failure: {0}")]
    Sqlx(#[from] sqlx::Error),

    /// Server returned a row shape inconsistent with what the schema
    /// migration installed. Indicates either a stale
    /// `auto_migrate=false` deployment or external schema drift.
    #[error("malformed row shape: {0}")]
    Malformed(String),

    /// Configuration error — invalid connection string, missing
    /// pool / pool URL, unsafe table name.
    #[error("configuration error: {0}")]
    Config(String),

    /// JSON encode / decode failure on a node / edge payload.
    #[error("payload codec failure: {0}")]
    Codec(#[from] serde_json::Error),
}

impl From<PgGraphMemoryError> for Error {
    fn from(err: PgGraphMemoryError) -> Self {
        match err {
            PgGraphMemoryError::Config(msg) => Self::config(msg),
            PgGraphMemoryError::Codec(e) => Self::Serde(e),
            other => Self::provider_network_from(other),
        }
    }
}
