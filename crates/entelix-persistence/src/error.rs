//! Crate-internal error type. Public APIs map this back into
//! `entelix_core::Error::Persistence` (or `Provider`) at the
//! orchestration layer.

use thiserror::Error;

/// Result alias used inside `entelix-persistence`.
pub type PersistenceResult<T> = std::result::Result<T, PersistenceError>;

/// Errors that the persistence layer surfaces to its callers.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PersistenceError {
    /// The lock was held by another process when a non-blocking
    /// `try_acquire` returned. Carries the lock key for diagnostics.
    #[error("lock held: {key}")]
    LockHeld {
        /// Lock key that could not be acquired.
        key: String,
    },

    /// The lock could not be acquired within the configured deadline.
    #[error("lock acquire timed out for key '{key}' after {attempts} attempts")]
    LockAcquireTimeout {
        /// Lock key that timed out.
        key: String,
        /// Attempt count that was exhausted.
        attempts: u32,
    },

    /// Schema version mismatch — a payload tagged with a version this
    /// build does not understand.
    #[error(
        "schema version mismatch: payload version {payload}, build version range \
         [{min}, {current}]"
    )]
    SchemaVersionMismatch {
        /// Version observed in the payload.
        payload: u32,
        /// Lowest version this build accepts.
        min: u32,
        /// Highest version this build understands.
        current: u32,
    },

    /// Configuration mistake — connection string malformed, builder
    /// missing required field, etc.
    #[error("configuration error: {0}")]
    Config(String),

    /// Generic backend failure — `sqlx` / `redis` error, network
    /// drop, malformed payload. Contains the original error message.
    #[error("backend failure: {0}")]
    Backend(String),

    /// JSON encode/decode failure on a stored payload.
    #[error(transparent)]
    Serde(#[from] serde_json::Error),
}

impl From<PersistenceError> for entelix_core::Error {
    fn from(err: PersistenceError) -> Self {
        match err {
            PersistenceError::LockHeld { .. } | PersistenceError::LockAcquireTimeout { .. } => {
                Self::invalid_request(err.to_string())
            }
            PersistenceError::SchemaVersionMismatch { .. } => Self::config(err.to_string()),
            PersistenceError::Config(msg) => Self::config(msg),
            PersistenceError::Backend(msg) => Self::provider_network(msg),
            PersistenceError::Serde(e) => Self::Serde(e),
        }
    }
}
