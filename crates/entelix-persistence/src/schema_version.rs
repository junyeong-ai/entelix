//! Persisted-payload schema version stamps.
//!
//! Every JSON blob written through this crate carries a
//! [`SessionSchemaVersion`] tag. On read, payloads with versions
//! outside the build's accepted range surface as
//! [`crate::PersistenceError::SchemaVersionMismatch`]. Silent
//! degradation is forbidden — a future-version payload is rejected
//! hard so a downgrade never corrupts state by writing back a
//! lower-version blob.

use serde::{Deserialize, Serialize};

use crate::error::{PersistenceError, PersistenceResult};

/// Highest schema version this build can read and write.
pub const CURRENT_VERSION: u32 = 1;

/// Lowest schema version this build still understands. When
/// migrations land, bump this only after the migration ladder writes
/// every prior payload to the new shape.
pub const MIN_SUPPORTED_VERSION: u32 = 1;

/// Schema version tag for persisted payloads.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionSchemaVersion(pub u32);

impl SessionSchemaVersion {
    /// The version this build emits.
    pub const CURRENT: Self = Self(CURRENT_VERSION);

    /// Validate that `payload` is in `[MIN_SUPPORTED_VERSION,
    /// CURRENT_VERSION]`. Errors are loud — never silently downgrade.
    pub fn validate(self) -> PersistenceResult<()> {
        if self.0 < MIN_SUPPORTED_VERSION || self.0 > CURRENT_VERSION {
            return Err(PersistenceError::SchemaVersionMismatch {
                payload: self.0,
                min: MIN_SUPPORTED_VERSION,
                current: CURRENT_VERSION,
            });
        }
        Ok(())
    }

    /// Raw integer.
    pub const fn raw(self) -> u32 {
        self.0
    }
}

impl Default for SessionSchemaVersion {
    fn default() -> Self {
        Self::CURRENT
    }
}
