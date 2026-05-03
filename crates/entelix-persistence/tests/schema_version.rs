//! `SessionSchemaVersion::validate` rejects future + below-min
//! payloads, accepts the current and supported range.

#![allow(clippy::unwrap_used)]

use entelix_persistence::SessionSchemaVersion;
use entelix_persistence::schema_version::{CURRENT_VERSION, MIN_SUPPORTED_VERSION};

#[test]
fn current_validates() {
    assert!(SessionSchemaVersion::CURRENT.validate().is_ok());
    assert_eq!(SessionSchemaVersion::CURRENT.raw(), CURRENT_VERSION);
}

#[test]
fn future_version_rejected() {
    let future = SessionSchemaVersion(CURRENT_VERSION + 1);
    let err = future.validate().unwrap_err();
    assert!(matches!(
        err,
        entelix_persistence::PersistenceError::SchemaVersionMismatch { .. }
    ));
}

#[test]
fn below_min_rejected() {
    if MIN_SUPPORTED_VERSION == 0 {
        // Cannot construct a payload below 0; skip this case while
        // MIN_SUPPORTED is still at the floor.
        return;
    }
    let stale = SessionSchemaVersion(MIN_SUPPORTED_VERSION - 1);
    assert!(stale.validate().is_err());
}

#[test]
fn default_is_current() {
    assert_eq!(
        SessionSchemaVersion::default(),
        SessionSchemaVersion::CURRENT
    );
}
