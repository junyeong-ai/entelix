//! Shared validation for stable SDK identifiers.

use crate::error::{Error, Result};

/// Validate a caller-supplied configuration identifier.
///
/// This accepts stable provider/user identifiers such as model names,
/// graph node names, tool names, tenant ids, routing keys, and cache
/// handles while rejecting ambiguous empty, padded, or control-character
/// values. Free-form prose, URIs, and human display labels should use
/// domain-specific validation instead of this helper.
pub fn validate_config_identifier(surface: &str, field: &str, value: &str) -> Result<()> {
    validate_identifier_parts(value, |detail| {
        Error::config(format!("{surface}: {field} {detail}"))
    })
}

pub(crate) fn validate_request_identifier(path: &str, value: &str) -> Result<()> {
    validate_identifier_parts(value, |detail| {
        Error::invalid_request(format!("{path} {detail}"))
    })
}

fn validate_identifier_parts(value: &str, error: impl FnOnce(&'static str) -> Error) -> Result<()> {
    if value.trim().is_empty() {
        return Err(error("must not be empty"));
    }
    if value.trim() != value {
        return Err(error("must not have leading or trailing whitespace"));
    }
    if value.chars().any(char::is_control) {
        return Err(error("must not contain control characters"));
    }
    Ok(())
}
