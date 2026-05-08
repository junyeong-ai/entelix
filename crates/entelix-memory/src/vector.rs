//! Shared vector validation helpers for embedder and vector-store boundaries.

use entelix_core::{Error, Result};

/// Return the first non-finite vector element, if any.
///
/// Provider-backed embedders use this to classify malformed provider
/// responses, while vector stores use [`validate_vector_shape`] to
/// reject invalid caller-supplied vectors before they reach an index.
#[must_use]
pub fn first_non_finite_vector_value(vector: &[f32]) -> Option<(usize, f32)> {
    vector
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
}

/// Validate vector dimension and finite values for vector-store calls.
///
/// Reach for this on the caller-input boundary of every
/// `VectorStore` impl (`add` / `search` / `update`) so all
/// backends reject the same two malformations — wrong dimension
/// and non-finite element — with identical error wording.
/// Returns [`Error::InvalidRequest`] because vectors passed into a
/// [`crate::VectorStore`] are caller input at the store boundary.
pub fn validate_vector_shape(
    surface: &str,
    label: &str,
    vector: &[f32],
    expected_dimension: usize,
) -> Result<()> {
    if vector.len() != expected_dimension {
        return Err(Error::invalid_request(format!(
            "{surface}: {label} dimension {} does not match index dimension {expected_dimension}",
            vector.len()
        )));
    }
    if let Some((index, value)) = first_non_finite_vector_value(vector) {
        return Err(Error::invalid_request(format!(
            "{surface}: {label} contains non-finite value at index {index}: {value}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{first_non_finite_vector_value, validate_vector_shape};
    use entelix_core::Error;

    #[test]
    fn first_non_finite_vector_value_reports_position_and_value() {
        assert_eq!(first_non_finite_vector_value(&[1.0, 2.0]), None);
        let Some((index, value)) = first_non_finite_vector_value(&[1.0, f32::INFINITY, f32::NAN])
        else {
            panic!("expected non-finite vector value");
        };
        assert_eq!(index, 1);
        assert!(value.is_infinite());
    }

    #[test]
    fn validate_vector_shape_rejects_dimension_mismatch_and_non_finite_values() {
        assert!(validate_vector_shape("Store::add", "vector", &[1.0, 2.0], 2).is_ok());

        let Err(err) = validate_vector_shape("Store::add", "vector", &[1.0], 2) else {
            panic!("dimension mismatch should be rejected");
        };
        assert!(matches!(err, Error::InvalidRequest(msg) if msg.contains("dimension")));

        let Err(err) = validate_vector_shape("Store::add", "vector", &[f32::NAN, 2.0], 2) else {
            panic!("non-finite vector should be rejected");
        };
        assert!(matches!(err, Error::InvalidRequest(msg) if msg.contains("non-finite")));
    }
}
