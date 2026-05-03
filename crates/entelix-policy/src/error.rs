//! Policy-layer errors. The hook integration translates these to
//! [`entelix_core::Error::Provider`] with `kind: ProviderErrorKind::Http(429)`
//! (rate / quota) or [`entelix_core::Error::Config`] (misconfiguration).

use thiserror::Error;

use entelix_core::error::Error;

/// Result alias used inside `entelix-policy`.
pub type PolicyResult<T> = std::result::Result<T, PolicyError>;

/// Policy-layer failures.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PolicyError {
    /// Rate limit denied — caller must retry after the indicated delay.
    #[error("rate limit exceeded for key '{key}': retry after {retry_after_ms} ms")]
    RateLimited {
        /// Bucket key (typically tenant + endpoint).
        key: String,
        /// How long until enough tokens accumulate to satisfy the
        /// requested cost.
        retry_after_ms: u64,
    },

    /// Cumulative spend has reached the configured ceiling.
    #[error("budget exhausted for tenant '{tenant}': spent {spent}, ceiling {ceiling}")]
    BudgetExhausted {
        /// Tenant identifier whose ledger overflowed.
        tenant: String,
        /// Cumulative spend at refusal time.
        spent: rust_decimal::Decimal,
        /// Configured ceiling.
        ceiling: rust_decimal::Decimal,
    },

    /// Pricing table has no entry for the named model.
    #[error("no pricing configured for model '{0}'")]
    UnknownModel(String),

    /// Configuration is malformed (e.g. invalid regex pattern).
    #[error("policy configuration error: {0}")]
    Config(String),
}

impl From<PolicyError> for Error {
    fn from(err: PolicyError) -> Self {
        match err {
            PolicyError::RateLimited { retry_after_ms, .. } => {
                Self::provider_http_from(429, err)
                    .with_retry_after(std::time::Duration::from_millis(retry_after_ms))
            }
            PolicyError::BudgetExhausted { .. } => Self::provider_http_from(402, err),
            PolicyError::UnknownModel(_) | PolicyError::Config(_) => Self::config(err.to_string()),
        }
    }
}
