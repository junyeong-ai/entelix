//! HTTP-layer error types.
//!
//! The crate ships *two* error types because they serve different
//! audiences and never appear in the same code path:
//!
//! - [`BuildError`] — surfaced by [`crate::AgentRouterBuilder::build`]
//!   to the *operator* at process startup. A `BuildError` means the
//!   router could not be constructed; there is nothing to serve. It
//!   never traverses an HTTP response.
//! - [`ServerError`] — surfaced by request handlers to the *caller*
//!   at request time. Maps to a JSON envelope of the shape
//!
//!   ```json
//!   { "error": { "kind": "...", "message": "..." } }
//!   ```
//!
//! Splitting them keeps the `IntoResponse` mapping focused on
//! request-time failures (no dead status codes) and lets `BuildError`
//! diagnostics stay startup-shaped (operator log, not JSON envelope).

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use thiserror::Error;

use entelix_core::Error;

/// Result alias used by every handler in `entelix-server`.
pub type ServerResult<T> = std::result::Result<T, ServerError>;

/// Result alias used by [`crate::AgentRouterBuilder::build`].
pub type BuildResult<T> = std::result::Result<T, BuildError>;

/// Construction-time failure surfaced by [`crate::AgentRouterBuilder::build`].
///
/// Never traverses an HTTP response — the router has not been
/// built yet, so there is nothing to serve. Operators see
/// `BuildError` at startup; the appropriate reaction is to fix
/// the configuration and retry, not to recover.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum BuildError {
    /// Tenant header configuration is invalid. The bytes passed to
    /// [`crate::AgentRouterBuilder::with_tenant_header`] do not parse
    /// as an HTTP header name. Validation is deferred from the
    /// builder chain to `build()` so the chain stays infallible
    /// (Tower middleware idiom).
    #[error("tenant header `{name}` is not a valid HTTP header name")]
    InvalidTenantHeader {
        /// The bytes the caller passed to `with_tenant_header`.
        name: String,
    },
}

/// HTTP-layer error returned by request handlers.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ServerError {
    /// Wrap a core SDK error.
    #[error(transparent)]
    Core(#[from] Error),

    /// Caller supplied a malformed body or path parameter.
    #[error("bad request: {0}")]
    BadRequest(String),

    /// Resource the caller asked about does not exist (e.g. unknown
    /// thread when wake / checkpoint endpoints arrive).
    #[error("not found: {0}")]
    NotFound(String),

    /// Required tenant header was absent from a request. Surfaces only
    /// when the router was built with `with_tenant_header(name)`
    /// (multi-tenant mode); single-tenant deployments that omit
    /// `with_tenant_header` never see this variant.
    #[error("missing required tenant header `{header}`")]
    MissingTenantHeader {
        /// Name of the header the router was configured to extract.
        header: String,
    },
}

impl ServerError {
    fn status(&self) -> StatusCode {
        match self {
            Self::BadRequest(_) | Self::MissingTenantHeader { .. } => StatusCode::BAD_REQUEST,
            Self::NotFound(_) => StatusCode::NOT_FOUND,
            Self::Core(err) => match err {
                Error::InvalidRequest(_) => StatusCode::BAD_REQUEST,
                Error::Provider { kind, .. } => match kind {
                    entelix_core::ProviderErrorKind::Http(status) => {
                        StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_GATEWAY)
                    }
                    // Network / TLS / DNS failures surface as 502
                    // Bad Gateway — the upstream is reachable in
                    // principle but this attempt did not produce a
                    // response.
                    _ => StatusCode::BAD_GATEWAY,
                },
                Error::Interrupted { .. } => StatusCode::ACCEPTED,
                // Cancellation usually means the client disconnected
                // mid-request; nginx convention is 499 (client closed
                // request). HTTP doesn't define 499 in any RFC, so we
                // construct it explicitly.
                Error::Cancelled => {
                    StatusCode::from_u16(499).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
                }
                // Deadline overrun is the gateway's failure to produce
                // a response in time — RFC 7231 §6.6.5.
                Error::DeadlineExceeded => StatusCode::GATEWAY_TIMEOUT,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            },
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Self::BadRequest(_) => "bad_request",
            Self::NotFound(_) => "not_found",
            Self::MissingTenantHeader { .. } => "missing_tenant_header",
            Self::Core(err) => match err {
                Error::InvalidRequest(_) => "invalid_request",
                Error::Config(_) => "config",
                Error::Provider { .. } => "provider",
                Error::Interrupted { .. } => "interrupted",
                Error::Cancelled => "cancelled",
                Error::DeadlineExceeded => "deadline_exceeded",
                Error::Serde(_) => "serde",
                _ => "internal",
            },
        }
    }
}

#[derive(Debug, Serialize)]
struct ErrorBody<'a> {
    error: ErrorEnvelope<'a>,
}

#[derive(Debug, Serialize)]
struct ErrorEnvelope<'a> {
    kind: &'a str,
    message: String,
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let status = self.status();
        let kind = self.kind();
        let body = ErrorBody {
            error: ErrorEnvelope {
                kind,
                message: self.to_string(),
            },
        };
        (status, Json(body)).into_response()
    }
}
