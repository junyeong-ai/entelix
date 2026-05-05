//! Top-level error type for `entelix-core` and the public API surface of the
//! facade crate.
//!
//! Conventions (see CLAUDE.md §"Error conventions"):
//! - Public crate APIs surface `entelix_core::Error`. Module-internal errors
//!   are typed enums (e.g. `CodecError`) that bubble up via `From` chains.
//! - Provider failures carry a typed `kind: ProviderErrorKind` (Network /
//!   Tls / Dns / Http(status)) — retry classifiers branch on the typed
//!   signal, not on parsed strings.
//! - `Result<T> = std::result::Result<T, Error>`.

use std::borrow::Cow;
use std::time::Duration;

use crate::auth::AuthError;

/// Convenience alias used across `entelix-core` and re-exported by the facade.
pub type Result<T> = core::result::Result<T, Error>;

/// Aggregate error returned from public entelix-core APIs.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Caller supplied an invalid request before any provider was contacted —
    /// e.g. empty message list, missing required field, schema mismatch.
    #[error("invalid request: {0}")]
    InvalidRequest(Cow<'static, str>),

    /// Configuration error detected at construction time (builders, factories,
    /// crate-init code).
    #[error("config error: {0}")]
    Config(Cow<'static, str>),

    /// Provider failure. `kind` distinguishes transport-class
    /// failures (network / TLS / DNS) from HTTP-class failures
    /// (4xx / 5xx) so retry classifiers can act on the typed signal
    /// rather than parsing strings or reading a `status: 0`
    /// sentinel. `retry_after` carries the vendor's `Retry-After`
    /// hint when present — the retry layer honours it ahead of its
    /// own backoff (invariant #17 — vendor authoritative signal
    /// beats self-jitter).
    #[error("provider {kind}: {message}")]
    Provider {
        /// Failure category — `Network`, `Tls`, `Dns`, or
        /// `Http(status)`.
        kind: ProviderErrorKind,
        /// Provider-supplied message, normalized to a string.
        message: String,
        /// Vendor `Retry-After` hint when present. Capped at the
        /// transport's parsing limit so a malicious vendor cannot
        /// pin a retry loop forever.
        #[allow(dead_code)]
        retry_after: Option<Duration>,
        /// Underlying error (transport / parser / signer) preserved
        /// for operator diagnostics. Walk it via [`std::error::Error::source`]
        /// or `{:?}`; the LLM-facing channel never sees it
        /// (invariant 16 — `LlmRenderable::render_for_llm` strips
        /// source chains).
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
    },

    /// The operation was cancelled via the `ExecutionContext` cancellation token.
    #[error("operation cancelled")]
    Cancelled,

    /// The operation hit the deadline carried by `ExecutionContext`.
    #[error("deadline exceeded")]
    DeadlineExceeded,

    /// A graph node requested human-in-the-loop intervention. The graph
    /// executor catches this, persists a checkpoint at the pre-node
    /// state, and returns the payload to the caller. Resume with
    /// `entelix_graph::CompiledGraph::resume_with`.
    #[error("graph interrupted for human review")]
    Interrupted {
        /// Caller-visible payload describing what input the node needs
        /// (e.g. an approval question, tool input awaiting human review).
        payload: serde_json::Value,
    },

    /// JSON serialization or deserialization failed at an entelix-managed
    /// boundary (codec, tool I/O, persistence write/read).
    #[error(transparent)]
    Serde(#[from] serde_json::Error),

    /// Credential resolution or use failed. Distinct from
    /// [`Self::Provider`] so retry policies and dashboards can
    /// distinguish "the model is down" from "our key is bad" without
    /// pattern-matching on error messages.
    #[error(transparent)]
    Auth(AuthError),
}

impl Error {
    /// Build an `InvalidRequest` from a static or owned string.
    pub fn invalid_request(msg: impl Into<Cow<'static, str>>) -> Self {
        Self::InvalidRequest(msg.into())
    }

    /// Build a `Config` error from a static or owned string.
    pub fn config(msg: impl Into<Cow<'static, str>>) -> Self {
        Self::Config(msg.into())
    }

    /// Build an HTTP-class provider error (4xx / 5xx). Use the
    /// `_network` / `_tls` / `_dns` variants for transport-class
    /// failures so retry classifiers see the typed signal rather
    /// than a stringly-typed status code.
    ///
    /// Synthetic-message form: use when the message is composed
    /// from vendor body fields (no source error). For
    /// [`std::error::Error`]-bearing failures, prefer
    /// [`Self::provider_http_from`] which preserves the source
    /// chain.
    pub fn provider_http(status: u16, message: impl Into<String>) -> Self {
        Self::Provider {
            kind: ProviderErrorKind::Http(status),
            message: message.into(),
            retry_after: None,
            source: None,
        }
    }

    /// Build an HTTP-class provider error from any
    /// [`std::error::Error`]. Message is `err.to_string()`; the
    /// original error is preserved as `#[source]`.
    pub fn provider_http_from<E>(status: u16, err: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Provider {
            kind: ProviderErrorKind::Http(status),
            message: err.to_string(),
            retry_after: None,
            source: Some(Box::new(err)),
        }
    }

    /// Build a network-class provider error (connect refused, read
    /// reset, peer hangup before HTTP framing). Distinguishes
    /// "vendor returned a 5xx" from "we never spoke to vendor".
    ///
    /// Synthetic-message form: use when no source error exists
    /// (e.g. vendor wire-format prose lifted from a JSON body).
    /// Source-bearing form: [`Self::provider_network_from`] derives
    /// the message from the source's `Display` and stores the source
    /// for `{:?}` walks (preferred for `map_err` chains).
    pub fn provider_network(message: impl Into<String>) -> Self {
        Self::Provider {
            kind: ProviderErrorKind::Network,
            message: message.into(),
            retry_after: None,
            source: None,
        }
    }

    /// Build a network-class provider error from any
    /// [`std::error::Error`]. Message is `err.to_string()`; the
    /// original error is preserved as `#[source]` so operator
    /// diagnostics walk the full chain. Pairs with `.map_err`:
    ///
    /// ```ignore
    /// http_req.send().await.map_err(Error::provider_network_from)?;
    /// ```
    pub fn provider_network_from<E>(err: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Provider {
            kind: ProviderErrorKind::Network,
            message: err.to_string(),
            retry_after: None,
            source: Some(Box::new(err)),
        }
    }

    /// Build a TLS-class provider error (handshake failure,
    /// certificate validation, protocol mismatch).
    pub fn provider_tls(message: impl Into<String>) -> Self {
        Self::Provider {
            kind: ProviderErrorKind::Tls,
            message: message.into(),
            retry_after: None,
            source: None,
        }
    }

    /// Build a TLS-class provider error from any
    /// [`std::error::Error`]. Message is `err.to_string()`; the
    /// original error is preserved as `#[source]`.
    pub fn provider_tls_from<E>(err: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Provider {
            kind: ProviderErrorKind::Tls,
            message: err.to_string(),
            retry_after: None,
            source: Some(Box::new(err)),
        }
    }

    /// Build a DNS-class provider error (name resolution failure,
    /// SSRF allowlist rejection at the resolver).
    pub fn provider_dns(message: impl Into<String>) -> Self {
        Self::Provider {
            kind: ProviderErrorKind::Dns,
            message: message.into(),
            retry_after: None,
            source: None,
        }
    }

    /// Build a DNS-class provider error from any
    /// [`std::error::Error`]. Message is `err.to_string()`; the
    /// original error is preserved as `#[source]`.
    pub fn provider_dns_from<E>(err: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Provider {
            kind: ProviderErrorKind::Dns,
            message: err.to_string(),
            retry_after: None,
            source: Some(Box::new(err)),
        }
    }

    /// Attach a `Retry-After` duration to a provider error. The
    /// duration arrives from the vendor's `Retry-After` response
    /// header (or equivalent body field). Returns `self` unchanged
    /// for non-`Provider` variants — callers know the variant they
    /// constructed, so this is `Self -> Self` rather than a typed
    /// projection.
    #[must_use]
    pub fn with_retry_after(mut self, duration: Duration) -> Self {
        if let Self::Provider {
            ref mut retry_after,
            ..
        } = self
        {
            *retry_after = Some(duration);
        }
        self
    }

    /// Attach the underlying error as the `Provider` variant's source
    /// chain, preserving root-cause context for operator diagnostics
    /// (`{:?}` / [`std::error::Error::source`] walk). Returns `self`
    /// unchanged for non-`Provider` variants.
    ///
    /// Channel-separation guarantee (invariant 16): the source chain
    /// is operator-only. [`crate::LlmRenderable::render_for_llm`]
    /// strips it for LLM-facing renderings; sinks / OTel / logs keep
    /// the full diagnostic.
    #[must_use]
    pub fn with_source<E>(mut self, err: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        if let Self::Provider { ref mut source, .. } = self {
            *source = Some(Box::new(err));
        }
        self
    }
}

/// Provider failure category — distinguishes transport-class
/// failures (the SDK never received a complete HTTP framing) from
/// HTTP-class failures (the vendor responded with a status). Retry
/// classifiers use this to make typed decisions rather than
/// pattern-matching on `status: 0` sentinels (invariant #17).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum ProviderErrorKind {
    /// Connect refused, read reset, peer hangup before HTTP framing
    /// completed.
    Network,
    /// TLS handshake failure, certificate validation failure,
    /// protocol mismatch.
    Tls,
    /// DNS resolution failure or SSRF allowlist rejection at the
    /// resolver.
    Dns,
    /// Vendor responded with an HTTP status. Carries the actual
    /// numeric code so classifiers can branch on `408|425|429|5xx`.
    Http(u16),
}

impl std::fmt::Display for ProviderErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Network => f.write_str("network"),
            Self::Tls => f.write_str("tls"),
            Self::Dns => f.write_str("dns"),
            Self::Http(status) => write!(f, "returned {status}"),
        }
    }
}
