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

    /// A dispatch (tool body, graph node, or middleware layer) requested
    /// human-in-the-loop intervention. The runtime catches this,
    /// persists a checkpoint at the pre-dispatch state, and returns
    /// `kind` + `payload` to the caller. Resume with
    /// `entelix_graph::CompiledGraph::resume_with`.
    ///
    /// See [`crate::interruption::InterruptionKind`] for the typed
    /// reason taxonomy and [`crate::interrupt`] /
    /// [`crate::interrupt_with`] for the canonical raise sites.
    #[error("dispatch interrupted for human review")]
    Interrupted {
        /// Typed reason — `Custom` for operator-defined pauses,
        /// `ApprovalPending { tool_use_id }` for tool-approval
        /// pauses raised by `ApprovalLayer`, or any future SDK
        /// variant. Operator match sites should carry a fall-through
        /// `_` arm.
        kind: crate::interruption::InterruptionKind,
        /// Operator free-form data describing what the resumer needs
        /// to know. For typed kinds this is often `Value::Null`; for
        /// `Custom` it carries whatever `interrupt(payload)` passed.
        payload: serde_json::Value,
    },

    /// A validator (typed-output `OutputValidator`, tool body, hook)
    /// requested the model retry the current turn with corrective
    /// feedback. Distinct from [`Self::Provider`] (transport
    /// retries — wire-level failure) and [`Self::InvalidRequest`]
    /// (operator misuse) so retry classifiers, OTel dashboards, and
    /// budget meters all branch on a typed signal.
    ///
    /// Catch-and-resume semantics: the surrounding agent or
    /// `complete_typed<O>` loop catches this variant, appends a
    /// `RetryPromptPart` to the conversation carrying `hint`, and
    /// re-invokes the model — counting one increment against
    /// `ChatModelConfig::validation_retries`. Operators that want to
    /// raise this variant build it via [`Error::model_retry`] so
    /// the `RenderedForLlm` funnel (invariant 16) cannot be
    /// bypassed.
    #[error("model retry requested (attempt {attempt})")]
    ModelRetry {
        /// Corrective text the loop will surface to the model on the
        /// retried turn. The `RenderedForLlm` carrier ensures the
        /// payload was filtered through the operator-controlled
        /// rendering funnel rather than copied raw from a
        /// vendor-side error string.
        hint: crate::llm_facing::RenderedForLlm<String>,
        /// Per-call attempt counter. The retry loop stamps this on
        /// emit so the variant is self-describing without callers
        /// tracking attempt state externally. The first retry sees
        /// `attempt = 1`.
        attempt: u32,
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

    /// A `RunBudget` axis was exceeded — request count, token
    /// totals, or tool calls hit the configured limit. The
    /// `axis` field identifies which axis fired; `limit` is the
    /// configured cap; `observed` is the value that breached it.
    /// Distinct from [`Self::Provider`] so retry classifiers can
    /// short-circuit (a budget breach does not retry) and from
    /// [`Self::InvalidRequest`] so dashboards see the budget
    /// signal as a first-class category.
    /// A [`crate::RunBudget`] axis was exceeded. The typed
    /// [`crate::run_budget::UsageLimitBreach`] enum carries both
    /// the breaching axis and its magnitude in one variant — axis
    /// and magnitude are paired by construction so consumers
    /// pattern-match a single value rather than checking the axis
    /// to know which numeric type to read.
    #[error("{0}")]
    UsageLimitExceeded(crate::run_budget::UsageLimitBreach),
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

    /// Build a [`Self::ModelRetry`] from an LLM-rendered hint. The
    /// `attempt` counter starts at zero and is incremented by the
    /// surrounding retry loop on each emit; validators / tools
    /// raising this variant from a fresh call site pass `0` and
    /// trust the loop to stamp the running counter.
    ///
    /// Construction goes through [`crate::llm_facing::RenderedForLlm`] so the
    /// hint is not a free-form `String` — the typed carrier ensures
    /// the message has been routed through the operator's rendering
    /// funnel (invariant 16). Consumers raising this variant from a
    /// validator typically obtain the rendered hint via
    /// `LlmRenderable::for_llm`.
    pub const fn model_retry(
        hint: crate::llm_facing::RenderedForLlm<String>,
        attempt: u32,
    ) -> Self {
        Self::ModelRetry { hint, attempt }
    }

    /// Build an HTTP-class provider error. Use the `_network` /
    /// `_tls` / `_dns` variants for transport-class failures so
    /// retry classifiers see the typed signal rather than a
    /// stringly-typed status code.
    ///
    /// Status `0` / 1xx / 2xx / 3xx / ≥600 do **not** represent a
    /// terminal vendor response. The constructor coerces them to
    /// [`ProviderErrorKind::Network`] so retry classifiers, wire
    /// codes, and dashboards see "we never received a terminal
    /// response" rather than a plausible-looking `upstream_error`
    /// (invariant 15).
    ///
    /// Synthetic-message form: use when the message is composed
    /// from vendor body fields (no source error). For
    /// [`std::error::Error`]-bearing failures, prefer
    /// [`Self::provider_http_from`] which preserves the source
    /// chain.
    pub fn provider_http(status: u16, message: impl Into<String>) -> Self {
        Self::Provider {
            kind: http_or_network(status),
            message: message.into(),
            retry_after: None,
            source: None,
        }
    }

    /// Build an HTTP-class provider error from any
    /// [`std::error::Error`]. Message is `err.to_string()`; the
    /// original error is preserved as `#[source]`. Status coercion
    /// follows [`Self::provider_http`] — non-4xx/5xx statuses
    /// surface as [`ProviderErrorKind::Network`].
    pub fn provider_http_from<E>(status: u16, err: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Provider {
            kind: http_or_network(status),
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

    /// Stable wire identifier for this error — a snake-case ASCII
    /// string suitable as a key into an integrator's i18n catalogue,
    /// metric label, or typed wire envelope.
    ///
    /// Guarantee: **the returned string never changes across patch
    /// versions**. Adding a new [`Error`] variant adds a new wire
    /// code; existing codes are forever-stable. Integrators map one
    /// branch per code instead of parsing `Display` output.
    ///
    /// HTTP provider failures bucket on the status family rather than
    /// the exact code so vendor drift (a new 4xx status added by an
    /// upstream) is silently absorbed into the right class without an
    /// SDK release.
    pub fn wire_code(&self) -> &'static str {
        match self {
            Self::InvalidRequest(_) => "invalid_request",
            Self::Config(_) => "config_error",
            Self::Provider { kind, .. } => match kind {
                ProviderErrorKind::Network => "transport_failure",
                ProviderErrorKind::Tls => "tls_failure",
                ProviderErrorKind::Dns => "dns_failure",
                ProviderErrorKind::Http(status) => match *status {
                    429 => "rate_limited",
                    401 | 403 => "upstream_unauthorized",
                    s if (400..500).contains(&s) => "upstream_invalid",
                    s if (500..600).contains(&s) => "upstream_unavailable",
                    _ => "upstream_error",
                },
            },
            Self::Auth(_) => "auth_failed",
            Self::Cancelled => "cancelled",
            Self::DeadlineExceeded => "deadline_exceeded",
            Self::Interrupted { .. } => "interrupted",
            Self::ModelRetry { .. } => "model_retry_exhausted",
            Self::Serde(_) => "serde",
            Self::UsageLimitExceeded(_) => "quota_exceeded",
        }
    }

    /// Coarse responsibility class — `Client` for caller-side failures
    /// (bad input, expired credentials, exceeded quota) and `Server`
    /// for SDK/vendor-side failures (deployment misconfiguration,
    /// upstream unavailability, transport breakage).
    ///
    /// Orthogonal to retry semantics: a `Server`-class error may still
    /// be permanent and a `Client`-class error may still be a
    /// transient rate limit. Retry decisions consume the typed
    /// `Error::Provider::retry_after` and `RetryClassifier` surfaces
    /// (invariant 17), not this method.
    pub fn wire_class(&self) -> ErrorClass {
        match self {
            Self::InvalidRequest(_)
            | Self::Auth(_)
            | Self::Cancelled
            | Self::Interrupted { .. }
            | Self::ModelRetry { .. }
            | Self::UsageLimitExceeded(_) => ErrorClass::Client,
            Self::Config(_) | Self::DeadlineExceeded | Self::Serde(_) => ErrorClass::Server,
            Self::Provider { kind, .. } => match kind {
                ProviderErrorKind::Network | ProviderErrorKind::Tls | ProviderErrorKind::Dns => {
                    ErrorClass::Server
                }
                ProviderErrorKind::Http(status) => match *status {
                    s if (400..500).contains(&s) => ErrorClass::Client,
                    _ => ErrorClass::Server,
                },
            },
        }
    }
}

/// Coarse responsibility class for an [`Error`]. Two values by design —
/// "transient" / "permanent" is a retry-policy axis, orthogonal to
/// responsibility, and surfaced via [`Error::Provider`]'s
/// `retry_after` field plus the `RetryClassifier` policy surface.
///
/// Maps onto the standard HTTP family split: `Client` ≈ 4xx-equivalent
/// (caller / integrator can act to fix), `Server` ≈ 5xx-equivalent
/// (vendor or deployment must act).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum ErrorClass {
    /// The caller — request shape, credentials, quota, cancellation
    /// choice — is the actor that can resolve the failure.
    Client,
    /// The SDK, vendor, or deployment environment is the actor that
    /// can resolve the failure.
    Server,
}

impl std::fmt::Display for ErrorClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Client => f.write_str("client"),
            Self::Server => f.write_str("server"),
        }
    }
}

/// Coerce a raw `u16` HTTP status into a typed
/// [`ProviderErrorKind`]. 4xx / 5xx surface as
/// [`ProviderErrorKind::Http`]; every other value collapses to
/// [`ProviderErrorKind::Network`] because the SDK never received a
/// terminal vendor response (invariant 15 — no silent fallback to
/// a plausible-looking `upstream_error`).
const fn http_or_network(status: u16) -> ProviderErrorKind {
    if status >= 400 && status < 600 {
        ProviderErrorKind::Http(status)
    } else {
        ProviderErrorKind::Network
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
