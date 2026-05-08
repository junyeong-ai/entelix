//! `McpServerConfig` — connection details for one MCP server.
//!
//! HTTP-only by design (invariant 9). Operators wanting stdio servers
//! wrap them externally and expose an HTTP endpoint.
//!
//! ## Server name format
//!
//! Server names route MCP tool dispatches and namespace the tools
//! they publish (`mcp:{server}:{tool}` qualified-name format). The
//! constructor validates the name against a conservative character
//! set so the qualified name is collision-proof and can travel
//! through arbitrary log pipelines without re-escaping:
//!
//! - Non-empty
//! - ASCII alphanumeric, dash (`-`), underscore (`_`), or dot (`.`)
//! - No colon (`:`) — reserved as the qualified-name separator
//! - No backslash (`\`) — reserved as the namespace escape character
//! - No whitespace, control characters, or path separators
//!
//! Operators with legacy names containing other characters rename
//! at the deployment boundary (a single mapping table).

use std::sync::Arc;
use std::time::Duration;

use reqwest::header::HeaderMap;
use secrecy::SecretString;

use crate::elicitation::ElicitationProvider;
use crate::error::{McpError, McpResult};
use crate::roots::RootsProvider;
use crate::sampling::SamplingProvider;

/// Hook that decorates outgoing MCP requests with extra HTTP headers.
///
/// Typically used to inject W3C trace-context (`traceparent`,
/// `tracestate`, `baggage`) so distributed traces span the SDK
/// caller and the MCP server it dispatches to.
///
/// `Send + Sync + 'static` because the hook is shared across every
/// HTTP request the client issues. Lives in [`McpServerConfig`] so
/// operators bind one propagator per server (some servers honour
/// W3C, others use B3, others none at all).
///
/// `entelix-mcp` ships no concrete propagator — the implementation
/// lives in `entelix-otel` so the MCP crate stays free of an
/// `opentelemetry` dependency. Operators wire the hook at config
/// time via [`McpServerConfig::with_request_decorator`].
pub type RequestDecorator = Arc<dyn Fn(&mut HeaderMap) + Send + Sync + 'static>;

/// Default request timeout for MCP HTTP calls. MCP servers are
/// usually local-network or sidecar deployments; 30s is generous
/// for tools that include LLM calls themselves.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Default idle TTL for cached MCP clients. Operators schedule
/// [`crate::McpManager::prune_idle`] (typically on a `tokio::time::interval`)
/// to evict clients whose last use exceeded this threshold.
pub const DEFAULT_IDLE_TTL: Duration = Duration::from_mins(5);

/// Default cap on a single SSE frame's accumulated byte length — 1 MiB.
///
/// JSON-RPC messages over MCP are typically kilobytes; a frame that
/// grows past 1 MiB without delivering a `\n\n` terminator
/// indicates a malicious or malfunctioning peer rather than
/// legitimate traffic. Override per-server with
/// [`McpServerConfig::with_max_frame_bytes`] when an operator
/// genuinely needs larger frames.
pub const DEFAULT_MAX_FRAME_BYTES: usize = 1 << 20;

/// Default cap on the number of in-flight server-initiated dispatches per client — 32.
///
/// Covers `roots/list`, `elicitation/create`,
/// `sampling/createMessage`, and future channels. New requests
/// beyond the cap are dropped — not queued — so a hostile or
/// noisy server cannot pin arbitrary executor memory by flooding
/// the listener. Override per-server with
/// [`McpServerConfig::with_listener_concurrency`].
pub const DEFAULT_LISTENER_CONCURRENCY: usize = 32;

/// Connection details for one MCP server.
#[derive(Clone)]
pub struct McpServerConfig {
    /// Stable identifier used for routing (`McpManager::call_tool`).
    pub(crate) name: String,
    /// HTTPS endpoint exposing the MCP JSON-RPC surface.
    pub(crate) url: String,
    /// Optional bearer token for `Authorization: Bearer ...`.
    pub(crate) bearer: Option<SecretString>,
    /// Per-call request timeout.
    pub(crate) timeout: Duration,
    /// Maximum idle window before this server's cached client
    /// becomes eligible for eviction by `McpManager::prune_idle`.
    pub(crate) idle_ttl: Duration,
    /// Optional decorator that mutates the outbound `HeaderMap`
    /// just before each request leaves the wire — used to inject
    /// W3C trace-context for distributed tracing across the
    /// SDK / MCP-server boundary.
    pub(crate) request_decorator: Option<RequestDecorator>,
    /// Optional source-of-truth for roots advertised to the server
    /// when it issues `roots/list`. Presence of a provider also
    /// flips the `roots` capability on during initialize, gating
    /// server-initiated traffic on operator opt-in.
    pub(crate) roots_provider: Option<Arc<dyn RootsProvider>>,
    /// Optional source-of-truth for elicitation answers when the
    /// server issues `elicitation/create`. Presence of a provider
    /// flips the `elicitation` capability on during initialize.
    pub(crate) elicitation_provider: Option<Arc<dyn ElicitationProvider>>,
    /// Optional source-of-truth for sampling completions when
    /// the server issues `sampling/createMessage`. Presence of
    /// a provider flips the `sampling` capability on during
    /// initialize.
    pub(crate) sampling_provider: Option<Arc<dyn SamplingProvider>>,
    /// Maximum byte length of one SSE frame before the listener
    /// closes the connection. See [`DEFAULT_MAX_FRAME_BYTES`].
    pub(crate) max_frame_bytes: usize,
    /// Maximum number of concurrent in-flight server-initiated
    /// request dispatches. See [`DEFAULT_LISTENER_CONCURRENCY`].
    pub(crate) listener_concurrency: usize,
}

impl std::fmt::Debug for McpServerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpServerConfig")
            .field("name", &self.name)
            .field("url", &self.url)
            .field("bearer", &self.bearer.as_ref().map(|_| "<redacted>"))
            .field("timeout", &self.timeout)
            .field("idle_ttl", &self.idle_ttl)
            .field(
                "request_decorator",
                &self.request_decorator.as_ref().map(|_| "<fn>"),
            )
            .field("roots_provider", &self.roots_provider)
            .field("elicitation_provider", &self.elicitation_provider)
            .field("sampling_provider", &self.sampling_provider)
            .field("max_frame_bytes", &self.max_frame_bytes)
            .field("listener_concurrency", &self.listener_concurrency)
            .finish()
    }
}

impl McpServerConfig {
    /// HTTP server with no auth.
    ///
    /// Returns [`McpError::Config`] when `name` violates the format
    /// rules in the module-level docstring (empty, contains `:`,
    /// `\`, whitespace, or non-ASCII-alphanumeric characters).
    pub fn http(name: impl Into<String>, url: impl Into<String>) -> McpResult<Self> {
        let name = name.into();
        validate_server_name(&name)?;
        Ok(Self {
            name,
            url: url.into(),
            bearer: None,
            timeout: DEFAULT_TIMEOUT,
            idle_ttl: DEFAULT_IDLE_TTL,
            request_decorator: None,
            roots_provider: None,
            elicitation_provider: None,
            sampling_provider: None,
            max_frame_bytes: DEFAULT_MAX_FRAME_BYTES,
            listener_concurrency: DEFAULT_LISTENER_CONCURRENCY,
        })
    }

    /// Attach a bearer token (carried in the `Authorization` header).
    #[must_use]
    pub fn with_bearer(mut self, token: SecretString) -> Self {
        self.bearer = Some(token);
        self
    }

    /// Override the request timeout.
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Override the idle TTL — how long an unused cached client
    /// can sit before [`crate::McpManager::prune_idle`] evicts it.
    #[must_use]
    pub const fn with_idle_ttl(mut self, ttl: Duration) -> Self {
        self.idle_ttl = ttl;
        self
    }

    /// Attach a [`RequestDecorator`] — invoked once per outgoing
    /// HTTP request to inject extra headers. The canonical use case
    /// is W3C trace-context propagation; `entelix-otel` ships a
    /// constructor returning a decorator that reads the active
    /// [`tracing`] span's `OTel` context and injects `traceparent` /
    /// `tracestate` / `baggage`.
    ///
    /// Multiple calls overwrite the previous decorator — operators
    /// that want a chain compose them in their own `Fn` body.
    #[must_use]
    pub fn with_request_decorator(mut self, decorator: RequestDecorator) -> Self {
        self.request_decorator = Some(decorator);
        self
    }

    /// Borrow the server name (used as a routing key).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Borrow the endpoint URL.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Effective idle TTL.
    pub const fn idle_ttl(&self) -> Duration {
        self.idle_ttl
    }

    /// Borrow the configured request decorator, if any. Used by
    /// [`crate::HttpMcpClient`] to invoke per-request header
    /// injection.
    pub(crate) fn request_decorator(&self) -> Option<&RequestDecorator> {
        self.request_decorator.as_ref()
    }

    /// Attach a [`RootsProvider`] — the source of truth for the
    /// roots this client exposes when the server issues
    /// `roots/list`. Presence of a provider also opts the client
    /// into the `roots` capability advertisement at initialize
    /// time, so servers gate their `roots/list` traffic on the
    /// flag rather than blindly probing.
    ///
    /// Replacing the provider after the client is built has no
    /// effect — the wired-in [`Arc`] is captured at config time.
    /// Operators that need swap-in-place behaviour wrap the
    /// concrete provider behind interior mutability of their own.
    #[must_use]
    pub fn with_roots_provider(mut self, provider: Arc<dyn RootsProvider>) -> Self {
        self.roots_provider = Some(provider);
        self
    }

    /// Borrow the configured roots provider, if any. Used by the
    /// streamable-http listener and the `notifications/roots/list_changed`
    /// dispatcher.
    pub(crate) fn roots_provider(&self) -> Option<&Arc<dyn RootsProvider>> {
        self.roots_provider.as_ref()
    }

    /// Attach an [`ElicitationProvider`] — the source of truth for
    /// `elicitation/create` answers. Presence of a provider also
    /// opts the client into the `elicitation` capability
    /// advertisement at initialize time, so servers gate their
    /// elicitation traffic on the flag rather than blindly
    /// probing.
    ///
    /// Replacing the provider after the client is built has no
    /// effect — the wired-in [`Arc`] is captured at config time.
    /// Operators that need swap-in-place behaviour wrap the
    /// concrete provider behind interior mutability of their own.
    #[must_use]
    pub fn with_elicitation_provider(mut self, provider: Arc<dyn ElicitationProvider>) -> Self {
        self.elicitation_provider = Some(provider);
        self
    }

    /// Borrow the configured elicitation provider, if any. Used
    /// by the streamable-http listener's `elicitation/create`
    /// dispatcher.
    pub(crate) fn elicitation_provider(&self) -> Option<&Arc<dyn ElicitationProvider>> {
        self.elicitation_provider.as_ref()
    }

    /// Attach a [`SamplingProvider`] — the source of truth for
    /// `sampling/createMessage` completions. Presence of a
    /// provider also opts the client into the `sampling`
    /// capability advertisement at initialize time.
    ///
    /// Replacing the provider after the client is built has no
    /// effect — the wired-in [`Arc`] is captured at config time.
    #[must_use]
    pub fn with_sampling_provider(mut self, provider: Arc<dyn SamplingProvider>) -> Self {
        self.sampling_provider = Some(provider);
        self
    }

    /// Borrow the configured sampling provider, if any. Used
    /// by the streamable-http listener's `sampling/createMessage`
    /// dispatcher.
    pub(crate) fn sampling_provider(&self) -> Option<&Arc<dyn SamplingProvider>> {
        self.sampling_provider.as_ref()
    }

    /// Cap on a single SSE frame's accumulated byte length. The
    /// listener closes the connection if any frame grows past this
    /// threshold without a `\n\n` terminator — the spec puts no
    /// upper bound on frame size, so an unbounded buffer is an
    /// open OOM vector against a hostile or malfunctioning server.
    /// Default: [`DEFAULT_MAX_FRAME_BYTES`] (1 MiB).
    #[must_use]
    pub const fn with_max_frame_bytes(mut self, n: usize) -> Self {
        self.max_frame_bytes = n;
        self
    }

    /// Effective max frame byte length.
    pub const fn max_frame_bytes(&self) -> usize {
        self.max_frame_bytes
    }

    /// Cap on the number of concurrent in-flight server-initiated
    /// dispatches per client. Excess requests are dropped with a
    /// `tracing::warn!` so a flooding server cannot pin executor
    /// memory; the server is expected to retry on its own cadence.
    /// Default: [`DEFAULT_LISTENER_CONCURRENCY`] (32).
    #[must_use]
    pub const fn with_listener_concurrency(mut self, n: usize) -> Self {
        self.listener_concurrency = n;
        self
    }

    /// Effective server-initiated dispatch concurrency cap.
    pub const fn listener_concurrency(&self) -> usize {
        self.listener_concurrency
    }
}

/// Validate an MCP server name against the documented format.
///
/// Public so MCP-aware infrastructure (admin tooling, manifest
/// loaders) can pre-flight names without constructing a full config.
pub fn validate_server_name(name: &str) -> McpResult<()> {
    if name.is_empty() {
        return Err(McpError::Config(
            "MCP server name must be non-empty".to_owned(),
        ));
    }
    for ch in name.chars() {
        let allowed = ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.');
        if !allowed {
            return Err(McpError::Config(format!(
                "MCP server name {name:?} contains disallowed character {ch:?}; \
                 allowed: ASCII alphanumeric, '-', '_', '.'"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn http_accepts_alphanumeric_and_dashes() {
        assert!(McpServerConfig::http("filesystem-2", "http://x").is_ok());
        assert!(McpServerConfig::http("ABC_123", "http://x").is_ok());
        assert!(McpServerConfig::http("svc.local", "http://x").is_ok());
    }

    #[test]
    fn http_rejects_empty_name() {
        let err = McpServerConfig::http("", "http://x").unwrap_err();
        assert!(matches!(err, McpError::Config(_)));
    }

    #[test]
    fn http_rejects_colon_in_name() {
        // The qualified-name format `mcp:{server}:{tool}` reserves
        // `:` as the separator — a server name containing one
        // would create ambiguous tool keys.
        let err = McpServerConfig::http("svc:local", "http://x").unwrap_err();
        assert!(matches!(err, McpError::Config(_)));
    }

    #[test]
    fn http_rejects_backslash() {
        let err = McpServerConfig::http("svc\\local", "http://x").unwrap_err();
        assert!(matches!(err, McpError::Config(_)));
    }

    #[test]
    fn http_rejects_whitespace_and_path_separators() {
        assert!(McpServerConfig::http("svc local", "http://x").is_err());
        assert!(McpServerConfig::http("svc/local", "http://x").is_err());
        assert!(McpServerConfig::http("svc\nlocal", "http://x").is_err());
    }
}
