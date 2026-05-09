//! Configuration values for [`super::ClaudeCodeOAuthProvider`].

use std::time::Duration;

/// Canonical Claude.ai OAuth token endpoint.
pub const DEFAULT_TOKEN_URL: &str = "https://console.anthropic.com/v1/oauth/token";

/// Anthropic-beta capability gate the `claude` CLI sets on every
/// request.
///
/// Operators using [`super::ClaudeCodeOAuthProvider`] must also pass
/// this value through
/// [`entelix_core::ir::AnthropicExt::with_betas`] so the codec emits
/// the matching header — the credential surface and the codec
/// surface stay independent (single responsibility).
///
/// **Source of truth**: Anthropic's beta-header registry under
/// <https://platform.claude.com/docs/en/release-notes/overview>.
/// Anthropic versions beta capabilities by date stamp; if Claude
/// Code rolls a successor (e.g. `claude-code-2026XXXX`) bump this
/// constant in lockstep — the OAuth refresh flow stays unchanged.
pub const CLAUDE_CODE_BETA: &str = "claude-code-20250219";

/// Default refresh-call HTTP timeout.
pub const DEFAULT_REFRESH_TIMEOUT: Duration = Duration::from_secs(30);

/// Configuration for [`super::ClaudeCodeOAuthProvider`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ClaudeCodeOAuthConfig {
    /// OAuth2 token endpoint URL. Defaults to
    /// [`DEFAULT_TOKEN_URL`].
    pub token_url: String,
    /// Optional OAuth client id sent with the `refresh_token`
    /// grant. The `claude` CLI omits this; supply a value only
    /// if your refresh-token policy requires one.
    pub client_id: Option<String>,
    /// HTTP timeout applied to refresh requests.
    pub refresh_timeout: Duration,
}

impl Default for ClaudeCodeOAuthConfig {
    fn default() -> Self {
        Self {
            token_url: DEFAULT_TOKEN_URL.to_owned(),
            client_id: None,
            refresh_timeout: DEFAULT_REFRESH_TIMEOUT,
        }
    }
}

impl ClaudeCodeOAuthConfig {
    /// Construct an empty config (`Self::default`).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the token endpoint URL — useful for staging
    /// environments or test mocks.
    #[must_use]
    pub fn with_token_url(mut self, url: impl Into<String>) -> Self {
        self.token_url = url.into();
        self
    }

    /// Set the OAuth client id sent with the refresh grant.
    #[must_use]
    pub fn with_client_id(mut self, client_id: impl Into<String>) -> Self {
        self.client_id = Some(client_id.into());
        self
    }

    /// Override the refresh-call HTTP timeout.
    #[must_use]
    pub const fn with_refresh_timeout(mut self, timeout: Duration) -> Self {
        self.refresh_timeout = timeout;
        self
    }
}
