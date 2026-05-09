//! `OAuthCredential` — the on-disk shape for Claude Code's OAuth
//! state, plus the refresh-flow data carrier.
//!
//! Wire-format reference: the file the `claude` CLI writes at
//! `~/.claude/.credentials.json` (or platform equivalent). Format
//! adapted from the public `claude` CLI.

use chrono::{DateTime, TimeZone, Utc};
use secrecy::SecretString;
use serde::{Deserialize, Serialize};

/// A refreshable OAuth credential read from / written to the
/// `claude` CLI's credential file.
///
/// Field naming mirrors the on-disk JSON exactly so deserialization
/// stays a one-line `serde_json::from_slice`. `#[non_exhaustive]`
/// closes off direct struct-literal construction so future
/// server-returned fields ship as additive minor releases — use
/// [`OAuthCredential::new`] plus the `with_*` chain.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub struct OAuthCredential {
    /// Bearer token forwarded to Anthropic in the `Authorization`
    /// header.
    #[serde(rename = "accessToken")]
    pub access_token: String,
    /// Long-lived refresh token. `None` means the credential cannot
    /// renew itself once the access token expires.
    #[serde(
        rename = "refreshToken",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub refresh_token: Option<String>,
    /// Unix-millis epoch at which the access token expires.
    #[serde(rename = "expiresAt")]
    pub expires_at_ms: i64,
    /// `pro` / `team` / etc. — informational, not load-bearing for
    /// transport. Preserved across refresh so storage round-trips
    /// keep the original metadata.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "subscriptionType"
    )]
    pub subscription_type: Option<String>,
    /// OAuth scopes granted to this token. Preserved across refresh.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scopes: Vec<String>,
}

impl OAuthCredential {
    /// Construct a credential from the two mandatory fields. Use
    /// the `with_*` chain for the optional ones.
    #[must_use]
    pub fn new(access_token: impl Into<String>, expires_at_ms: i64) -> Self {
        Self {
            access_token: access_token.into(),
            refresh_token: None,
            expires_at_ms,
            subscription_type: None,
            scopes: Vec::new(),
        }
    }

    /// Set the long-lived refresh token.
    #[must_use]
    pub fn with_refresh_token(mut self, token: impl Into<String>) -> Self {
        self.refresh_token = Some(token.into());
        self
    }

    /// Set the subscription tier (`pro` / `team` / …).
    #[must_use]
    pub fn with_subscription_type(mut self, tier: impl Into<String>) -> Self {
        self.subscription_type = Some(tier.into());
        self
    }

    /// Replace the granted-scopes list.
    #[must_use]
    pub fn with_scopes<I, S>(mut self, scopes: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.scopes = scopes.into_iter().map(Into::into).collect();
        self
    }

    /// Wall-clock instant the access token expires, or `None` when
    /// `expires_at_ms` does not represent a valid Unix-millis epoch
    /// (storage corruption / server bug). Surfaces the loss
    /// explicitly rather than coercing to "now" — the credential
    /// is then treated as already expired by [`Self::needs_refresh`].
    #[must_use]
    pub fn expires_at(&self) -> Option<DateTime<Utc>> {
        Utc.timestamp_millis_opt(self.expires_at_ms).single()
    }

    /// True when the access token is past its expiry — `resolve`
    /// must trigger a refresh before handing the token to a
    /// transport. An unrepresentable `expires_at_ms` (storage
    /// corruption, server bug) reads as already-expired so the
    /// provider refreshes rather than handing out a stale token.
    /// Includes a 60-second skew window so a token about to expire
    /// mid-flight refreshes proactively.
    #[must_use]
    pub fn needs_refresh(&self) -> bool {
        let Some(expires_at) = self.expires_at() else {
            return true;
        };
        Utc::now() + chrono::Duration::seconds(60) >= expires_at
    }

    /// Wrap the access token in a [`SecretString`] formatted as
    /// `Bearer <token>` for transport use. Allocates a fresh secret
    /// per call so callers can hand it straight to
    /// [`entelix_core::auth::Credentials::header_value`].
    #[must_use]
    pub fn to_bearer_secret(&self) -> SecretString {
        SecretString::from(format!("Bearer {}", self.access_token))
    }
}

/// On-disk envelope for the credential file.
///
/// The file currently houses one OAuth credential under
/// `claudeAiOauth`. Future schema additions land here additively —
/// `#[non_exhaustive]` keeps external construction on
/// [`CredentialFile::with_oauth`] / [`CredentialFile::empty`] so
/// new fields are non-breaking.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CredentialFile {
    /// The Claude.ai OAuth credential, if present.
    #[serde(
        default,
        rename = "claudeAiOauth",
        skip_serializing_if = "Option::is_none"
    )]
    pub claude_ai_oauth: Option<OAuthCredential>,
}

impl CredentialFile {
    /// Empty envelope.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Envelope wrapping one OAuth credential.
    #[must_use]
    pub const fn with_oauth(credential: OAuthCredential) -> Self {
        Self {
            claude_ai_oauth: Some(credential),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_minimal_credential_file() {
        let json = r#"{
            "claudeAiOauth": {
                "accessToken": "sk-ant-oat01-x",
                "refreshToken": "sk-ant-ort01-y",
                "expiresAt": 9999999999000,
                "subscriptionType": "pro",
                "scopes": ["user:inference"]
            }
        }"#;
        let file: CredentialFile = serde_json::from_str(json).unwrap();
        let oauth = file.claude_ai_oauth.unwrap();
        assert_eq!(oauth.access_token, "sk-ant-oat01-x");
        assert_eq!(oauth.refresh_token.as_deref(), Some("sk-ant-ort01-y"));
        assert_eq!(oauth.subscription_type.as_deref(), Some("pro"));
        assert!(!oauth.needs_refresh());
    }

    #[test]
    fn needs_refresh_when_within_skew_window() {
        let near_expiry = OAuthCredential::new(
            "tok",
            (Utc::now() - chrono::Duration::seconds(1)).timestamp_millis(),
        )
        .with_refresh_token("ref");
        assert!(near_expiry.needs_refresh());
    }

    #[test]
    fn empty_envelope_has_no_oauth() {
        let file: CredentialFile = serde_json::from_str("{}").unwrap();
        assert!(file.claude_ai_oauth.is_none());
    }

    #[test]
    fn expires_at_returns_none_for_unrepresentable_millis() {
        // i64::MAX as Unix-millis is past the chrono representable
        // range — the helper surfaces this as `None` instead of
        // silently coercing to "now" (which would fool
        // `needs_refresh` into thinking the credential is fresh).
        let cred = OAuthCredential::new("tok", i64::MAX);
        assert!(cred.expires_at().is_none());
    }

    #[test]
    fn unrepresentable_expires_treated_as_already_expired() {
        // Even when the timestamp can't be rendered, the credential
        // must read as expired so the provider triggers a refresh
        // rather than handing out a stale token. Both i64 extremes
        // round-trip through Option<DateTime>::None and surface as
        // needs_refresh — the security-relevant default.
        let past = OAuthCredential::new("tok", i64::MIN);
        assert!(past.needs_refresh());
        let future = OAuthCredential::new("tok", i64::MAX);
        assert!(future.needs_refresh());
    }

    #[test]
    fn builder_chain_populates_optional_fields() {
        let cred = OAuthCredential::new("acc", 9_999_999_999_000)
            .with_refresh_token("ref")
            .with_subscription_type("team")
            .with_scopes(["user:inference", "user:profile"]);
        assert_eq!(cred.access_token, "acc");
        assert_eq!(cred.refresh_token.as_deref(), Some("ref"));
        assert_eq!(cred.subscription_type.as_deref(), Some("team"));
        assert_eq!(cred.scopes, vec!["user:inference", "user:profile"]);
    }
}
