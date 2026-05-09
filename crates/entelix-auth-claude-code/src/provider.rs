//! [`ClaudeCodeOAuthProvider`] — [`CredentialProvider`] impl that
//! resolves the access token the `claude` CLI manages, refreshing
//! it through the standard OAuth2 `refresh_token` grant when needed.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use entelix_core::auth::{CredentialProvider, Credentials};
use entelix_core::error::Result;

use crate::config::ClaudeCodeOAuthConfig;
use crate::credential::{CredentialFile, OAuthCredential};
use crate::error::ClaudeCodeAuthError;
use crate::refresh::refresh_access_token;
use crate::store::CredentialStore;

/// Resolve credentials from a [`CredentialStore`] backend,
/// refreshing the access token via the Anthropic console token
/// endpoint when expiry is imminent.
///
/// Concurrent refresh attempts are serialised through an internal
/// mutex — refresh tokens may rotate on every grant, so two
/// in-flight refresh calls would race each other into rejection.
pub struct ClaudeCodeOAuthProvider {
    store: Arc<dyn CredentialStore>,
    http: reqwest::Client,
    refresh_guard: Mutex<()>,
    config: ClaudeCodeOAuthConfig,
}

impl std::fmt::Debug for ClaudeCodeOAuthProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClaudeCodeOAuthProvider")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl ClaudeCodeOAuthProvider {
    /// Build a provider over the supplied store backend with the
    /// canonical Anthropic console token endpoint.
    pub fn new(store: impl CredentialStore) -> Self {
        Self::with_config(store, ClaudeCodeOAuthConfig::default())
    }

    /// Build a provider with an explicit config (custom token URL,
    /// refresh timeout, …).
    pub fn with_config(store: impl CredentialStore, config: ClaudeCodeOAuthConfig) -> Self {
        let http = reqwest::Client::builder()
            .timeout(config.refresh_timeout)
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            store: Arc::new(store),
            http,
            refresh_guard: Mutex::new(()),
            config,
        }
    }

    async fn load_oauth(&self) -> Result<OAuthCredential> {
        let envelope =
            self.store
                .load()
                .await?
                .ok_or_else(|| ClaudeCodeAuthError::CredentialsMissing {
                    path: "<store>".into(),
                })?;
        envelope
            .claude_ai_oauth
            .ok_or_else(|| ClaudeCodeAuthError::OAuthSectionMissing {
                path: "<store>".into(),
            })
            .map_err(Into::into)
    }

    async fn refresh(&self, prior: OAuthCredential) -> Result<OAuthCredential> {
        // Serialise refresh attempts — concurrent refresh_token
        // usage races into the server-side rotation and one of the
        // two callers loses.
        let _guard = self.refresh_guard.lock().await;

        // Re-load under the lock: another caller may have refreshed
        // while we were waiting. Fall back to the prior in-memory
        // credential when the store has no record (tests with
        // ephemeral backends).
        let current = self
            .store
            .load()
            .await?
            .and_then(|e| e.claude_ai_oauth)
            .unwrap_or(prior);
        if !current.needs_refresh() {
            return Ok(current);
        }

        let refresh_token = current
            .refresh_token
            .as_deref()
            .ok_or(ClaudeCodeAuthError::RefreshTokenMissing)?;

        let mut refreshed = refresh_access_token(
            &self.http,
            &self.config.token_url,
            refresh_token,
            self.config.client_id.as_deref(),
        )
        .await?;

        // The token endpoint only returns token fields; carry the
        // operator-visible metadata through unchanged so store
        // round-trips preserve it. Rotated refresh tokens come back
        // populated; missing means "reuse prior".
        if refreshed.subscription_type.is_none() {
            refreshed
                .subscription_type
                .clone_from(&current.subscription_type);
        }
        if refreshed.scopes.is_empty() {
            refreshed.scopes.clone_from(&current.scopes);
        }
        if refreshed.refresh_token.is_none() {
            refreshed.refresh_token.clone_from(&current.refresh_token);
        }

        self.store
            .save(&CredentialFile::with_oauth(refreshed.clone()))
            .await?;
        Ok(refreshed)
    }
}

#[async_trait]
impl CredentialProvider for ClaudeCodeOAuthProvider {
    async fn resolve(&self) -> Result<Credentials> {
        let oauth = self.load_oauth().await?;
        let active = if oauth.needs_refresh() {
            self.refresh(oauth).await?
        } else {
            oauth
        };
        Ok(Credentials {
            header_name: http::header::AUTHORIZATION,
            header_value: active.to_bearer_secret(),
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::store::CredentialStore;
    use chrono::Utc;
    use secrecy::ExposeSecret;
    use std::sync::Mutex as StdMutex;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[derive(Clone, Default)]
    struct MemoryCredentialStore {
        inner: Arc<StdMutex<Option<CredentialFile>>>,
    }

    impl MemoryCredentialStore {
        fn seeded(file: CredentialFile) -> Self {
            Self {
                inner: Arc::new(StdMutex::new(Some(file))),
            }
        }
    }

    #[async_trait]
    impl CredentialStore for MemoryCredentialStore {
        async fn load(&self) -> crate::error::ClaudeCodeAuthResult<Option<CredentialFile>> {
            Ok(self.inner.lock().unwrap().clone())
        }
        async fn save(&self, file: &CredentialFile) -> crate::error::ClaudeCodeAuthResult<()> {
            *self.inner.lock().unwrap() = Some(file.clone());
            Ok(())
        }
    }

    fn fresh_oauth() -> OAuthCredential {
        OAuthCredential::new(
            "fresh-access",
            (Utc::now() + chrono::Duration::hours(1)).timestamp_millis(),
        )
        .with_refresh_token("ref")
        .with_subscription_type("pro")
        .with_scopes(["user:inference"])
    }

    fn expired_oauth() -> OAuthCredential {
        OAuthCredential::new(
            "stale-access",
            (Utc::now() - chrono::Duration::seconds(5)).timestamp_millis(),
        )
        .with_refresh_token("ref")
        .with_subscription_type("pro")
        .with_scopes(["user:inference"])
    }

    #[tokio::test]
    async fn resolve_returns_bearer_when_token_fresh() {
        let store = MemoryCredentialStore::seeded(CredentialFile::with_oauth(fresh_oauth()));
        let provider = ClaudeCodeOAuthProvider::new(store);
        let creds = provider.resolve().await.unwrap();
        assert_eq!(creds.header_name, http::header::AUTHORIZATION);
        assert_eq!(creds.header_value.expose_secret(), "Bearer fresh-access");
    }

    #[tokio::test]
    async fn resolve_refreshes_when_token_expired() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/oauth/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "access_token": "renewed-access",
                "refresh_token": "renewed-refresh",
                "expires_in": 3600
            })))
            .mount(&server)
            .await;

        let store = MemoryCredentialStore::seeded(CredentialFile::with_oauth(expired_oauth()));
        let provider = ClaudeCodeOAuthProvider::with_config(
            store.clone(),
            ClaudeCodeOAuthConfig::new().with_token_url(format!("{}/oauth/token", server.uri())),
        );
        let creds = provider.resolve().await.unwrap();
        assert_eq!(creds.header_value.expose_secret(), "Bearer renewed-access");

        // Store round-trip preserves operator metadata + persists
        // the rotated refresh token.
        let saved = store
            .load()
            .await
            .unwrap()
            .unwrap()
            .claude_ai_oauth
            .unwrap();
        assert_eq!(saved.access_token, "renewed-access");
        assert_eq!(saved.refresh_token.as_deref(), Some("renewed-refresh"));
        assert_eq!(saved.subscription_type.as_deref(), Some("pro"));
        assert!(saved.scopes.contains(&"user:inference".to_owned()));
    }

    #[tokio::test]
    async fn resolve_errors_when_store_empty() {
        let store = MemoryCredentialStore::default();
        let provider = ClaudeCodeOAuthProvider::new(store);
        let err = provider.resolve().await.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("not found"), "got: {msg}");
    }

    #[tokio::test]
    async fn resolve_errors_when_refresh_token_absent_and_expired() {
        let stale = OAuthCredential::new(
            "stale-access",
            (Utc::now() - chrono::Duration::seconds(5)).timestamp_millis(),
        )
        .with_subscription_type("pro");
        let store = MemoryCredentialStore::seeded(CredentialFile::with_oauth(stale));
        let provider = ClaudeCodeOAuthProvider::new(store);
        let err = provider.resolve().await.unwrap_err();
        assert!(err.to_string().contains("refresh token absent"));
    }
}
