//! GCP credential adapter — wraps `gcp_auth` into a
//! [`crate::refresh::TokenRefresher`] that the transport's
//! [`crate::refresh::CachedTokenProvider`] consumes.

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use gcp_auth::TokenProvider;
use secrecy::SecretString;

use crate::CloudError;
use crate::refresh::{TokenRefresher, TokenSnapshot};

/// Standard GCP IAM scope for Vertex AI generative models.
pub const VERTEX_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";

/// `gcp_auth`-backed token refresher.
///
/// Resolves credentials via the Application Default Credentials chain
/// (`GOOGLE_APPLICATION_CREDENTIALS` → metadata server → gcloud
/// auth) — the same chain the gcloud CLI uses, so workstation and
/// production are configured identically.
pub struct VertexCredentialProvider {
    inner: Arc<dyn TokenProvider>,
    scopes: Vec<String>,
}

impl VertexCredentialProvider {
    /// Build with the default Vertex scope.
    pub async fn default_chain() -> Result<Self, CloudError> {
        Self::with_scopes(&[VERTEX_SCOPE]).await
    }

    /// Build with custom OAuth scopes (e.g. limited service-specific
    /// scopes for stricter IAM).
    pub async fn with_scopes(scopes: &[&str]) -> Result<Self, CloudError> {
        let provider = gcp_auth::provider()
            .await
            .map_err(|e| CloudError::Credential {
                message: format!("gcp_auth provider: {e}"),
                source: Some(Box::new(e)),
            })?;
        Ok(Self {
            inner: provider,
            scopes: scopes.iter().map(|s| (*s).to_owned()).collect(),
        })
    }

    /// Wrap an externally-built provider — used in tests.
    pub fn from_provider(provider: Arc<dyn TokenProvider>, scopes: Vec<String>) -> Self {
        Self {
            inner: provider,
            scopes,
        }
    }
}

#[async_trait]
impl TokenRefresher<SecretString> for VertexCredentialProvider {
    async fn refresh(&self) -> Result<TokenSnapshot<SecretString>, CloudError> {
        let scopes_ref: Vec<&str> = self.scopes.iter().map(String::as_str).collect();
        let token = self
            .inner
            .token(&scopes_ref)
            .await
            .map_err(|e| CloudError::Credential {
                message: format!("gcp_auth token: {e}"),
                source: Some(Box::new(e)),
            })?;
        let value = SecretString::from(token.as_str().to_owned());
        let expires_at = gcp_to_instant(token.expires_at());
        Ok(TokenSnapshot { value, expires_at })
    }
}

fn gcp_to_instant(at: chrono::DateTime<chrono::Utc>) -> Instant {
    let now_chrono = chrono::Utc::now();
    let now_inst = Instant::now();
    let delta = at.signed_duration_since(now_chrono);
    if delta.num_milliseconds() <= 0 {
        return now_inst;
    }
    now_inst + Duration::from_millis(u64::try_from(delta.num_milliseconds()).unwrap_or(u64::MAX)) // silent-fallback-ok: defense-in-depth, clamp duration overflow before Instant arithmetic
}
