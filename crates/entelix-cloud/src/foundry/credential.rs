//! Azure credential adapter — wraps an
//! `azure_core::credentials::TokenCredential` into a
//! [`crate::refresh::TokenRefresher`].
//!
//! Production callers typically pass
//! `azure_identity::DeveloperToolsCredential::new(None)?` (which
//! itself returns an `Arc<DeveloperToolsCredential>` and unsizes
//! into `Arc<dyn TokenCredential>`). Tests inject a deterministic
//! mock `TokenRefresher` directly without touching this adapter.

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use azure_core::credentials::TokenCredential;
use secrecy::SecretString;

use crate::CloudError;
use crate::refresh::{TokenRefresher, TokenSnapshot};

/// Standard scope for Azure OpenAI / Foundry data plane access.
pub const FOUNDRY_SCOPE: &str = "https://cognitiveservices.azure.com/.default";

/// Azure-credential-backed refresher. Generic over any
/// `Arc<dyn TokenCredential>` — the caller picks the chain.
pub struct FoundryCredentialProvider {
    inner: Arc<dyn TokenCredential>,
    scopes: Vec<String>,
}

impl FoundryCredentialProvider {
    /// Wrap an externally-built `TokenCredential` (e.g.
    /// `azure_identity::DeveloperToolsCredential::new(None)?` —
    /// which already returns an `Arc<dyn TokenCredential>` once
    /// unsized) plus the OAuth scopes the call needs.
    pub fn new(cred: Arc<dyn TokenCredential>, scopes: &[&str]) -> Self {
        Self {
            inner: cred,
            scopes: scopes.iter().map(|s| (*s).to_owned()).collect(),
        }
    }
}

#[async_trait]
impl TokenRefresher<SecretString> for FoundryCredentialProvider {
    async fn refresh(&self) -> Result<TokenSnapshot<SecretString>, CloudError> {
        let scopes_ref: Vec<&str> = self.scopes.iter().map(String::as_str).collect();
        let token =
            self.inner
                .get_token(&scopes_ref, None)
                .await
                .map_err(|e| CloudError::Credential {
                    message: format!("get_token: {e}"),
                    source: Some(Box::new(e)),
                })?;
        let value = SecretString::from(token.token.secret().to_owned());
        let expires_at = offset_to_instant(token.expires_on);
        Ok(TokenSnapshot { value, expires_at })
    }
}

fn offset_to_instant(at: time::OffsetDateTime) -> Instant {
    let now_t = time::OffsetDateTime::now_utc();
    let now_inst = Instant::now();
    let delta_ms = (at - now_t).whole_milliseconds();
    if delta_ms <= 0 {
        return now_inst;
    }
    now_inst + Duration::from_millis(u64::try_from(delta_ms).unwrap_or(u64::MAX)) // silent-fallback-ok: defense-in-depth, clamp duration overflow before Instant arithmetic
}
