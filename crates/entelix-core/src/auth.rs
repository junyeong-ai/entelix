//! Credential resolution for transports.
//!
//! Per invariant 10, credentials live exclusively in this module and are
//! plumbed through `Transport`. `ExecutionContext` does NOT embed a
//! [`CredentialProvider`], so `Tool::execute` never sees a token.
//!
//! Two ready-made impls cover the two header conventions used by
//! every shipped provider:
//! - [`ApiKeyProvider`] — emits a custom header (e.g. Anthropic
//!   `x-api-key: <key>`).
//! - [`BearerProvider`] — emits `authorization: Bearer <token>`.
//!
//! Failures surface as [`Error::Auth`] carrying a typed [`AuthError`]
//! so credential-chain bugs (missing keys, expired tokens, refused
//! refresh) are distinguishable from generic provider HTTP failures
//! at the application layer.

use async_trait::async_trait;
use secrecy::{ExposeSecret, SecretString};
use thiserror::Error;

use crate::error::{Error, Result};

/// Typed credential failure. Public APIs raise [`Error::Auth`] which
/// wraps this enum; downstream layers (retry policies, circuit
/// breakers, dashboards) can match on the variant rather than on a
/// stringly-typed `Error::Provider` blob.
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum AuthError {
    /// No credential is configured for the requested scope. Most
    /// often a deployment-time misconfiguration: the operator forgot
    /// to wire a [`CredentialProvider`] into the transport, or a
    /// chained provider exhausted every source without finding one.
    #[error("auth: no credential available{}", source_hint.as_ref().map(|s| format!(" (source: {s})")).unwrap_or_default())]
    Missing {
        /// Human-readable hint about which source was expected
        /// (`"env:ANTHROPIC_API_KEY"`, `"vault:secret/llm"`).
        source_hint: Option<String>,
    },

    /// The credential resolved successfully but the provider rejected
    /// it (HTTP 401/403, vendor-specific "invalid token" payloads).
    /// Distinct from [`Self::Expired`] because retries against the
    /// same source will keep failing — the operator must rotate the
    /// secret or fix the IAM grant.
    #[error("auth: credential refused: {message}")]
    Refused {
        /// Provider-supplied rejection message, normalised.
        message: String,
    },

    /// The credential's TTL elapsed and the refresh path failed (or
    /// is not configured). Caller can react by triggering a
    /// rotation; downstream retry policies often back off briefly
    /// and retry once.
    #[error("auth: credential expired{}", message.as_ref().map(|m| format!(": {m}")).unwrap_or_default())]
    Expired {
        /// Optional detail (`"refresh endpoint returned 503"`).
        message: Option<String>,
    },

    /// Resolving the credential required talking to a remote service
    /// (vault, IMDS, KMS, OAuth refresh endpoint) and that service
    /// was unreachable. Distinct from [`Self::Refused`] because the
    /// credential itself may still be valid; the issue is transport
    /// to the credential source.
    #[error("auth: credential source unreachable: {message}")]
    SourceUnreachable {
        /// Description of the failed source call.
        message: String,
    },
}

impl AuthError {
    /// Build a `Missing` variant with no source hint.
    #[must_use]
    pub const fn missing() -> Self {
        Self::Missing { source_hint: None }
    }

    /// Build a `Missing` variant labelled with the source the caller
    /// expected (`"env:OPENAI_API_KEY"`, `"chained:[env, vault]"`).
    pub fn missing_from(source: impl Into<String>) -> Self {
        Self::Missing {
            source_hint: Some(source.into()),
        }
    }

    /// Build a `Refused` variant from the provider's rejection message.
    pub fn refused(message: impl Into<String>) -> Self {
        Self::Refused {
            message: message.into(),
        }
    }

    /// Build an `Expired` variant with no extra detail.
    #[must_use]
    pub const fn expired() -> Self {
        Self::Expired { message: None }
    }

    /// Build an `Expired` variant with refresh-path detail.
    pub fn expired_with(message: impl Into<String>) -> Self {
        Self::Expired {
            message: Some(message.into()),
        }
    }

    /// Build a `SourceUnreachable` variant.
    pub fn source_unreachable(message: impl Into<String>) -> Self {
        Self::SourceUnreachable {
            message: message.into(),
        }
    }
}

impl From<AuthError> for Error {
    fn from(err: AuthError) -> Self {
        Self::Auth(err)
    }
}

/// Header pair a transport adds immediately before sending.
///
/// The value is a `SecretString` so logging or `Debug` output never leaks
/// it; the transport calls [`ExposeSecret::expose_secret`] only when
/// assembling the wire request.
#[derive(Clone, Debug)]
pub struct Credentials {
    /// HTTP header name (`x-api-key`, `authorization`, etc.).
    pub header_name: http::HeaderName,
    /// Secret-wrapped header value. Use `expose_secret()` at send time.
    pub header_value: SecretString,
}

/// Async source-of-truth for credentials.
///
/// Implementors may cache, refresh OAuth tokens, call a vault, etc. The
/// transport calls `resolve()` once per request and discards the result
/// after the headers are written.
#[async_trait]
pub trait CredentialProvider: Send + Sync + 'static {
    /// Resolve current credentials. Long-running impls should respect
    /// `tokio` cancellation in their internals; the transport supplies the
    /// `ExecutionContext` indirectly via the surrounding async task.
    async fn resolve(&self) -> Result<Credentials>;
}

/// Static API-key provider. The header name is configurable so this works
/// for both Anthropic (`x-api-key`) and any other vendor that uses a
/// non-`Authorization` header.
#[derive(Debug)]
pub struct ApiKeyProvider {
    header_name: http::HeaderName,
    api_key: SecretString,
}

impl ApiKeyProvider {
    /// Construct from a header name and a raw key string.
    ///
    /// Returns `Error::Config` if `header_name` cannot be parsed as a valid
    /// HTTP header name.
    pub fn new(header_name: &str, api_key: impl Into<SecretString>) -> Result<Self> {
        let header_name = http::HeaderName::from_bytes(header_name.as_bytes())
            .map_err(|e| Error::config(format!("invalid header name: {e}")))?;
        Ok(Self {
            header_name,
            api_key: api_key.into(),
        })
    }

    /// Convenience: Anthropic-style `x-api-key` provider.
    pub fn anthropic(api_key: impl Into<SecretString>) -> Self {
        Self {
            header_name: http::HeaderName::from_static("x-api-key"),
            api_key: api_key.into(),
        }
    }
}

#[async_trait]
impl CredentialProvider for ApiKeyProvider {
    async fn resolve(&self) -> Result<Credentials> {
        Ok(Credentials {
            header_name: self.header_name.clone(),
            header_value: self.api_key.clone(),
        })
    }
}

/// `Authorization: Bearer <token>` provider. Used by `OpenAI`, Gemini, and
/// most cloud transports as the inner credential.
#[derive(Debug)]
pub struct BearerProvider {
    token: SecretString,
}

impl BearerProvider {
    /// Construct from a raw token string.
    pub fn new(token: impl Into<SecretString>) -> Self {
        Self {
            token: token.into(),
        }
    }
}

#[async_trait]
impl CredentialProvider for BearerProvider {
    async fn resolve(&self) -> Result<Credentials> {
        let formatted = format!("Bearer {}", self.token.expose_secret());
        Ok(Credentials {
            header_name: http::header::AUTHORIZATION,
            header_value: SecretString::from(formatted),
        })
    }
}

/// TTL cache wrapping any inner [`CredentialProvider`]. Resolves
/// once, hands back the cached value until `ttl` elapses, then
/// refreshes by calling the inner provider exactly once even under
/// concurrent load (concurrent waiters share the in-flight refresh
/// future).
///
/// The wrapper is the recommended baseline for production credential
/// chains: short-lived bearer tokens (OAuth, AWS STS, Azure AAD) all
/// expose a TTL, and refusing to cache hammers the credential source
/// once per request.
///
/// On refresh failure the cache surfaces the inner error and does
/// **not** poison the slot — a subsequent call retries. This keeps
/// transient credential-source outages from cascading into
/// permanent agent failure.
pub struct CachedCredentialProvider<P> {
    inner: std::sync::Arc<P>,
    ttl: std::time::Duration,
    state: tokio::sync::Mutex<CachedState>,
}

struct CachedState {
    cached: Option<(Credentials, std::time::Instant)>,
}

impl<P> CachedCredentialProvider<P>
where
    P: CredentialProvider,
{
    /// Wrap `inner` with a TTL cache. The first call to `resolve`
    /// populates the cache; subsequent calls within `ttl` reuse it.
    pub fn new(inner: P, ttl: std::time::Duration) -> Self {
        Self {
            inner: std::sync::Arc::new(inner),
            ttl,
            state: tokio::sync::Mutex::new(CachedState { cached: None }),
        }
    }

    /// Convenience constructor for impls already wrapped in `Arc`.
    pub fn from_arc(inner: std::sync::Arc<P>, ttl: std::time::Duration) -> Self {
        Self {
            inner,
            ttl,
            state: tokio::sync::Mutex::new(CachedState { cached: None }),
        }
    }

    /// Effective TTL.
    pub const fn ttl(&self) -> std::time::Duration {
        self.ttl
    }
}

#[async_trait]
impl<P> CredentialProvider for CachedCredentialProvider<P>
where
    P: CredentialProvider,
{
    async fn resolve(&self) -> Result<Credentials> {
        let mut guard = self.state.lock().await;
        if let Some((creds, fetched_at)) = &guard.cached
            && fetched_at.elapsed() < self.ttl
        {
            let cached = creds.clone();
            // Drop the guard before returning so callers waiting
            // on the lock don't block on the cache-hit path's
            // implicit drop at end-of-scope.
            drop(guard);
            return Ok(cached);
        }
        // Slot is empty or stale — refresh under the lock so
        // concurrent callers share the result rather than pile on
        // the credential source.
        let fresh = self.inner.resolve().await?;
        guard.cached = Some((fresh.clone(), std::time::Instant::now()));
        drop(guard);
        Ok(fresh)
    }
}

/// Try a sequence of [`CredentialProvider`]s in order, returning
/// the first one that resolves successfully. A provider that
/// returns [`AuthError::Missing`] (the configured "not my source"
/// signal) is skipped; any other [`enum@Error`] short-circuits and is
/// returned to the caller — failed-but-real credential sources
/// must surface their failure rather than silently fall through.
///
/// Typical layout: try environment first, fall back to vault, fall
/// back to instance metadata. The chain is built once at
/// transport-construction time; `resolve` is hot-path safe.
pub struct ChainedCredentialProvider {
    providers: Vec<std::sync::Arc<dyn CredentialProvider>>,
}

impl ChainedCredentialProvider {
    /// Build a chain from the supplied provider list. An empty list
    /// is permitted but pointless — every `resolve` call returns
    /// [`AuthError::Missing`].
    #[must_use]
    pub const fn new(providers: Vec<std::sync::Arc<dyn CredentialProvider>>) -> Self {
        Self { providers }
    }

    /// Number of providers in the chain.
    #[must_use]
    pub fn len(&self) -> usize {
        self.providers.len()
    }

    /// True when no providers are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }
}

#[async_trait]
impl CredentialProvider for ChainedCredentialProvider {
    async fn resolve(&self) -> Result<Credentials> {
        for provider in &self.providers {
            match provider.resolve().await {
                Ok(creds) => return Ok(creds),
                // Missing → fall through to the next provider; any
                // other error is a real failure that must surface
                // rather than be masked.
                Err(Error::Auth(AuthError::Missing { .. })) => {}
                Err(other) => return Err(other),
            }
        }
        Err(AuthError::missing_from(format!(
            "chained: {} provider(s) exhausted",
            self.providers.len()
        ))
        .into())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// Minimal counting provider used to verify cache hits and chain ordering.
    struct CountingProvider {
        calls: Arc<AtomicUsize>,
        outcome: Outcome,
    }

    enum Outcome {
        Ok(SecretString),
        Missing,
        Refused(String),
    }

    impl CountingProvider {
        fn ok(token: &str) -> (Self, Arc<AtomicUsize>) {
            let calls = Arc::new(AtomicUsize::new(0));
            (
                Self {
                    calls: calls.clone(),
                    outcome: Outcome::Ok(SecretString::from(token.to_owned())),
                },
                calls,
            )
        }

        fn missing() -> (Self, Arc<AtomicUsize>) {
            let calls = Arc::new(AtomicUsize::new(0));
            (
                Self {
                    calls: calls.clone(),
                    outcome: Outcome::Missing,
                },
                calls,
            )
        }

        fn refused(msg: &str) -> (Self, Arc<AtomicUsize>) {
            let calls = Arc::new(AtomicUsize::new(0));
            (
                Self {
                    calls: calls.clone(),
                    outcome: Outcome::Refused(msg.to_owned()),
                },
                calls,
            )
        }
    }

    #[async_trait]
    impl CredentialProvider for CountingProvider {
        async fn resolve(&self) -> Result<Credentials> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            match &self.outcome {
                Outcome::Ok(token) => Ok(Credentials {
                    header_name: http::header::AUTHORIZATION,
                    header_value: token.clone(),
                }),
                Outcome::Missing => Err(AuthError::missing().into()),
                Outcome::Refused(msg) => Err(AuthError::refused(msg.clone()).into()),
            }
        }
    }

    #[tokio::test]
    async fn cached_provider_serves_from_cache_within_ttl() {
        let (inner, calls) = CountingProvider::ok("tok-1");
        let cached = CachedCredentialProvider::new(inner, Duration::from_secs(60));
        let _ = cached.resolve().await.unwrap();
        let _ = cached.resolve().await.unwrap();
        let _ = cached.resolve().await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn cached_provider_refreshes_after_ttl() {
        let (inner, calls) = CountingProvider::ok("tok-2");
        let cached = CachedCredentialProvider::new(inner, Duration::from_millis(20));
        let _ = cached.resolve().await.unwrap();
        tokio::time::sleep(Duration::from_millis(40)).await;
        let _ = cached.resolve().await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn chained_provider_falls_through_on_missing() {
        let (a, a_calls) = CountingProvider::missing();
        let (b, b_calls) = CountingProvider::ok("from-b");
        let chain = ChainedCredentialProvider::new(vec![Arc::new(a), Arc::new(b)]);
        let creds = chain.resolve().await.unwrap();
        assert_eq!(creds.header_name, http::header::AUTHORIZATION);
        assert_eq!(creds.header_value.expose_secret(), "from-b");
        assert_eq!(a_calls.load(Ordering::SeqCst), 1);
        assert_eq!(b_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn chained_provider_short_circuits_on_real_error() {
        // Refused is NOT Missing — chain must surface immediately
        // rather than mask the rejection by falling through.
        let (a, a_calls) = CountingProvider::refused("vault: 401");
        let (b, b_calls) = CountingProvider::ok("from-b");
        let chain = ChainedCredentialProvider::new(vec![Arc::new(a), Arc::new(b)]);
        let err = chain.resolve().await.unwrap_err();
        assert!(matches!(err, Error::Auth(AuthError::Refused { .. })));
        assert_eq!(a_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            b_calls.load(Ordering::SeqCst),
            0,
            "chain must not consult later providers after a real failure"
        );
    }

    #[tokio::test]
    async fn chained_provider_returns_missing_when_all_sources_exhausted() {
        let (a, _) = CountingProvider::missing();
        let (b, _) = CountingProvider::missing();
        let chain = ChainedCredentialProvider::new(vec![Arc::new(a), Arc::new(b)]);
        let err = chain.resolve().await.unwrap_err();
        assert!(matches!(err, Error::Auth(AuthError::Missing { .. })));
    }
}
