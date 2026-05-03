//! `FoundryTransport` — `entelix_core::transports::Transport` over
//! Azure AI Foundry. Two auth modes:
//! - [`FoundryAuth::ApiKey`]: static `api-key` header (the bridge
//!   path most operators use day-1).
//! - [`FoundryAuth::Entra`]: OAuth via `azure_identity` flowing
//!   through [`crate::refresh::CachedTokenProvider`].

use std::sync::Arc;

use bytes::Bytes;
use futures::StreamExt;
use secrecy::{ExposeSecret, SecretString};

use entelix_core::codecs::EncodedRequest;
use entelix_core::context::ExecutionContext;
use entelix_core::error::{Error, Result};
use entelix_core::transports::{Transport, TransportResponse, TransportStream};

use crate::CloudError;
use crate::refresh::{CachedTokenProvider, TokenRefresher};

/// Auth strategy for [`FoundryTransport`].
#[derive(Clone)]
#[non_exhaustive]
pub enum FoundryAuth {
    /// Static API key — sent as `api-key: {value}`.
    ApiKey {
        /// Pre-resolved key, redacted in `Debug`.
        token: SecretString,
    },
    /// Entra ID (Azure AD) OAuth — token resolved through the
    /// supplied refresher.
    Entra {
        /// Refresher driven by `azure_identity`.
        refresher: Arc<dyn TokenRefresher<SecretString>>,
    },
}

impl std::fmt::Debug for FoundryAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ApiKey { .. } => f.write_str("FoundryAuth::ApiKey {{ <redacted> }}"),
            Self::Entra { .. } => f.write_str("FoundryAuth::Entra {{ .. }}"),
        }
    }
}

#[derive(Clone)]
enum ResolvedAuth {
    ApiKey(SecretString),
    Entra(Arc<CachedTokenProvider<SecretString>>),
}

/// Azure AI Foundry HTTP transport.
#[derive(Clone)]
pub struct FoundryTransport {
    client: reqwest::Client,
    base_url: String,
    auth: ResolvedAuth,
}

impl FoundryTransport {
    /// Start a fluent builder.
    pub fn builder() -> FoundryTransportBuilder {
        FoundryTransportBuilder {
            base_url: None,
            auth: None,
        }
    }

    /// Borrow the resolved base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    async fn build_headers(
        &self,
        request_headers: &http::HeaderMap,
    ) -> Result<Vec<(String, String)>> {
        let mut pairs: Vec<(String, String)> = Vec::with_capacity(request_headers.len() + 1);
        for (name, value) in request_headers {
            if let Ok(v) = value.to_str() {
                pairs.push((name.as_str().to_owned(), v.to_owned()));
            }
        }
        match &self.auth {
            ResolvedAuth::ApiKey(token) => {
                pairs.push(("api-key".to_owned(), token.expose_secret().to_owned()));
            }
            ResolvedAuth::Entra(refreshable) => {
                let token = refreshable.current().await.map_err(Error::from)?;
                pairs.push((
                    "authorization".to_owned(),
                    format!("Bearer {}", token.expose_secret()),
                ));
            }
        }
        Ok(pairs)
    }

    fn maybe_invalidate_on_unauthorized(&self, status: u16) {
        if status == 401
            && let ResolvedAuth::Entra(token) = &self.auth
        {
            token.invalidate();
        }
    }

    fn apply_pairs(
        req: reqwest::RequestBuilder,
        pairs: &[(String, String)],
    ) -> reqwest::RequestBuilder {
        let mut out = req;
        for (name, value) in pairs {
            out = out.header(name.as_str(), value.as_str());
        }
        out
    }
}

#[async_trait::async_trait]
impl Transport for FoundryTransport {
    fn name(&self) -> &'static str {
        "foundry"
    }

    async fn send(
        &self,
        request: EncodedRequest,
        ctx: &ExecutionContext,
    ) -> Result<TransportResponse> {
        if ctx.is_cancelled() {
            return Err(Error::Cancelled);
        }
        let url = format!("{}{}", self.base_url, request.path);
        // Header build can stall on Entra token refresh — race the
        // caller's cancellation token so a cancel surfaces within
        // one HTTP round-trip instead of waiting for the full
        // OAuth refresh to complete.
        let pairs = tokio::select! {
            biased;
            () = ctx.cancellation().cancelled() => return Err(Error::Cancelled),
            p = self.build_headers(&request.headers) => p?,
        };
        let body_bytes = Bytes::clone(&request.body);
        let mut http_req = self.client.request(request.method.clone(), &url);
        http_req = Self::apply_pairs(http_req, &pairs).body(body_bytes);
        let response = tokio::select! {
            biased;
            () = ctx.cancellation().cancelled() => return Err(Error::Cancelled),
            r = http_req.send() => r,
        }
        .map_err(Error::provider_network_from)?;
        let status = response.status().as_u16();
        let headers = response.headers().clone();
        let body = response.bytes().await.map_err(|e| {
            Error::provider_http(status, format!("response body read failed: {e}"))
        })?;
        self.maybe_invalidate_on_unauthorized(status);
        Ok(TransportResponse {
            status,
            headers,
            body,
        })
    }

    #[allow(tail_expr_drop_order)]
    async fn send_streaming(
        &self,
        request: EncodedRequest,
        ctx: &ExecutionContext,
    ) -> Result<TransportStream> {
        if ctx.is_cancelled() {
            return Err(Error::Cancelled);
        }
        let url = format!("{}{}", self.base_url, request.path);
        // Header build can stall on Entra token refresh — race the
        // caller's cancellation token so a cancel surfaces within
        // one HTTP round-trip instead of waiting for the full
        // OAuth refresh to complete.
        let pairs = tokio::select! {
            biased;
            () = ctx.cancellation().cancelled() => return Err(Error::Cancelled),
            p = self.build_headers(&request.headers) => p?,
        };
        let body_bytes = Bytes::clone(&request.body);
        let mut http_req = self.client.request(request.method.clone(), &url);
        http_req = Self::apply_pairs(http_req, &pairs).body(body_bytes);
        let response = tokio::select! {
            biased;
            () = ctx.cancellation().cancelled() => return Err(Error::Cancelled),
            r = http_req.send() => r,
        }
        .map_err(Error::provider_network_from)?;
        let status = response.status().as_u16();
        let headers = response.headers().clone();
        self.maybe_invalidate_on_unauthorized(status);
        if !(200..300).contains(&status) {
            let body = response.bytes().await.unwrap_or_else(|_| Bytes::new()); // silent-fallback-ok: error-response body read already failed; empty body preserves status + headers for caller diagnostics
            let body_stream = futures::stream::once(async move { Ok::<_, Error>(body) });
            return Ok(TransportStream {
                status,
                headers,
                body: Box::pin(body_stream),
            });
        }
        let cancellation = ctx.cancellation().clone();
        let raw_stream = response.bytes_stream();
        let body = async_stream::stream! {
            let mut s = raw_stream;
            loop {
                tokio::select! {
                    biased;
                    () = cancellation.cancelled() => {
                        yield Err(Error::Cancelled);
                        return;
                    }
                    item = s.next() => match item {
                        Some(Ok(b)) => yield Ok(b),
                        Some(Err(e)) => {
                            yield Err(Error::provider_http(status, format!("stream chunk read failed: {e}")));
                            return;
                        }
                        None => return,
                    }
                }
            }
        };
        Ok(TransportStream {
            status,
            headers,
            body: Box::pin(body),
        })
    }
}

/// Fluent builder for [`FoundryTransport`].
#[must_use]
pub struct FoundryTransportBuilder {
    base_url: Option<String>,
    auth: Option<FoundryAuth>,
}

impl FoundryTransportBuilder {
    /// Foundry endpoint base URL — typically
    /// `https://{resource}.services.ai.azure.com/anthropic` or
    /// `https://{resource}.openai.azure.com`. Required.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Pick the auth strategy.
    pub fn with_auth(mut self, auth: FoundryAuth) -> Self {
        self.auth = Some(auth);
        self
    }

    /// Resolve and return the transport.
    pub fn build(self) -> Result<FoundryTransport> {
        let base_url = self
            .base_url
            .ok_or_else(|| Error::config("FoundryTransport: base_url is required"))?;
        let auth = self
            .auth
            .ok_or_else(|| Error::config("FoundryTransport: auth is required"))?;
        let resolved = match auth {
            FoundryAuth::ApiKey { token } => ResolvedAuth::ApiKey(token),
            FoundryAuth::Entra { refresher } => {
                ResolvedAuth::Entra(Arc::new(CachedTokenProvider::new(refresher)))
            }
        };
        let client = reqwest::Client::builder()
            .build()
            .map_err(|e| Error::config(format!("failed to build HTTP client: {e}")))?;
        Ok(FoundryTransport {
            client,
            base_url,
            auth: resolved,
        })
    }
}

const _: fn() = || {
    let _ = std::marker::PhantomData::<CloudError>;
};
