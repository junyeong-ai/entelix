//! `DirectTransport` — a `reqwest`-based HTTPS carrier that talks straight
//! to the vendor's public endpoint (e.g. `api.anthropic.com`,
//! `api.openai.com`).
//!
//! Cloud-signed transports (Bedrock `SigV4`, Vertex `GoogleAuth`, Foundry
//! `AAD`) live in `entelix-cloud`. This crate ships only the minimum
//! needed to reach `api.anthropic.com`, `api.openai.com`, etc.

use std::sync::Arc;

use bytes::Bytes;
use futures::StreamExt;
use secrecy::ExposeSecret;

use crate::auth::CredentialProvider;
use crate::codecs::EncodedRequest;
use crate::context::ExecutionContext;
use crate::error::{Error, Result};
use crate::transports::transport::{Transport, TransportResponse, TransportStream};

/// HTTPS transport that adds credentials at send time and forwards the
/// codec-produced bytes verbatim.
///
/// Cheap to clone — internal state is reference-counted.
#[derive(Clone)]
pub struct DirectTransport {
    client: reqwest::Client,
    base_url: String,
    credentials: Arc<dyn CredentialProvider>,
}

impl DirectTransport {
    /// Build a transport pointing at `base_url` and pulling credentials from
    /// `credentials`. Returns `Error::Config` when the underlying HTTP
    /// client cannot be initialized (e.g. TLS feature mis-configured).
    pub fn new(
        base_url: impl Into<String>,
        credentials: Arc<dyn CredentialProvider>,
    ) -> Result<Self> {
        let client = reqwest::Client::builder()
            .build()
            .map_err(|e| Error::config(format!("failed to build HTTP client: {e}")))?;
        Ok(Self {
            client,
            base_url: base_url.into(),
            credentials,
        })
    }

    /// Convenience: transport pre-pointed at `https://api.anthropic.com`.
    pub fn anthropic(credentials: Arc<dyn CredentialProvider>) -> Result<Self> {
        Self::new("https://api.anthropic.com", credentials)
    }

    /// Convenience: transport pre-pointed at `https://api.openai.com`.
    /// Pair with [`crate::auth::BearerProvider`] — OpenAI uses
    /// `Authorization: Bearer <key>`.
    pub fn openai(credentials: Arc<dyn CredentialProvider>) -> Result<Self> {
        Self::new("https://api.openai.com", credentials)
    }

    /// Convenience: transport pre-pointed at
    /// `https://generativelanguage.googleapis.com`. Pair with
    /// [`crate::auth::BearerProvider`] for Gemini's `Authorization:
    /// Bearer <key>` flow.
    pub fn gemini(credentials: Arc<dyn CredentialProvider>) -> Result<Self> {
        Self::new("https://generativelanguage.googleapis.com", credentials)
    }

    /// Borrow the configured base URL — useful for diagnostics.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

#[async_trait::async_trait]
impl Transport for DirectTransport {
    fn name(&self) -> &'static str {
        "direct"
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
        let creds = self.credentials.resolve().await?;

        let credential_value = http::HeaderValue::from_str(creds.header_value.expose_secret())
            .map_err(|e| Error::config(format!("invalid credential header value: {e}")))?;

        let mut http_req = self.client.request(request.method.clone(), &url);
        for (name, value) in &request.headers {
            http_req = http_req.header(name, value);
        }
        http_req = http_req.header(creds.header_name.clone(), credential_value);
        // Vendor-side dedupe — same key across every retry of one
        // logical call (invariant #17). `RetryService` stamps it
        // before the first attempt; calls that bypass the retry
        // layer can pre-allocate via
        // `ExecutionContext::with_idempotency_key`.
        if let Some(key) = ctx.idempotency_key() {
            http_req = http_req.header("idempotency-key", key);
        }
        http_req = http_req.body(Bytes::clone(&request.body));

        let response = match ctx.deadline() {
            Some(deadline) => {
                let now = tokio::time::Instant::now();
                let timeout = deadline.saturating_duration_since(now);
                tokio::select! {
                    biased;
                    () = ctx.cancellation().cancelled() => return Err(Error::Cancelled),
                    r = tokio::time::timeout(timeout, http_req.send()) => match r {
                        Ok(inner) => inner,
                        Err(_) => return Err(Error::DeadlineExceeded),
                    }
                }
            }
            None => {
                tokio::select! {
                    biased;
                    () = ctx.cancellation().cancelled() => return Err(Error::Cancelled),
                    r = http_req.send() => r,
                }
            }
        };

        let response =
            response.map_err(Error::provider_network_from)?;

        let status = response.status().as_u16();
        let headers = response.headers().clone();
        let body = response.bytes().await.map_err(|e| {
            Error::provider_http(status, format!("response body read failed: {e}"))
        })?;

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
        let creds = self.credentials.resolve().await?;
        let credential_value = http::HeaderValue::from_str(creds.header_value.expose_secret())
            .map_err(|e| Error::config(format!("invalid credential header value: {e}")))?;

        let mut http_req = self.client.request(request.method.clone(), &url);
        for (name, value) in &request.headers {
            http_req = http_req.header(name, value);
        }
        http_req = http_req.header(creds.header_name.clone(), credential_value);
        if let Some(key) = ctx.idempotency_key() {
            http_req = http_req.header("idempotency-key", key);
        }
        http_req = http_req.body(Bytes::clone(&request.body));

        let response = match ctx.deadline() {
            Some(deadline) => {
                let now = tokio::time::Instant::now();
                let timeout = deadline.saturating_duration_since(now);
                tokio::select! {
                    biased;
                    () = ctx.cancellation().cancelled() => return Err(Error::Cancelled),
                    r = tokio::time::timeout(timeout, http_req.send()) => match r {
                        Ok(inner) => inner,
                        Err(_) => return Err(Error::DeadlineExceeded),
                    }
                }
            }
            None => {
                tokio::select! {
                    biased;
                    () = ctx.cancellation().cancelled() => return Err(Error::Cancelled),
                    r = http_req.send() => r,
                }
            }
        };

        let response =
            response.map_err(Error::provider_network_from)?;
        let status = response.status().as_u16();
        let headers = response.headers().clone();

        // Non-2xx: drain the buffered body so callers see the error
        // text rather than a cryptic stream of bytes.
        if !(200..300).contains(&status) {
            let body = response.bytes().await.unwrap_or_else(|_| Bytes::new());
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
                            yield Err(Error::provider_http(
                                status,
                                format!("stream chunk read failed: {e}"),
                            ));
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
