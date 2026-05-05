//! `BedrockTransport` — `entelix_core::transports::Transport` over
//! AWS Bedrock Runtime.
//!
//! Two auth modes:
//! - [`BedrockAuth::SigV4`]: full SigV4 signing via
//!   [`crate::bedrock::BedrockSigner`] + [`BedrockCredentialProvider`].
//! - [`BedrockAuth::Bearer`]: when the operator has a static bearer
//!   token (e.g. `AWS_BEARER_TOKEN_BEDROCK` for Anthropic-on-Bedrock
//!   via the Anthropic-managed bridge).
//!
//! Streaming: `send_streaming` consumes the response body bytes,
//! feeds them into [`crate::bedrock::EventStreamDecoder`], and
//! re-emits each frame's payload as a separate chunk in the returned
//! `TransportStream::body`. Codec layers see only the JSON payloads —
//! the binary envelope is unwrapped here (codec/transport
//! orthogonality, invariant 4).

use std::sync::Arc;

use bytes::Bytes;
use futures::StreamExt;
use secrecy::{ExposeSecret, SecretString};

use entelix_core::codecs::EncodedRequest;
use entelix_core::context::ExecutionContext;
use entelix_core::error::{Error, Result};
use entelix_core::transports::{Transport, TransportResponse, TransportStream};

use crate::CloudError;
use crate::bedrock::credential::BedrockCredentialProvider;
use crate::bedrock::event_stream::EventStreamDecoder;
use crate::bedrock::signer::BedrockSigner;

/// Auth strategy for [`BedrockTransport`].
#[derive(Clone)]
#[non_exhaustive]
pub enum BedrockAuth {
    /// SigV4 signing via the AWS credential chain. Default for
    /// production AWS deployments.
    SigV4 {
        /// Resolves credentials per-call so rotated keys are picked
        /// up automatically.
        provider: BedrockCredentialProvider,
    },
    /// Static bearer token. Used by the Anthropic-on-Bedrock bridge
    /// (`AWS_BEARER_TOKEN_BEDROCK`).
    Bearer {
        /// Pre-resolved token, redacted in `Debug`.
        token: SecretString,
    },
}

impl std::fmt::Debug for BedrockAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SigV4 { .. } => f.write_str("BedrockAuth::SigV4 {{ .. }}"),
            Self::Bearer { .. } => f.write_str("BedrockAuth::Bearer {{ <redacted> }}"),
        }
    }
}

/// AWS Bedrock Runtime transport.
#[derive(Clone)]
pub struct BedrockTransport {
    client: reqwest::Client,
    base_url: String,
    auth: Arc<BedrockAuth>,
    signer: BedrockSigner,
}

impl BedrockTransport {
    /// Start a fluent builder. `region` is required when
    /// `auth = SigV4`.
    pub fn builder() -> BedrockTransportBuilder {
        BedrockTransportBuilder {
            region: None,
            base_url: None,
            auth: None,
        }
    }

    /// Borrow the configured region.
    pub fn region(&self) -> &str {
        self.signer.region()
    }

    /// Borrow the resolved base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    async fn build_signed_headers(
        &self,
        method: &str,
        url: &str,
        request_headers: &http::HeaderMap,
        body: &[u8],
    ) -> Result<Vec<(String, String)>> {
        let mut header_pairs: Vec<(String, String)> = Vec::with_capacity(request_headers.len() + 2);
        for (name, value) in request_headers {
            if let Ok(v) = value.to_str() {
                header_pairs.push((name.as_str().to_owned(), v.to_owned()));
            }
        }
        match self.auth.as_ref() {
            BedrockAuth::SigV4 { provider } => {
                let creds = provider.resolve().await.map_err(Error::from)?;
                let signed = self
                    .signer
                    .sign_request(&creds, method, url, &header_pairs, body)
                    .map_err(Error::from)?;
                header_pairs.extend(signed);
            }
            BedrockAuth::Bearer { token } => {
                header_pairs.push((
                    "authorization".to_owned(),
                    format!("Bearer {}", token.expose_secret()),
                ));
            }
        }
        Ok(header_pairs)
    }

    fn apply_pairs<'a>(
        request: reqwest::RequestBuilder,
        pairs: &'a [(String, String)],
    ) -> reqwest::RequestBuilder {
        let mut req = request;
        for (name, value) in pairs {
            req = req.header(name.as_str(), value.as_str());
        }
        req
    }
}

#[async_trait::async_trait]
impl Transport for BedrockTransport {
    fn name(&self) -> &'static str {
        "bedrock"
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
        let pairs = self
            .build_signed_headers(
                request.method.as_str(),
                &url,
                &request.headers,
                &request.body,
            )
            .await?;
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
        let body = response
            .bytes()
            .await
            .map_err(|e| Error::provider_http(status, format!("response body read failed: {e}")))?;
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
        let pairs = self
            .build_signed_headers(
                request.method.as_str(),
                &url,
                &request.headers,
                &request.body,
            )
            .await?;
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
            let mut decoder = EventStreamDecoder::new();
            let mut s = raw_stream;
            loop {
                tokio::select! {
                    biased;
                    () = cancellation.cancelled() => {
                        yield Err(Error::Cancelled);
                        return;
                    }
                    item = s.next() => match item {
                        Some(Ok(bytes)) => {
                            decoder.push(&bytes);
                            loop {
                                match decoder.next_frame() {
                                    Ok(Some(frame)) => {
                                        // Pre-release: surface only the
                                        // payload bytes — codec layers see
                                        // a logical NDJSON-like stream of
                                        // event JSON blobs.
                                        yield Ok(Bytes::from(frame.payload));
                                    }
                                    Ok(None) => break,
                                    Err(e) => {
                                        yield Err(CloudError::EventStream(e).into());
                                        return;
                                    }
                                }
                            }
                        }
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

/// Fluent builder for [`BedrockTransport`].
#[must_use]
pub struct BedrockTransportBuilder {
    region: Option<String>,
    base_url: Option<String>,
    auth: Option<BedrockAuth>,
}

impl BedrockTransportBuilder {
    /// AWS region (`us-east-1`, `eu-west-1`, …). Required for SigV4.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Override the base URL. Defaults to
    /// `https://bedrock-runtime.<region>.amazonaws.com` when
    /// `region` is set.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Set the auth strategy.
    pub fn with_auth(mut self, auth: BedrockAuth) -> Self {
        self.auth = Some(auth);
        self
    }

    /// Resolve everything and return the transport.
    pub fn build(self) -> Result<BedrockTransport> {
        let region = self
            .region
            .ok_or_else(|| Error::config("BedrockTransport: region is required"))?;
        let base_url = self
            .base_url
            .unwrap_or_else(|| format!("https://bedrock-runtime.{region}.amazonaws.com")); // silent-fallback-ok: builder default — regional Bedrock Runtime endpoint when base_url is not overridden
        let auth = self
            .auth
            .ok_or_else(|| Error::config("BedrockTransport: auth is required"))?;
        let client = reqwest::Client::builder()
            .build()
            .map_err(|e| Error::config(format!("failed to build HTTP client: {e}")))?;
        let signer = BedrockSigner::new(region);
        Ok(BedrockTransport {
            client,
            base_url,
            auth: Arc::new(auth),
            signer,
        })
    }
}
