//! `VertexTransport` — `entelix_core::transports::Transport` over
//! GCP Vertex AI. Token refresh through
//! [`crate::refresh::CachedTokenProvider`] (single-flight, parking_lot
//! read fast-path). Quota project header support via
//! [`VertexTransportBuilder::quota_project`].

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

/// GCP Vertex AI HTTP transport.
#[derive(Clone)]
pub struct VertexTransport {
    client: reqwest::Client,
    base_url: String,
    project_id: String,
    location: String,
    quota_project: Option<String>,
    token: Arc<CachedTokenProvider<SecretString>>,
}

impl VertexTransport {
    /// Start a fluent builder.
    pub fn builder() -> VertexTransportBuilder {
        VertexTransportBuilder {
            project_id: None,
            location: None,
            quota_project: None,
            base_url: None,
            refresher: None,
        }
    }

    /// Borrow the configured GCP project id.
    pub fn project_id(&self) -> &str {
        &self.project_id
    }

    /// Borrow the configured GCP location (e.g. `us-central1`,
    /// `global`).
    pub fn location(&self) -> &str {
        &self.location
    }

    /// Resolve the on-wire URL from the codec's emitted path.
    ///
    /// Codecs are project-agnostic by contract (invariant 5) — they
    /// emit *what* resource the request targets without knowing
    /// *which* GCP project / location hosts it. Vertex publisher-
    /// model paths (`/publishers/{provider}/models/{model}:{action}`)
    /// receive the `/v1/projects/{project}/locations/{location}`
    /// prefix here; absolute paths (already carrying the project /
    /// location segments, or codec-internal endpoints that bypass
    /// the publisher routing) flow through verbatim.
    fn resolve_url(&self, path: &str) -> String {
        if path.starts_with("/publishers/") {
            format!(
                "{}/v1/projects/{}/locations/{}{}",
                self.base_url, self.project_id, self.location, path
            )
        } else {
            format!("{}{}", self.base_url, path)
        }
    }

    async fn build_headers(
        &self,
        request_headers: &http::HeaderMap,
    ) -> Result<Vec<(String, String)>> {
        let mut pairs: Vec<(String, String)> = Vec::with_capacity(request_headers.len() + 2);
        for (name, value) in request_headers {
            if let Ok(v) = value.to_str() {
                pairs.push((name.as_str().to_owned(), v.to_owned()));
            }
        }
        let token = self.token.current().await.map_err(Error::from)?;
        pairs.push((
            "authorization".to_owned(),
            format!("Bearer {}", token.expose_secret()),
        ));
        if let Some(qp) = &self.quota_project {
            pairs.push(("x-goog-user-project".to_owned(), qp.clone()));
        }
        Ok(pairs)
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
impl Transport for VertexTransport {
    fn name(&self) -> &'static str {
        "vertex"
    }

    async fn send(
        &self,
        request: EncodedRequest,
        ctx: &ExecutionContext,
    ) -> Result<TransportResponse> {
        if ctx.is_cancelled() {
            return Err(Error::Cancelled);
        }
        let url = self.resolve_url(&request.path);
        // Header build can stall on OAuth token refresh — race the
        // caller's cancellation token so a cancel surfaces within
        // one HTTP round-trip instead of waiting for the full
        // refresh to complete.
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
        let body = response
            .bytes()
            .await
            .map_err(|e| Error::provider_http(status, format!("response body read failed: {e}")))?;
        // 401: token may have rotated mid-flight — invalidate cache so
        // the next call refreshes.
        if status == 401 {
            self.token.invalidate();
        }
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
        let url = self.resolve_url(&request.path);
        // Header build can stall on OAuth token refresh — race the
        // caller's cancellation token so a cancel surfaces within
        // one HTTP round-trip instead of waiting for the full
        // refresh to complete.
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
        if status == 401 {
            self.token.invalidate();
        }
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

/// Fluent builder for [`VertexTransport`].
#[must_use]
pub struct VertexTransportBuilder {
    project_id: Option<String>,
    location: Option<String>,
    quota_project: Option<String>,
    base_url: Option<String>,
    refresher: Option<Arc<dyn TokenRefresher<SecretString>>>,
}

impl VertexTransportBuilder {
    /// GCP project id (`my-gcp-project`). Required.
    pub fn with_project_id(mut self, project_id: impl Into<String>) -> Self {
        self.project_id = Some(project_id.into());
        self
    }

    /// GCP location (`us-central1`, `europe-west4`, `global`).
    /// Required.
    pub fn with_location(mut self, location: impl Into<String>) -> Self {
        self.location = Some(location.into());
        self
    }

    /// Override the project that pays for the API call (required when
    /// the calling principal lacks `serviceusage.services.use` on the
    /// resource project).
    pub fn with_quota_project(mut self, qp: impl Into<String>) -> Self {
        self.quota_project = Some(qp.into());
        self
    }

    /// Override the base URL (default
    /// `https://{location}-aiplatform.googleapis.com` — Vertex
    /// publisher-routed REST endpoint).
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Inject a token refresher. Production usage:
    /// `VertexCredentialProvider::default_chain().await?` wrapped in
    /// `Arc::new(...)`.
    pub fn with_token_refresher(mut self, r: Arc<dyn TokenRefresher<SecretString>>) -> Self {
        self.refresher = Some(r);
        self
    }

    /// Resolve and return the transport.
    pub fn build(self) -> Result<VertexTransport> {
        let project_id = self
            .project_id
            .ok_or_else(|| Error::config("VertexTransport: project_id is required"))?;
        let location = self
            .location
            .ok_or_else(|| Error::config("VertexTransport: location is required"))?;
        let refresher = self
            .refresher
            .ok_or_else(|| Error::config("VertexTransport: token_refresher is required"))?;
        let base_url = self.base_url.unwrap_or_else(|| {
            // silent-fallback-ok: builder default — regional Vertex AI endpoint when base_url is not overridden
            if location == "global" {
                "https://aiplatform.googleapis.com".to_owned()
            } else {
                format!("https://{location}-aiplatform.googleapis.com")
            }
        });
        let client = reqwest::Client::builder()
            .build()
            .map_err(|e| Error::config(format!("failed to build HTTP client: {e}")))?;
        Ok(VertexTransport {
            client,
            base_url,
            project_id,
            location,
            quota_project: self.quota_project,
            token: Arc::new(CachedTokenProvider::new(refresher)),
        })
    }
}

// CloudError is re-exported from the crate root; this `use` keeps it
// visible to doc-link rendering on this module.
const _: fn() = || {
    let _ = std::marker::PhantomData::<CloudError>;
};
