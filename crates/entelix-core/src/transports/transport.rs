//! `Transport` — stateful HTTP carrier that ferries [`EncodedRequest`] →
//! [`TransportResponse`] (one-shot) or [`TransportStream`] (streaming).
//!
//! Concrete transports own a `reqwest::Client`, a `CredentialProvider`,
//! retry policy, and (for cloud variants) signing logic. They never see
//! the IR; they ship bytes. This decoupling is invariant 4 — codec and
//! transport are orthogonal.

use std::pin::Pin;

use bytes::Bytes;
use futures::Stream;

use crate::codecs::EncodedRequest;
use crate::context::ExecutionContext;
use crate::error::Result;

/// What the provider returned in a one-shot response.
#[derive(Clone, Debug)]
pub struct TransportResponse {
    /// HTTP status code from the provider.
    pub status: u16,
    /// Response headers — used by codecs that read e.g. `request-id` for
    /// observability.
    pub headers: http::HeaderMap,
    /// Raw response body.
    pub body: Bytes,
}

/// What the provider returned in a streaming response. The body arrives
/// as an async byte stream; codecs feed it into `decode_stream`.
pub struct TransportStream {
    /// HTTP status code resolved at handshake time. For SSE, an
    /// in-protocol error encoded later in the stream surfaces as a
    /// stream-level `Result::Err` instead.
    pub status: u16,
    /// Response headers from the handshake.
    pub headers: http::HeaderMap,
    /// Incremental body chunks. Closes when the provider terminates
    /// the stream.
    pub body: Pin<Box<dyn Stream<Item = Result<Bytes>> + Send + 'static>>,
}

impl std::fmt::Debug for TransportStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransportStream")
            .field("status", &self.status)
            .field("headers", &self.headers)
            .field("body", &"<byte stream>")
            .finish()
    }
}

/// HTTP carrier. Concrete impls inject credentials, sign requests, manage
/// connection pools, and apply retry policy.
#[async_trait::async_trait]
pub trait Transport: Send + Sync + 'static {
    /// Stable transport identifier — `"direct"`, `"bedrock"`, `"vertex"`,
    /// `"foundry"`. Used in logs and metrics tags.
    fn name(&self) -> &'static str;

    /// Send a codec-produced request and return the raw response. The
    /// transport applies credentials immediately before send and discards
    /// the resolved value before returning (invariant 10 — tokens never
    /// reach `ExecutionContext`).
    async fn send(
        &self,
        request: EncodedRequest,
        ctx: &ExecutionContext,
    ) -> Result<TransportResponse>;

    /// Streaming variant of [`Self::send`]: opens a connection that
    /// returns body bytes incrementally (typically `text/event-stream`).
    /// The default impl falls back to `send`, wrapping the buffered body
    /// in a single-chunk stream so codecs that rely on
    /// [`Codec::decode_stream`] still work — at the cost of no real-time
    /// delivery. Concrete transports override to keep the connection
    /// open and yield bytes as they arrive.
    ///
    /// [`Codec::decode_stream`]: crate::codecs::Codec::decode_stream
    async fn send_streaming(
        &self,
        request: EncodedRequest,
        ctx: &ExecutionContext,
    ) -> Result<TransportStream> {
        let response = self.send(request, ctx).await?;
        let body = response.body;
        let stream = futures::stream::once(async move { Ok(body) });
        Ok(TransportStream {
            status: response.status,
            headers: response.headers,
            body: Box::pin(stream),
        })
    }
}
