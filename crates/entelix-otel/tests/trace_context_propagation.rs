//! `TraceContextTransport` integration — wraps a fake transport and
//! verifies that:
//!
//! - When a parent OTel span is active, `traceparent` (and
//!   `tracestate` when set) headers appear on the inner transport's
//!   request.
//! - When no OTel context is set, no propagation headers leak.
//!
//! The test uses a `TraceContextPropagator` registered via
//! `opentelemetry::global::set_text_map_propagator` so the assertion
//! exercises the production code path end-to-end.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::significant_drop_in_scrutinee,
    clippy::significant_drop_tightening
)]

use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use opentelemetry::global;
use opentelemetry::trace::{TraceContextExt, TracerProvider};
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::SdkTracerProvider;
use parking_lot::Mutex;

use entelix_core::codecs::EncodedRequest;
use entelix_core::context::ExecutionContext;
use entelix_core::error::Result;
use entelix_core::transports::{Transport, TransportResponse, TransportStream};
use entelix_otel::TraceContextTransport;

/// Records every outbound request the transport receives so the test
/// can assert on header presence.
#[derive(Debug, Default)]
struct RecordingTransport {
    requests: Mutex<Vec<EncodedRequest>>,
}

#[async_trait]
impl Transport for RecordingTransport {
    fn name(&self) -> &'static str {
        "recording"
    }

    async fn send(
        &self,
        request: EncodedRequest,
        _ctx: &ExecutionContext,
    ) -> Result<TransportResponse> {
        self.requests.lock().push(request);
        Ok(TransportResponse {
            status: 200,
            headers: http::HeaderMap::new(),
            body: Bytes::new(),
        })
    }

    async fn send_streaming(
        &self,
        request: EncodedRequest,
        _ctx: &ExecutionContext,
    ) -> Result<TransportStream> {
        self.requests.lock().push(request);
        Ok(TransportStream {
            status: 200,
            headers: http::HeaderMap::new(),
            body: Box::pin(futures::stream::empty()),
        })
    }
}

fn install_propagator() {
    global::set_text_map_propagator(TraceContextPropagator::new());
}

#[tokio::test]
async fn traceparent_is_injected_under_active_otel_context() {
    install_propagator();

    // Build an OTel SDK tracer + capture provider, register
    // tracing-opentelemetry layer so spans propagate.
    let provider = SdkTracerProvider::builder().build();
    let tracer = provider.tracer("entelix-test");
    let subscriber = tracing_subscriber::Registry::default()
        .with(tracing_opentelemetry::layer().with_tracer(tracer));
    let _guard = tracing::subscriber::set_default(subscriber);

    // Build the recording stack.
    let recording = Arc::new(RecordingTransport::default());
    let traced = TraceContextTransport::from_arc(Arc::clone(&recording));

    // Run the send inside a span so `Span::current()` carries an OTel context.
    let span = tracing::info_span!("test_outer_span");
    let _entered = span.enter();

    let request = EncodedRequest::post_json("/v1/anything", Bytes::from_static(b"{}"));
    let _ = traced
        .send(request, &ExecutionContext::new())
        .await
        .unwrap();

    let captured = recording.requests.lock();
    assert_eq!(captured.len(), 1);
    let headers = &captured[0].headers;
    assert!(
        headers.contains_key("traceparent"),
        "traceparent must be injected under an active span; got {headers:?}"
    );
}

#[tokio::test]
async fn no_propagation_headers_when_no_active_span() {
    install_propagator();

    let recording = Arc::new(RecordingTransport::default());
    let traced = TraceContextTransport::from_arc(Arc::clone(&recording));

    let request = EncodedRequest::post_json("/v1/anything", Bytes::from_static(b"{}"));
    let _ = traced
        .send(request, &ExecutionContext::new())
        .await
        .unwrap();

    let captured = recording.requests.lock();
    let headers = &captured[0].headers;
    // Without an active span the propagator has no context to write —
    // the W3C TraceContext propagator no-ops cleanly in that case.
    assert!(!headers.contains_key("traceparent"));
}

#[tokio::test]
async fn streaming_path_also_injects_traceparent() {
    install_propagator();
    let provider = SdkTracerProvider::builder().build();
    let tracer = provider.tracer("entelix-test-stream");
    let subscriber = tracing_subscriber::Registry::default()
        .with(tracing_opentelemetry::layer().with_tracer(tracer));
    let _guard = tracing::subscriber::set_default(subscriber);

    let recording = Arc::new(RecordingTransport::default());
    let traced = TraceContextTransport::from_arc(Arc::clone(&recording));

    let span = tracing::info_span!("test_stream_span");
    let _entered = span.enter();

    let request = EncodedRequest::post_json("/v1/stream", Bytes::from_static(b"{}"));
    let _ = traced
        .send_streaming(request, &ExecutionContext::new())
        .await
        .unwrap();

    let captured = recording.requests.lock();
    assert!(captured[0].headers.contains_key("traceparent"));
}

// `TraceContextExt` import keeps the trait in scope — the test
// indirectly exercises it by entering a span with an OTel context.
#[allow(dead_code)]
fn _force_link() {
    let _ = TraceContextExt::span(&opentelemetry::Context::current());
}

use tracing_subscriber::layer::SubscriberExt as _;
