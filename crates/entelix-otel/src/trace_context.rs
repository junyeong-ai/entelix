//! `TraceContextTransport<T>` — `Transport` decorator that injects
//! W3C Trace Context (`traceparent` / `tracestate`) headers into
//! every outbound request.
//!
//! Without this layer, distributed tracing terminates at the
//! `entelix → vendor` HTTP boundary: the entelix-side span tree is
//! complete, but the vendor sees no parent context and cannot link
//! its server-side spans to the calling span. Wrapping the transport
//! is the cleanest place to inject — the underlying `EncodedRequest`
//! is in scope, no codec or layer changes are needed, and operators
//! who don't want propagation simply do not wrap.
//!
//! ## Wiring
//!
//! ```ignore
//! use entelix_core::transports::DirectTransport;
//! use entelix_otel::TraceContextTransport;
//!
//! let transport = TraceContextTransport::new(DirectTransport::new(
//!     "https://api.anthropic.com",
//!     api_key_provider,
//! ));
//! let model = ChatModel::new(codec, transport, "claude-opus-4-7");
//! ```
//!
//! `TraceContextTransport` reads the *globally-installed* OTel
//! propagator (typically W3C TraceContext + Baggage). To register a
//! propagator at process start:
//!
//! ```ignore
//! opentelemetry::global::set_text_map_propagator(
//!     opentelemetry_sdk::propagation::TraceContextPropagator::new(),
//! );
//! ```

use std::sync::Arc;

use async_trait::async_trait;
use opentelemetry::Context as OtelContext;
use opentelemetry::global;
use opentelemetry::propagation::Injector;
use opentelemetry::trace::TraceContextExt;
use tracing::Span;
use tracing_opentelemetry::OpenTelemetrySpanExt;

use entelix_core::codecs::EncodedRequest;
use entelix_core::context::ExecutionContext;
use entelix_core::error::Result;
use entelix_core::transports::{Transport, TransportResponse, TransportStream};

/// `Transport` decorator that stamps W3C Trace Context headers on
/// every outbound request before delegating to the inner transport.
///
/// Cloning is cheap (`Arc`-backed inner). The decorator is a
/// `Transport` itself, so it composes everywhere a `Transport` is
/// expected.
pub struct TraceContextTransport<T: Transport> {
    inner: Arc<T>,
}

impl<T: Transport> TraceContextTransport<T> {
    /// Wrap an existing `Transport`.
    #[must_use]
    pub fn new(inner: T) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Build from an existing `Arc`-shared transport — useful when
    /// the inner is shared across multiple decorators.
    #[must_use]
    pub fn from_arc(inner: Arc<T>) -> Self {
        Self { inner }
    }

    /// Borrow the inner transport.
    #[must_use]
    pub fn inner(&self) -> &Arc<T> {
        &self.inner
    }
}

impl<T: Transport> Clone for TraceContextTransport<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T: Transport> std::fmt::Debug for TraceContextTransport<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TraceContextTransport")
            .field("inner", &self.inner.name())
            .finish()
    }
}

#[async_trait]
impl<T: Transport> Transport for TraceContextTransport<T> {
    fn name(&self) -> &'static str {
        // Distinct identifier so logs / metrics can tell the
        // decorator apart from the inner transport.
        "trace-context"
    }

    async fn send(
        &self,
        mut request: EncodedRequest,
        ctx: &ExecutionContext,
    ) -> Result<TransportResponse> {
        inject_trace_context(&mut request.headers);
        self.inner.send(request, ctx).await
    }

    async fn send_streaming(
        &self,
        mut request: EncodedRequest,
        ctx: &ExecutionContext,
    ) -> Result<TransportStream> {
        inject_trace_context(&mut request.headers);
        self.inner.send_streaming(request, ctx).await
    }
}

/// Inject the *current* OTel context's text-map representation
/// (`traceparent`, `tracestate`, optionally `baggage`) into the
/// outbound headers. The current context is read from the active
/// `tracing` span via `tracing-opentelemetry`'s bridge so the
/// decorator works transparently inside `OtelLayer`-instrumented
/// call paths.
fn inject_trace_context(headers: &mut http::HeaderMap) {
    let cx = current_otel_context();
    global::get_text_map_propagator(|propagator| {
        let mut injector = HeaderMapInjector { headers };
        propagator.inject_context(&cx, &mut injector);
    });
}

/// Resolve the active `OtelContext` — first checks the current
/// `tracing` span (the common case when the call stack runs under
/// `OtelLayer` and `tracing-opentelemetry` is wired), falls back to
/// `OtelContext::current()` (the OTel-native current context).
fn current_otel_context() -> OtelContext {
    let span = Span::current();
    let from_span = span.context();
    // `Span::current()`'s OTel context is `OtelContext::default()`
    // when no opentelemetry layer is installed. In that case fall
    // back to the OTel native current context so callers using the
    // OTel API directly still see propagation.
    if from_span.span().span_context().is_valid() {
        from_span
    } else {
        OtelContext::current()
    }
}

/// Build an `Arc<Fn(&mut HeaderMap)>` that injects the active
/// W3C trace-context (`traceparent` / `tracestate` / `baggage`)
/// into any outbound `http::HeaderMap`.
///
/// Designed for transports outside the `ChatModel` path — the
/// canonical use case is wiring it onto
/// `entelix_mcp::McpServerConfig::with_request_decorator` so
/// distributed traces span the SDK caller and the MCP server it
/// dispatches to. The function reads from the active `tracing`
/// span (via `tracing-opentelemetry`) so it Just Works inside any
/// `OtelLayer`-instrumented call path.
///
/// Returns an `Arc` so a single decorator is shared across every
/// MCP client + every concurrent request; cheap to clone.
#[must_use]
pub fn trace_context_injector() -> Arc<dyn Fn(&mut http::HeaderMap) + Send + Sync + 'static> {
    Arc::new(inject_trace_context)
}

/// `Injector` adapter that writes header keys/values into a
/// `http::HeaderMap`. Skips entries that fail header-name or
/// header-value validation (the propagator emits well-formed values
/// in practice; the guard is defensive).
struct HeaderMapInjector<'a> {
    headers: &'a mut http::HeaderMap,
}

impl Injector for HeaderMapInjector<'_> {
    fn set(&mut self, key: &str, value: String) {
        if let (Ok(name), Ok(val)) = (
            http::HeaderName::from_bytes(key.as_bytes()),
            http::HeaderValue::from_str(&value),
        ) {
            self.headers.insert(name, val);
        }
    }
}
