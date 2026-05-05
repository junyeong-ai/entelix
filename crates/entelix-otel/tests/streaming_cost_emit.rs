//! G-1 / invariant 12 regression — streaming dispatch flows through
//! the same `tower::Service` spine as one-shot, and `OtelLayer` 's
//! cost emission gates on the streaming-side completion future's
//! `Ok` branch only. A stream that errors mid-flight resolves
//! `completion` to `Err` and emits no `gen_ai.usage.cost` attribute;
//! a stream that completes cleanly resolves `Ok(ModelResponse)` and
//! emits the cost the cost calculator computed.
//!
//! The test drives a stub `Service<StreamingModelInvocation, Response
//! = ModelStream>` through `OtelLayer` and observes events captured
//! by a `tracing::subscriber` so the assertion is on the actual
//! emitted attributes, not on a derived metric.

// `option_if_let_else` would force the substantial 2-branch
// `match inject_at` body into a `map_or_else` chain that hurts
// readability — the branches build distinct delta sequences and
// belong as readable per-arm code, not as closures. Opted out by
// intent.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::option_if_let_else)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll};

use async_trait::async_trait;
use entelix_core::context::ExecutionContext;
use entelix_core::cost::CostCalculator;
use entelix_core::error::Error;
use entelix_core::ir::{ContentPart, Message, ModelRequest, Role, StopReason, Usage};
use entelix_core::service::{ModelInvocation, ModelStream, StreamingModelInvocation};
use entelix_core::stream::{StreamDelta, tap_aggregator};
use entelix_otel::OtelLayer;
use futures::StreamExt;
use parking_lot::Mutex;
use tower::{Layer, Service, ServiceExt};
use tracing::Subscriber;
use tracing::field::Visit;
use tracing::span::{Attributes, Id, Record};
use tracing_subscriber::layer::{Context as LayerCtx, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;

/// Stub leaf service that emits a fixed delta sequence terminating
/// in `Stop` (or in mid-stream `Err`, controlled by the
/// `inject_error_at` field). One-shot path is irrelevant — only
/// the `Service<StreamingModelInvocation, Response = ModelStream>`
/// impl exists.
#[derive(Clone)]
struct StubStreamingService {
    /// Inject a mid-stream `Err` after this many `Ok` deltas.
    /// `None` ⇒ the stream completes cleanly with `Stop`.
    inject_error_at: Option<usize>,
}

impl Service<StreamingModelInvocation> for StubStreamingService {
    type Response = ModelStream;
    type Error = Error;
    type Future = futures::future::BoxFuture<'static, Result<ModelStream, Error>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, _invocation: StreamingModelInvocation) -> Self::Future {
        let inject_at = self.inject_error_at;
        Box::pin(async move {
            let deltas: Vec<Result<StreamDelta, Error>> = match inject_at {
                None => vec![
                    Ok(StreamDelta::Start {
                        id: "msg_01".into(),
                        model: "claude-opus-4-7".into(),
                    }),
                    Ok(StreamDelta::TextDelta {
                        text: "hello".into(),
                    }),
                    Ok(StreamDelta::Usage(Usage::new(100, 50))),
                    Ok(StreamDelta::Stop {
                        stop_reason: StopReason::EndTurn,
                    }),
                ],
                Some(n) => {
                    let mut v: Vec<Result<StreamDelta, Error>> = vec![
                        Ok(StreamDelta::Start {
                            id: "msg_01".into(),
                            model: "claude-opus-4-7".into(),
                        }),
                        Ok(StreamDelta::TextDelta {
                            text: "partial".into(),
                        }),
                    ];
                    v.truncate(n);
                    v.push(Err(Error::provider_network("connection reset mid-stream")));
                    v
                }
            };
            let raw = futures::stream::iter(deltas);
            Ok(tap_aggregator(Box::pin(raw)))
        })
    }
}

/// Minimal `CostCalculator` returning a fixed cost so the test can
/// assert the emitted `gen_ai.usage.cost` attribute matches what
/// the calculator produced.
struct FixedCost {
    /// Number of times `compute_cost` was invoked. The test asserts
    /// that the error path leaves this at zero (invariant 12 — no
    /// charge on the error branch).
    calls: Arc<AtomicUsize>,
    /// Cost the calculator returns on Ok-branch invocations.
    cost: f64,
}

#[async_trait]
impl CostCalculator for FixedCost {
    async fn compute_cost(
        &self,
        _model: &str,
        _usage: &Usage,
        _ctx: &ExecutionContext,
    ) -> Option<f64> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Some(self.cost)
    }
}

/// `tracing` subscriber layer that records every event's
/// `gen_ai.usage.cost` field value plus the event name. The test
/// reads back the recorded snapshots after the dispatch.
#[derive(Default, Clone)]
struct CostCapture {
    events: Arc<Mutex<Vec<CapturedEvent>>>,
}

#[derive(Debug, Clone)]
struct CapturedEvent {
    name: &'static str,
    cost: Option<f64>,
}

struct CostVisitor<'a> {
    message: &'a mut Option<String>,
    cost: &'a mut Option<f64>,
}

impl Visit for CostVisitor<'_> {
    fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
        if field.name() == "gen_ai.usage.cost" {
            *self.cost = Some(value);
        }
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "message" {
            *self.message = Some(value.to_owned());
        }
    }

    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            *self.message = Some(format!("{value:?}").trim_matches('"').to_owned());
        }
    }
}

impl<S: Subscriber + for<'lookup> LookupSpan<'lookup>> tracing_subscriber::Layer<S>
    for CostCapture
{
    fn on_new_span(&self, _attrs: &Attributes<'_>, _id: &Id, _ctx: LayerCtx<'_, S>) {}
    fn on_record(&self, _id: &Id, _values: &Record<'_>, _ctx: LayerCtx<'_, S>) {}
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: LayerCtx<'_, S>) {
        let mut message: Option<String> = None;
        let mut cost: Option<f64> = None;
        let mut visitor = CostVisitor {
            message: &mut message,
            cost: &mut cost,
        };
        event.record(&mut visitor);
        let Some(msg) = message else { return };
        let name: &'static str = if msg.contains("gen_ai.response") {
            "gen_ai.response"
        } else if msg.contains("gen_ai.error") {
            "gen_ai.error"
        } else if msg.contains("gen_ai.request") {
            "gen_ai.request"
        } else {
            return;
        };
        self.events.lock().push(CapturedEvent { name, cost });
    }
}

#[tokio::test]
async fn streaming_ok_branch_emits_cost_attribute() {
    let capture = CostCapture::default();
    let calls = Arc::new(AtomicUsize::new(0));
    let cost_calc: Arc<dyn CostCalculator> = Arc::new(FixedCost {
        calls: Arc::clone(&calls),
        cost: 0.0042,
    });

    let subscriber = tracing_subscriber::registry().with(capture.clone());
    let _guard = tracing::subscriber::set_default(subscriber);

    let layer = OtelLayer::new("test").with_cost_calculator(Arc::clone(&cost_calc));
    let stub = StubStreamingService {
        inject_error_at: None,
    };
    let svc = layer.layer(stub);

    let request = ModelRequest {
        model: "claude-opus-4-7".into(),
        messages: vec![Message::new(
            Role::User,
            vec![ContentPart::Text {
                text: "hi".into(),
                cache_control: None,
            }],
        )],
        max_tokens: Some(100),
        ..Default::default()
    };
    let invocation =
        StreamingModelInvocation::new(ModelInvocation::new(request, ExecutionContext::new()));

    let model_stream = svc.oneshot(invocation).await.expect("dispatch ok");
    // Drain the delta stream — that triggers `tap_aggregator` to
    // resolve `completion`, which the OtelLayer wrap then awaits to
    // emit the cost.
    let mut stream = model_stream.stream;
    while stream.next().await.is_some() {}
    // Await completion so the OtelLayer's wrap runs to its
    // emission point — this is the documented contract for callers
    // that want the post-stream observability to fire.
    let response = model_stream
        .completion
        .await
        .expect("completion resolves Ok");

    assert_eq!(response.model, "claude-opus-4-7");
    assert_eq!(response.usage.input_tokens, 100);
    assert_eq!(response.usage.output_tokens, 50);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "cost calculator should fire exactly once on the Ok branch"
    );

    let events = capture.events.lock().clone();
    let response_event = events
        .iter()
        .find(|e| e.name == "gen_ai.response")
        .expect("gen_ai.response event recorded");
    assert!(
        (response_event.cost.expect("cost attribute present") - 0.0042).abs() < f64::EPSILON,
        "expected cost=0.0042, got {:?}",
        response_event.cost
    );
    assert!(
        !events.iter().any(|e| e.name == "gen_ai.error"),
        "Ok-branch streaming dispatch must not emit gen_ai.error"
    );
}

#[tokio::test]
async fn streaming_mid_stream_error_emits_no_cost() {
    let capture = CostCapture::default();
    let calls = Arc::new(AtomicUsize::new(0));
    let cost_calc: Arc<dyn CostCalculator> = Arc::new(FixedCost {
        calls: Arc::clone(&calls),
        cost: 0.0042,
    });

    let subscriber = tracing_subscriber::registry().with(capture.clone());
    let _guard = tracing::subscriber::set_default(subscriber);

    let layer = OtelLayer::new("test").with_cost_calculator(Arc::clone(&cost_calc));
    // Stub injects an error after the second delta — the stream
    // never reaches terminal `Stop`, so `completion` resolves
    // `Err` and the OtelLayer wrap takes the error branch.
    let stub = StubStreamingService {
        inject_error_at: Some(2),
    };
    let svc = layer.layer(stub);

    let request = ModelRequest {
        model: "claude-opus-4-7".into(),
        max_tokens: Some(100),
        ..Default::default()
    };
    let invocation =
        StreamingModelInvocation::new(ModelInvocation::new(request, ExecutionContext::new()));

    let model_stream = svc.oneshot(invocation).await.expect("dispatch ok");
    let mut stream = model_stream.stream;
    while stream.next().await.is_some() {}
    // The completion resolves to Err — the consumer either matches
    // on it or ignores it. Awaiting is what fires the OtelLayer's
    // post-stream emission branch.
    let result = model_stream.completion.await;
    assert!(
        result.is_err(),
        "mid-stream error should resolve completion to Err"
    );

    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "cost calculator must not fire on the error branch (invariant 12)"
    );

    let events = capture.events.lock().clone();
    assert!(
        !events
            .iter()
            .any(|e| e.name == "gen_ai.response" && e.cost.is_some()),
        "no gen_ai.response event with cost attribute on the error branch"
    );
    assert!(
        events.iter().any(|e| e.name == "gen_ai.error"),
        "mid-stream error must surface as gen_ai.error event"
    );
}
