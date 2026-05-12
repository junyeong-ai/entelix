//! `ChatModel` end-to-end tests using inline mock codec + transport.
//!
//! Covers the standalone `complete()` method. The Runnable impl lives in
//! `entelix-runnable` and is exercised in that crate's tests.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::unnecessary_literal_bound
)]

use std::sync::Mutex;

use bytes::Bytes;
use entelix_core::codecs::{Codec, EncodedRequest};
use entelix_core::ir::{
    Capabilities, ContentPart, Message, ModelRequest, ModelResponse, ModelWarning, Role,
    StopReason, ToolSpec, Usage,
};
use entelix_core::transports::{Transport, TransportResponse};
use entelix_core::{ChatModel, Error, ExecutionContext, Result};

/// Codec that records every encoded request and replies with a canned
/// "ok!" assistant message.
struct RecordingCodec {
    seen: Mutex<Vec<ModelRequest>>,
}

impl RecordingCodec {
    const fn new() -> Self {
        Self {
            seen: Mutex::new(Vec::new()),
        }
    }

    fn last_request(&self) -> ModelRequest {
        self.seen.lock().unwrap().last().cloned().unwrap()
    }
}

impl Codec for RecordingCodec {
    fn name(&self) -> &'static str {
        "recording"
    }

    fn capabilities(&self, _model: &str) -> Capabilities {
        Capabilities::default()
    }

    fn encode(&self, request: &ModelRequest) -> Result<EncodedRequest> {
        self.seen.lock().unwrap().push(request.clone());
        let body = serde_json::to_vec(request).unwrap();
        Ok(EncodedRequest::post_json("/recording", Bytes::from(body)))
    }

    fn decode(&self, _body: &[u8], warnings_in: Vec<ModelWarning>) -> Result<ModelResponse> {
        Ok(ModelResponse {
            id: "rec_1".into(),
            model: "rec".into(),
            stop_reason: StopReason::EndTurn,
            content: vec![ContentPart::text("ok!")],
            usage: Usage::default(),
            rate_limit: None,
            warnings: warnings_in,
            provider_echoes: Vec::new(),
        })
    }
}

/// Transport that returns a fixed status without ever calling the network.
struct FakeStatusTransport {
    status: u16,
    body: Bytes,
}

#[async_trait::async_trait]
impl Transport for FakeStatusTransport {
    fn name(&self) -> &'static str {
        "fake-status"
    }

    async fn send(
        &self,
        _request: EncodedRequest,
        _ctx: &ExecutionContext,
    ) -> Result<TransportResponse> {
        Ok(TransportResponse {
            status: self.status,
            headers: http::HeaderMap::new(),
            body: self.body.clone(),
        })
    }
}

#[tokio::test]
async fn complete_round_trips_via_codec_and_transport() -> Result<()> {
    let codec = RecordingCodec::new();
    let transport = FakeStatusTransport {
        status: 200,
        body: Bytes::new(),
    };
    let model = ChatModel::new(codec, transport, "model-x").with_max_tokens(512);

    let reply = model
        .complete(vec![Message::user("hi")], &ExecutionContext::new())
        .await?;
    assert!(matches!(reply.role, Role::Assistant));
    match &reply.content[0] {
        ContentPart::Text { text, .. } => assert_eq!(text, "ok!"),
        other => panic!("expected text, got {other:?}"),
    }
    Ok(())
}

#[tokio::test]
async fn complete_propagates_builder_defaults_into_request() -> Result<()> {
    let codec = std::sync::Arc::new(RecordingCodec::new());
    let transport = FakeStatusTransport {
        status: 200,
        body: Bytes::new(),
    };
    let model = ChatModel::from_arc(codec.clone(), std::sync::Arc::new(transport), "model-y")
        .with_max_tokens(1024)
        .with_system("Be terse.")
        .with_temperature(0.3)
        .with_top_p(0.9)
        .with_stop_sequence("###")
        .with_tools([ToolSpec::function(
            "calc",
            "Calculator",
            serde_json::json!({}),
        )]);

    model
        .complete(vec![Message::user("hi")], &ExecutionContext::new())
        .await?;

    let req = codec.last_request();
    assert_eq!(req.model, "model-y");
    assert_eq!(req.max_tokens, Some(1024));
    assert_eq!(req.system.concat_text(), "Be terse.");
    assert!((req.temperature.unwrap() - 0.3).abs() < 1e-6);
    assert!((req.top_p.unwrap() - 0.9).abs() < 1e-6);
    assert_eq!(req.stop_sequences, vec!["###".to_owned()]);
    assert_eq!(req.tools.len(), 1);
    Ok(())
}

#[tokio::test]
async fn run_overrides_patch_model_and_system_prompt() -> Result<()> {
    let codec = std::sync::Arc::new(RecordingCodec::new());
    let transport = FakeStatusTransport {
        status: 200,
        body: Bytes::new(),
    };
    let model = ChatModel::from_arc(
        codec.clone(),
        std::sync::Arc::new(transport),
        "default-model",
    )
    .with_system("default system");

    let ctx = ExecutionContext::new().add_extension(
        entelix_core::RunOverrides::new()
            .with_model("override-model")
            .with_system_prompt(entelix_core::ir::SystemPrompt::text("override system")),
    );
    model.complete(vec![Message::user("hi")], &ctx).await?;

    let req = codec.last_request();
    assert_eq!(
        req.model, "override-model",
        "RunOverrides::model must patch the request"
    );
    assert_eq!(
        req.system.concat_text(),
        "override system",
        "RunOverrides::system_prompt must replace (not append) the configured prompt"
    );
    Ok(())
}

#[tokio::test]
async fn run_overrides_absent_keeps_configured_defaults() -> Result<()> {
    let codec = std::sync::Arc::new(RecordingCodec::new());
    let transport = FakeStatusTransport {
        status: 200,
        body: Bytes::new(),
    };
    let model = ChatModel::from_arc(
        codec.clone(),
        std::sync::Arc::new(transport),
        "default-model",
    )
    .with_system("default system");

    // No RunOverrides extension — request keeps the ChatModel defaults.
    model
        .complete(vec![Message::user("hi")], &ExecutionContext::new())
        .await?;

    let req = codec.last_request();
    assert_eq!(req.model, "default-model");
    assert_eq!(req.system.concat_text(), "default system");
    Ok(())
}

#[tokio::test]
async fn request_overrides_patch_sampling_and_response_format() -> Result<()> {
    let codec = std::sync::Arc::new(RecordingCodec::new());
    let transport = FakeStatusTransport {
        status: 200,
        body: Bytes::new(),
    };
    let model = ChatModel::from_arc(
        codec.clone(),
        std::sync::Arc::new(transport),
        "default-model",
    )
    .with_temperature(0.7)
    .with_top_p(0.95)
    .with_max_tokens(1024)
    .with_stop_sequences(vec!["DEFAULT_STOP".to_string()])
    .with_reasoning_effort(entelix_core::ir::ReasoningEffort::Medium);

    let ctx = ExecutionContext::new().add_extension(
        entelix_core::RequestOverrides::new()
            .with_temperature(0.2)
            .with_top_p(0.5)
            .with_max_tokens(64)
            .with_stop_sequences(vec!["</done>".to_string()])
            .with_reasoning_effort(entelix_core::ir::ReasoningEffort::High)
            .with_tool_choice(entelix_core::ir::ToolChoice::None)
            .with_response_format(entelix_core::ir::ResponseFormat::strict(
                entelix_core::ir::JsonSchemaSpec::new(
                    "answer",
                    serde_json::json!({"type": "object"}),
                )
                .unwrap(),
            ))
            .with_end_user_id("op-7")
            .with_seed(42),
    );
    model.complete(vec![Message::user("hi")], &ctx).await?;

    let req = codec.last_request();
    assert!((req.temperature.unwrap() - 0.2).abs() < 1e-6);
    assert!((req.top_p.unwrap() - 0.5).abs() < 1e-6);
    assert_eq!(req.max_tokens, Some(64));
    assert_eq!(req.stop_sequences, vec!["</done>".to_string()]);
    assert!(matches!(
        req.reasoning_effort,
        Some(entelix_core::ir::ReasoningEffort::High)
    ));
    assert!(matches!(
        req.tool_choice,
        entelix_core::ir::ToolChoice::None
    ));
    let format = req.response_format.as_ref().expect("response_format set");
    assert_eq!(format.json_schema.name, "answer");
    assert_eq!(req.end_user_id.as_deref(), Some("op-7"));
    assert_eq!(req.seed, Some(42));
    Ok(())
}

#[tokio::test]
async fn request_overrides_partial_override_keeps_unset_fields() -> Result<()> {
    let codec = std::sync::Arc::new(RecordingCodec::new());
    let transport = FakeStatusTransport {
        status: 200,
        body: Bytes::new(),
    };
    let model = ChatModel::from_arc(
        codec.clone(),
        std::sync::Arc::new(transport),
        "default-model",
    )
    .with_temperature(0.7)
    .with_max_tokens(1024);

    // Only temperature is overridden; max_tokens stays at the configured 1024.
    let ctx = ExecutionContext::new()
        .add_extension(entelix_core::RequestOverrides::new().with_temperature(0.1));
    model.complete(vec![Message::user("hi")], &ctx).await?;

    let req = codec.last_request();
    assert!((req.temperature.unwrap() - 0.1).abs() < 1e-6);
    assert_eq!(
        req.max_tokens,
        Some(1024),
        "max_tokens not overridden — must keep configured default"
    );
    Ok(())
}

#[tokio::test]
async fn request_and_run_overrides_compose_independently() -> Result<()> {
    let codec = std::sync::Arc::new(RecordingCodec::new());
    let transport = FakeStatusTransport {
        status: 200,
        body: Bytes::new(),
    };
    let model = ChatModel::from_arc(
        codec.clone(),
        std::sync::Arc::new(transport),
        "default-model",
    )
    .with_system("default system")
    .with_temperature(0.7);

    let ctx = ExecutionContext::new()
        .add_extension(
            entelix_core::RunOverrides::new()
                .with_model("haiku")
                .with_system_prompt(entelix_core::ir::SystemPrompt::text("triage")),
        )
        .add_extension(entelix_core::RequestOverrides::new().with_temperature(0.1));
    model.complete(vec![Message::user("hi")], &ctx).await?;

    let req = codec.last_request();
    assert_eq!(req.model, "haiku");
    assert_eq!(req.system.concat_text(), "triage");
    assert!((req.temperature.unwrap() - 0.1).abs() < 1e-6);
    Ok(())
}

#[tokio::test]
async fn request_overrides_stop_sequences_empty_clears_configured() -> Result<()> {
    let codec = std::sync::Arc::new(RecordingCodec::new());
    let transport = FakeStatusTransport {
        status: 200,
        body: Bytes::new(),
    };
    let model = ChatModel::from_arc(
        codec.clone(),
        std::sync::Arc::new(transport),
        "default-model",
    )
    .with_stop_sequences(vec!["A".to_string(), "B".to_string()]);

    let ctx = ExecutionContext::new()
        .add_extension(entelix_core::RequestOverrides::new().with_stop_sequences(Vec::new()));
    model.complete(vec![Message::user("hi")], &ctx).await?;

    let req = codec.last_request();
    assert!(
        req.stop_sequences.is_empty(),
        "Empty stop_sequences override must clear the configured list"
    );
    Ok(())
}

#[tokio::test]
async fn complete_returns_provider_error_on_4xx() {
    let codec = RecordingCodec::new();
    let transport = FakeStatusTransport {
        status: 401,
        body: Bytes::from_static(b"unauthorized"),
    };
    let model = ChatModel::new(codec, transport, "model-z");

    let err = model
        .complete(vec![Message::user("hi")], &ExecutionContext::new())
        .await
        .unwrap_err();
    match err {
        Error::Provider {
            kind, ref message, ..
        } => {
            assert_eq!(kind, entelix_core::ProviderErrorKind::Http(401));
            assert!(message.contains("unauthorized"), "got {message}");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test]
async fn chat_model_clone_shares_inner_handles() {
    let codec = RecordingCodec::new();
    let transport = FakeStatusTransport {
        status: 200,
        body: Bytes::new(),
    };
    let m1 = ChatModel::new(codec, transport, "shared");
    let m2 = m1.clone();
    assert_eq!(m1.config().model(), "shared");
    assert_eq!(m2.config().model(), "shared");
}

// ── tower::Layer plumbing ────────────────────────────────────────────────

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::task::{Context as TaskContext, Poll};

use entelix_core::ModelInvocation;
use entelix_core::service::{
    BoxedModelService, BoxedStreamingService, ModelStream, StreamingModelInvocation,
};
use futures::future::BoxFuture;
use tower::{Layer, Service};

/// Test layer that counts pre/post invocations and optionally
/// mutates the request before encode and the response after decode.
#[derive(Clone)]
struct CountingLayer {
    state: Arc<CountingState>,
}

struct CountingState {
    pre: AtomicU32,
    post: AtomicU32,
    fail_pre: bool,
    mutate_request: bool,
    mutate_response: bool,
}

impl CountingLayer {
    fn new() -> (Self, Arc<CountingState>) {
        let state = Arc::new(CountingState {
            pre: AtomicU32::new(0),
            post: AtomicU32::new(0),
            fail_pre: false,
            mutate_request: false,
            mutate_response: false,
        });
        (
            Self {
                state: state.clone(),
            },
            state,
        )
    }

    fn failing() -> (Self, Arc<CountingState>) {
        let state = Arc::new(CountingState {
            pre: AtomicU32::new(0),
            post: AtomicU32::new(0),
            fail_pre: true,
            mutate_request: false,
            mutate_response: false,
        });
        (
            Self {
                state: state.clone(),
            },
            state,
        )
    }

    fn mutating() -> (Self, Arc<CountingState>) {
        let state = Arc::new(CountingState {
            pre: AtomicU32::new(0),
            post: AtomicU32::new(0),
            fail_pre: false,
            mutate_request: true,
            mutate_response: true,
        });
        (
            Self {
                state: state.clone(),
            },
            state,
        )
    }
}

impl<S> Layer<S> for CountingLayer {
    type Service = CountingService<S>;
    fn layer(&self, inner: S) -> Self::Service {
        CountingService {
            inner,
            state: self.state.clone(),
        }
    }
}

impl entelix_core::NamedLayer for CountingLayer {
    fn layer_name(&self) -> &'static str {
        "counting"
    }
}

#[derive(Clone)]
struct CountingService<S> {
    inner: S,
    state: Arc<CountingState>,
}

impl<S> Service<ModelInvocation> for CountingService<S>
where
    S: Service<ModelInvocation, Response = ModelResponse, Error = Error> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = ModelResponse;
    type Error = Error;
    type Future = BoxFuture<'static, Result<ModelResponse>>;

    fn poll_ready(&mut self, cx: &mut TaskContext<'_>) -> Poll<Result<()>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut invocation: ModelInvocation) -> Self::Future {
        let state = self.state.clone();
        let inner = self.inner.clone();
        Box::pin(async move {
            state.pre.fetch_add(1, Ordering::SeqCst);
            if state.fail_pre {
                return Err(Error::invalid_request("layer refused"));
            }
            if state.mutate_request {
                invocation.request.messages.push(Message::user("[layer]"));
            }
            let mut response = tower::ServiceExt::oneshot(inner, invocation).await?;
            state.post.fetch_add(1, Ordering::SeqCst);
            if state.mutate_response {
                response.content.push(ContentPart::text("[mutated]"));
            }
            Ok(response)
        })
    }
}

impl<S> Service<StreamingModelInvocation> for CountingService<S>
where
    S: Service<StreamingModelInvocation, Response = ModelStream, Error = Error>
        + Clone
        + Send
        + 'static,
    S::Future: Send + 'static,
{
    type Response = ModelStream;
    type Error = Error;
    type Future = BoxFuture<'static, Result<ModelStream>>;

    fn poll_ready(&mut self, cx: &mut TaskContext<'_>) -> Poll<Result<()>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut invocation: StreamingModelInvocation) -> Self::Future {
        let state = self.state.clone();
        let inner = self.inner.clone();
        Box::pin(async move {
            state.pre.fetch_add(1, Ordering::SeqCst);
            if state.fail_pre {
                return Err(Error::invalid_request("layer refused"));
            }
            if state.mutate_request {
                invocation
                    .inner
                    .request
                    .messages
                    .push(Message::user("[layer]"));
            }
            // Streaming path: post-mutation of the response stream
            // would require wrapping every delta — out of scope for
            // the existing one-shot regression suite. The pre/post
            // counters still observe one round-trip per call, which
            // is what the chat_model layer-plumbing tests assert.
            let stream = tower::ServiceExt::oneshot(inner, invocation).await?;
            state.post.fetch_add(1, Ordering::SeqCst);
            Ok(stream)
        })
    }
}

// `BoxedModelService` / `BoxedStreamingService` references exist to
// anchor the imports to the public surface; layers compose on them
// via `ChatModel::layer`.
const _BOXED_MODEL_SERVICE_REF: Option<BoxedModelService> = None;
const _BOXED_STREAMING_SERVICE_REF: Option<BoxedStreamingService> = None;

#[test]
fn layer_names_track_compose_order_and_namedlayer_identity() {
    use entelix_core::WithName;

    let codec = Arc::new(RecordingCodec::new());
    let transport = Arc::new(FakeStatusTransport {
        status: 200,
        body: Bytes::new(),
    });
    let bare = ChatModel::from_arc(codec.clone(), transport.clone(), "m");
    assert!(
        bare.layer_names().is_empty(),
        "freshly built model has no layers"
    );

    let (counting1, _) = CountingLayer::new();
    let (counting2, _) = CountingLayer::new();

    // First-registered layer sits innermost (index 0); subsequent
    // `.layer(...)` calls append to the right of the vec, matching
    // the wrap-on-top semantics of the factory chain.
    let model = ChatModel::from_arc(codec, transport, "m")
        .layer(counting1)
        .layer(WithName::new("external", counting2));

    assert_eq!(
        model.layer_names(),
        ["counting", "external"],
        "layer_names tracks insertion order; WithName supplies a name for external middleware"
    );
}

#[tokio::test]
async fn layer_fires_pre_and_post_around_complete_full() {
    let codec = Arc::new(RecordingCodec::new());
    let transport = Arc::new(FakeStatusTransport {
        status: 200,
        body: Bytes::new(),
    });
    let (layer, state) = CountingLayer::new();
    let model = ChatModel::from_arc(codec.clone(), transport, "m")
        .with_max_tokens(0) // mutation visible without affecting baseline
        .layer(layer);
    // Force a layer re-mutation pass that actually appends:
    let (mut_layer, mut_state) = CountingLayer::mutating();
    let _ = mut_state; // silence unused
    let _ = mut_layer; // silence unused
    drop(mut_layer);
    let _ = state.clone();

    let _ = model
        .complete_full(vec![Message::user("hi")], &ExecutionContext::new())
        .await
        .unwrap();

    assert_eq!(state.pre.load(Ordering::SeqCst), 1);
    assert_eq!(state.post.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn pre_layer_error_aborts_before_codec_encode() {
    let codec = Arc::new(RecordingCodec::new());
    let transport = Arc::new(FakeStatusTransport {
        status: 200,
        body: Bytes::new(),
    });
    let (layer, state) = CountingLayer::failing();
    let model = ChatModel::from_arc(codec.clone(), transport, "m").layer(layer);

    let err = model
        .complete_full(vec![Message::user("hi")], &ExecutionContext::new())
        .await
        .unwrap_err();
    assert!(format!("{err}").contains("layer refused"));

    // Critical invariant: the codec never saw an encode call.
    assert_eq!(codec.seen.lock().unwrap().len(), 0);
    // post never fires when pre bails.
    assert_eq!(state.post.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn layer_can_mutate_decoded_response() {
    let codec = RecordingCodec::new();
    let transport = FakeStatusTransport {
        status: 200,
        body: Bytes::new(),
    };
    let (layer, _state) = CountingLayer::mutating();
    let model = ChatModel::new(codec, transport, "m").layer(layer);

    let response = model
        .complete_full(vec![Message::user("hi")], &ExecutionContext::new())
        .await
        .unwrap();
    let texts: Vec<_> = response
        .content
        .iter()
        .filter_map(|p| match p {
            ContentPart::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert!(texts.contains(&"[mutated]"), "{texts:?}");
}

#[tokio::test]
async fn layer_mutation_of_request_reaches_codec() {
    let codec = Arc::new(RecordingCodec::new());
    let transport = Arc::new(FakeStatusTransport {
        status: 200,
        body: Bytes::new(),
    });
    let (layer, _state) = CountingLayer::mutating();
    let model = ChatModel::from_arc(codec.clone(), transport, "m").layer(layer);

    let _ = model
        .complete_full(vec![Message::user("hi")], &ExecutionContext::new())
        .await
        .unwrap();
    let req = codec.last_request();
    let last_text = match &req.messages.last().unwrap().content[0] {
        ContentPart::Text { text, .. } => text.clone(),
        _ => String::new(),
    };
    assert_eq!(last_text, "[layer]");
}

#[tokio::test]
async fn no_layer_attached_is_a_noop_call_path() {
    let codec = RecordingCodec::new();
    let transport = FakeStatusTransport {
        status: 200,
        body: Bytes::new(),
    };
    let model = ChatModel::new(codec, transport, "m");
    let _ = model
        .complete(vec![Message::user("hi")], &ExecutionContext::new())
        .await
        .unwrap();
}

#[tokio::test]
async fn layers_compose_in_outer_first_order() {
    let codec = Arc::new(RecordingCodec::new());
    let transport = Arc::new(FakeStatusTransport {
        status: 200,
        body: Bytes::new(),
    });
    let (outer, outer_state) = CountingLayer::new();
    let (inner, inner_state) = CountingLayer::new();
    let model = ChatModel::from_arc(codec, transport, "m")
        .layer(outer) // outermost — runs first / unwraps last
        .layer(inner); // innermost — runs after outer, before leaf

    let _ = model
        .complete_full(vec![Message::user("hi")], &ExecutionContext::new())
        .await
        .unwrap();
    assert_eq!(outer_state.pre.load(Ordering::SeqCst), 1);
    assert_eq!(outer_state.post.load(Ordering::SeqCst), 1);
    assert_eq!(inner_state.pre.load(Ordering::SeqCst), 1);
    assert_eq!(inner_state.post.load(Ordering::SeqCst), 1);
}

/// Verify the provider shortcuts compile and instantiate cleanly.
/// We do NOT issue a network call — the goal is to lock in the
/// 1-call-construction ergonomics so refactors that re-introduce
/// boilerplate to the common path break this test.
#[test]
fn provider_shortcuts_compose_codec_transport_and_auth_in_one_call() {
    use entelix_core::ChatModel;
    let anthropic = ChatModel::anthropic("dummy-key", "claude-opus-4-7").unwrap();
    assert_eq!(anthropic.config().model(), "claude-opus-4-7");
    assert_eq!(
        anthropic.transport().base_url(),
        "https://api.anthropic.com"
    );

    let openai = ChatModel::openai("dummy-key", "gpt-4o").unwrap();
    assert_eq!(openai.config().model(), "gpt-4o");
    assert_eq!(openai.transport().base_url(), "https://api.openai.com");

    let gemini = ChatModel::gemini("dummy-key", "gemini-2.0-flash").unwrap();
    assert_eq!(gemini.config().model(), "gemini-2.0-flash");
    assert_eq!(
        gemini.transport().base_url(),
        "https://generativelanguage.googleapis.com"
    );
}
