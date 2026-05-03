//! `ChatModel` — composes a `Codec` and a `Transport` into a layered
//! `tower::Service<ModelInvocation, Response = ModelResponse>` plus
//! a streaming surface.
//!
//! The composition is two-tiered:
//!
//! - An internal leaf service performs the raw
//!   `codec.encode → transport.send → codec.decode` round trip,
//!   implementing `tower::Service<ModelInvocation>`.
//! - [`ChatModel<C, T>`] is the user-facing builder. It owns the
//!   leaf inner service and a layer stack assembled via
//!   [`ChatModel::layer`]. The composed `tower::Service` materialises
//!   on each [`ChatModel::complete_full`] call.
//!
//! Cross-cutting concerns (PII redaction, quota gates, cost
//! metering, OTel observability) live as `tower::Layer<S>` types in
//! sibling crates (`entelix-policy::PolicyLayer`,
//! `entelix-otel::OtelLayer`). Same layer wraps both this surface
//! and [`crate::tools::ToolRegistry`] dispatch — one composition
//! primitive across the whole agent stack.

use std::sync::Arc;
use std::task::{Context, Poll};

use futures::StreamExt;
use futures::future::BoxFuture;
use tower::util::BoxCloneService;
use tower::{Layer, Service, ServiceExt};

use crate::codecs::{BoxDeltaStream, Codec};
use crate::context::ExecutionContext;
use crate::error::{Error, Result};
use crate::ir::{Message, ModelRequest, ModelResponse, Role, SystemPrompt, ToolChoice, ToolSpec};
use crate::overrides::RunOverrides;
use crate::service::{BoxedModelService, ModelInvocation};
use crate::stream::StreamDelta;
use crate::transports::Transport;

/// Patch `request` with any [`RunOverrides`] attached to `ctx`. Both
/// `complete_full` and `stream_deltas` route through this helper so
/// the override semantics stay identical across the two surfaces.
fn apply_run_overrides(request: &mut ModelRequest, ctx: &ExecutionContext) {
    let Some(overrides) = ctx.extension::<RunOverrides>() else {
        return;
    };
    if let Some(model) = overrides.model() {
        model.clone_into(&mut request.model);
    }
    if let Some(system) = overrides.system_prompt() {
        request.system = system.clone();
    }
}

/// Builder-side configuration that flows into every `ModelRequest`
/// the `ChatModel` issues. Stored separately from the leaf service
/// so layers / streaming / future surfaces share one source of
/// truth.
///
/// Fields are private. Construct via [`ChatModelConfig::new`];
/// mutate through [`ChatModel`]'s `with_*` setters; inspect through
/// the bare accessors on this type. `#[non_exhaustive]` plus the
/// privatised fields together mean post-1.0 additions ship as MINOR
/// without surprising any downstream consumer.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ChatModelConfig {
    model: String,
    max_tokens: Option<u32>,
    system: SystemPrompt,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stop_sequences: Vec<String>,
    tools: Vec<ToolSpec>,
    tool_choice: ToolChoice,
}

impl ChatModelConfig {
    /// Build a fresh config seeded with `model` and otherwise
    /// defaulted.
    #[must_use]
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            max_tokens: None,
            system: SystemPrompt::default(),
            temperature: None,
            top_p: None,
            stop_sequences: Vec::new(),
            tools: Vec::new(),
            tool_choice: ToolChoice::default(),
        }
    }

    /// Provider model identifier sent on the wire.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Per-call `max_tokens` cap (`None` = vendor default).
    pub const fn max_tokens(&self) -> Option<u32> {
        self.max_tokens
    }

    /// System-prompt blocks prepended to every call. Supports
    /// per-block prompt caching (Anthropic / Bedrock Converse).
    pub const fn system(&self) -> &SystemPrompt {
        &self.system
    }

    /// Sampling temperature.
    pub const fn temperature(&self) -> Option<f32> {
        self.temperature
    }

    /// Nucleus-sampling parameter.
    pub const fn top_p(&self) -> Option<f32> {
        self.top_p
    }

    /// Stop sequences.
    pub fn stop_sequences(&self) -> &[String] {
        &self.stop_sequences
    }

    /// Advertised tools.
    pub fn tools(&self) -> &[ToolSpec] {
        &self.tools
    }

    /// Tool-choice mode.
    pub const fn tool_choice(&self) -> &ToolChoice {
        &self.tool_choice
    }

    /// Combine config with caller-supplied messages into a full
    /// [`ModelRequest`].
    #[must_use]
    pub fn build_request(&self, messages: Vec<Message>) -> ModelRequest {
        ModelRequest {
            model: self.model.clone(),
            messages,
            system: self.system.clone(),
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            stop_sequences: self.stop_sequences.clone(),
            tools: self.tools.clone(),
            tool_choice: self.tool_choice.clone(),
            response_format: None,
            cache_key: None,
            cached_content: None,
            provider_extensions: crate::ir::ProviderExtensions::default(),
        }
    }
}

/// Leaf service: raw `codec.encode → transport.send → codec.decode`.
/// `tower::Service<ModelInvocation, Response = ModelResponse>`.
/// Cloning is cheap.
///
/// Internal composition primitive — users compose at the
/// [`ChatModel`] level. Exposing the leaf service publicly would
/// invite callers to bypass the layer stack and miss
/// observability / policy / cost middleware.
pub(crate) struct InnerChatModel<C: Codec, T: Transport> {
    codec: Arc<C>,
    transport: Arc<T>,
}

impl<C: Codec, T: Transport> Clone for InnerChatModel<C, T> {
    fn clone(&self) -> Self {
        Self {
            codec: Arc::clone(&self.codec),
            transport: Arc::clone(&self.transport),
        }
    }
}

impl<C: Codec, T: Transport> InnerChatModel<C, T> {
    /// Wrap shared `Arc`s.
    pub(crate) const fn from_arc(codec: Arc<C>, transport: Arc<T>) -> Self {
        Self { codec, transport }
    }

    /// Borrow the codec.
    pub(crate) fn codec(&self) -> &C {
        &self.codec
    }

    /// Borrow the transport.
    pub(crate) fn transport(&self) -> &T {
        &self.transport
    }
}

impl<C: Codec + 'static, T: Transport + 'static> Service<ModelInvocation> for InnerChatModel<C, T> {
    type Response = ModelResponse;
    type Error = Error;
    type Future = BoxFuture<'static, Result<ModelResponse>>;

    #[inline]
    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, invocation: ModelInvocation) -> Self::Future {
        let codec = Arc::clone(&self.codec);
        let transport = Arc::clone(&self.transport);
        Box::pin(async move {
            let ModelInvocation { request, ctx } = invocation;
            let encoded = codec.encode(&request)?;
            let warnings = encoded.warnings.clone();
            let response = transport.send(encoded, &ctx).await?;
            if !(200..300).contains(&response.status) {
                let body_text = String::from_utf8_lossy(&response.body).into_owned();
                let mut err = Error::provider_http(response.status, body_text);
                if let Some(after) =
                    crate::transports::parse_retry_after(response.headers.get("retry-after"))
                {
                    err = err.with_retry_after(after);
                }
                return Err(err);
            }
            let rate_limit = codec.extract_rate_limit(&response.headers);
            let mut decoded = codec.decode(&response.body, warnings)?;
            decoded.rate_limit = rate_limit;
            Ok(decoded)
        })
    }
}

/// Boxed factory: `InnerChatModel` → layered [`BoxedModelService`].
/// Stored on `ChatModel` so the type of the (potentially deeply
/// nested) layer stack stays opaque to users; one concrete
/// `ChatModel<C, T>` shape regardless of how many layers are
/// attached.
type LayerFactory<C, T> = Arc<dyn Fn(InnerChatModel<C, T>) -> BoxedModelService + Send + Sync>;

/// Configurable chat model — codec + transport + layer stack.
///
/// Cheap to clone (handles are `Arc`-backed). Builder-style
/// configuration via `with_*` methods; layer attachment via
/// [`Self::layer`].
pub struct ChatModel<C: Codec + 'static, T: Transport + 'static> {
    inner: InnerChatModel<C, T>,
    config: ChatModelConfig,
    factory: Option<LayerFactory<C, T>>,
}

impl<C: Codec + 'static, T: Transport + 'static> Clone for ChatModel<C, T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            config: self.config.clone(),
            factory: self.factory.clone(),
        }
    }
}

impl<C: Codec + 'static, T: Transport + 'static> ChatModel<C, T> {
    /// Build a model bundling owned codec + transport + model name.
    pub fn new(codec: C, transport: T, model: impl Into<String>) -> Self {
        Self::from_arc(Arc::new(codec), Arc::new(transport), model)
    }

    /// Build from already-shared `Arc`s — useful when many models
    /// share one transport / codec instance.
    pub fn from_arc(codec: Arc<C>, transport: Arc<T>, model: impl Into<String>) -> Self {
        Self {
            inner: InnerChatModel::from_arc(codec, transport),
            config: ChatModelConfig::new(model),
            factory: None,
        }
    }

    /// Set per-call `max_tokens`.
    #[must_use]
    pub const fn with_max_tokens(mut self, n: u32) -> Self {
        self.config.max_tokens = Some(n);
        self
    }

    /// Attach a system / instruction prompt. Convenience
    /// shorthand for a single-block uncached prompt — for
    /// multi-block or cached prompts, set `config.system` to a
    /// pre-built [`SystemPrompt`] directly.
    #[must_use]
    pub fn with_system(mut self, s: impl Into<String>) -> Self {
        self.config.system = SystemPrompt::text(s);
        self
    }

    /// Attach a pre-built [`SystemPrompt`] — supports multi-block
    /// and per-block cache control.
    #[must_use]
    pub fn with_system_prompt(mut self, prompt: SystemPrompt) -> Self {
        self.config.system = prompt;
        self
    }

    /// Set sampling temperature.
    #[must_use]
    pub const fn with_temperature(mut self, t: f32) -> Self {
        self.config.temperature = Some(t);
        self
    }

    /// Set nucleus sampling parameter.
    #[must_use]
    pub const fn with_top_p(mut self, p: f32) -> Self {
        self.config.top_p = Some(p);
        self
    }

    /// Append one stop sequence.
    #[must_use]
    pub fn with_stop_sequence(mut self, s: impl Into<String>) -> Self {
        self.config.stop_sequences.push(s.into());
        self
    }

    /// Replace the full stop sequences list.
    #[must_use]
    pub fn with_stop_sequences(mut self, seqs: Vec<String>) -> Self {
        self.config.stop_sequences = seqs;
        self
    }

    /// Append a tool to the advertised set.
    #[must_use]
    pub fn with_tool(mut self, t: ToolSpec) -> Self {
        self.config.tools.push(t);
        self
    }

    /// Replace the full tools list.
    #[must_use]
    pub fn with_tools(mut self, tools: Vec<ToolSpec>) -> Self {
        self.config.tools = tools;
        self
    }

    /// Set the tool-choice mode.
    #[must_use]
    pub fn with_tool_choice(mut self, c: ToolChoice) -> Self {
        self.config.tool_choice = c;
        self
    }

    /// Append a `tower::Layer` to the dispatch stack. Layers run in
    /// registration order around the internal leaf service — the
    /// first-registered layer is *outermost* (sees the request
    /// first, the response last).
    ///
    /// The layer must produce a service that handles
    /// `ModelInvocation → ModelResponse`. `PolicyLayer` and
    /// `OtelLayer` from the policy / otel crates satisfy this for
    /// both `ModelInvocation` and `ToolInvocation` shapes — the
    /// same struct stacks on both `ChatModel::layer` and
    /// [`crate::tools::ToolRegistry::layer`].
    #[must_use]
    pub fn layer<L>(mut self, layer: L) -> Self
    where
        L: Layer<BoxedModelService> + Clone + Send + Sync + 'static,
        L::Service: Service<ModelInvocation, Response = ModelResponse, Error = Error>
            + Clone
            + Send
            + 'static,
        <L::Service as Service<ModelInvocation>>::Future: Send + 'static,
    {
        let prev = self.factory.take();
        let layer = layer;
        let new_factory: LayerFactory<C, T> = Arc::new(move |inner: InnerChatModel<C, T>| {
            let stacked: BoxedModelService = match &prev {
                Some(prev_factory) => prev_factory(inner),
                None => BoxCloneService::new(inner),
            };
            BoxCloneService::new(layer.clone().layer(stacked))
        });
        self.factory = Some(new_factory);
        self
    }

    /// Borrow the configured codec — exposes its `name()` and
    /// `capabilities()` for diagnostics.
    pub fn codec(&self) -> &C {
        self.inner.codec()
    }

    /// Borrow the configured transport.
    pub fn transport(&self) -> &T {
        self.inner.transport()
    }

    /// Borrow the configured request shape — `model()`,
    /// `max_tokens()`, `system()`, `temperature()`, `top_p()`,
    /// `stop_sequences()`, `tools()`, `tool_choice()` accessors all
    /// live on [`ChatModelConfig`].
    pub const fn config(&self) -> &ChatModelConfig {
        &self.config
    }

    /// Build the layered [`BoxedModelService`] — used by callers
    /// who want to drive the service directly (e.g. wrap with
    /// further `tower::ServiceBuilder` middleware externally).
    #[must_use]
    pub fn service(&self) -> BoxedModelService {
        match &self.factory {
            Some(factory) => factory(self.inner.clone()),
            None => BoxCloneService::new(self.inner.clone()),
        }
    }

    /// Send a conversation and return the assistant reply as a single
    /// [`Message`]. The full pipeline routes through the layer
    /// stack: each layer's pre-call work runs before encode, each
    /// layer's post-call work runs after decode.
    pub async fn complete(
        &self,
        messages: Vec<Message>,
        ctx: &ExecutionContext,
    ) -> Result<Message> {
        let response = self.complete_full(messages, ctx).await?;
        Ok(Message::new(Role::Assistant, response.content))
    }

    /// Same pipeline as [`Self::complete`], but returns the full
    /// [`ModelResponse`] — usage, stop reason, codec warnings, and
    /// the provider rate-limit snapshot when the codec could parse
    /// one from the response headers.
    pub async fn complete_full(
        &self,
        messages: Vec<Message>,
        ctx: &ExecutionContext,
    ) -> Result<ModelResponse> {
        let mut request = self.config.build_request(messages);
        apply_run_overrides(&mut request, ctx);
        let invocation = ModelInvocation::new(request, ctx.clone());
        self.service().oneshot(invocation).await
    }

    /// Open a streaming model call and return an IR `StreamDelta`
    /// stream.
    ///
    /// Pipeline: `codec.encode_streaming` → `transport.send_streaming`
    /// → `codec.decode_stream` (which owns the wire-format parser
    /// state machine). Codecs without true streaming support fall
    /// back to a single-shot pseudo-stream.
    ///
    /// This surface bypasses the layer stack — middleware that
    /// observes streaming should hook the underlying transport
    /// or wrap the returned stream directly.
    pub async fn stream_deltas<'a>(
        &'a self,
        messages: Vec<Message>,
        ctx: &ExecutionContext,
    ) -> Result<BoxDeltaStream<'a>> {
        let mut request = self.config.build_request(messages);
        apply_run_overrides(&mut request, ctx);
        let encoded = self.inner.codec().encode_streaming(&request)?;
        let warnings = encoded.warnings.clone();
        let stream = self.inner.transport().send_streaming(encoded, ctx).await?;
        if !(200..300).contains(&stream.status) {
            let mut buf = Vec::new();
            let mut body = stream.body;
            while let Some(chunk) = body.next().await {
                if let Ok(b) = chunk {
                    buf.extend_from_slice(&b);
                }
            }
            let body_text = String::from_utf8_lossy(&buf).into_owned();
            let mut err = Error::provider_http(stream.status, body_text);
            if let Some(after) =
                crate::transports::parse_retry_after(stream.headers.get("retry-after"))
            {
                err = err.with_retry_after(after);
            }
            return Err(err);
        }
        let rate_limit = self.inner.codec().extract_rate_limit(&stream.headers);
        let codec_stream = self.inner.codec().decode_stream(stream.body, warnings);
        match rate_limit {
            Some(snapshot) => {
                let prepend = futures::stream::iter(vec![Ok(StreamDelta::RateLimit(snapshot))]);
                Ok(Box::pin(prepend.chain(codec_stream)))
            }
            None => Ok(codec_stream),
        }
    }
}

impl<C: Codec + 'static, T: Transport + 'static> std::fmt::Debug for ChatModel<C, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatModel")
            .field("model", &self.config.model)
            .field("codec", &self.codec().name())
            .field("transport", &self.transport().name())
            .field("layers_attached", &self.factory.is_some())
            .finish()
    }
}

// ── Provider shortcuts ────────────────────────────────────────────
//
// One-call constructors that bundle the codec + transport + auth
// setup for the three first-party direct-API providers. These
// trade flexibility for ergonomics: callers wanting custom base
// URLs, per-call retry policies, or alternative auth shapes
// continue to use [`ChatModel::new`] with the explicit components.
//
// Why these three: they ship in `entelix-core` as built-in codecs
// against `DirectTransport`. Cloud transports (Bedrock, Vertex,
// Foundry) live in `entelix-cloud` and get their own shortcuts in
// that crate to avoid pulling AWS / GCP / Azure SDKs into the
// always-on dependency tree.
//
// Per Invariant 10, no credential ever lands on
// [`ExecutionContext`]; the API keys are wrapped in
// [`secrecy::SecretString`] and held by the credential provider,
// then read by the transport at request time.

use crate::auth::{ApiKeyProvider, BearerProvider};
use crate::codecs::{AnthropicMessagesCodec, GeminiCodec, OpenAiChatCodec};
use crate::transports::DirectTransport;
use secrecy::SecretString;

impl ChatModel<AnthropicMessagesCodec, DirectTransport> {
    /// One-call construction against `https://api.anthropic.com`.
    /// Bundles [`AnthropicMessagesCodec`] + [`DirectTransport`] +
    /// [`ApiKeyProvider`]. Mirrors LangChain's `ChatAnthropic(api_key=…)`
    /// surface so the 5-line agent path is achievable without
    /// hand-wiring four components.
    ///
    /// Returns [`Error::Config`] if the underlying HTTP client
    /// cannot be initialised.
    pub fn anthropic(api_key: impl Into<SecretString>, model: impl Into<String>) -> Result<Self> {
        let credentials = Arc::new(ApiKeyProvider::anthropic(api_key));
        let transport = DirectTransport::anthropic(credentials)?;
        Ok(Self::new(AnthropicMessagesCodec::new(), transport, model))
    }
}

impl ChatModel<OpenAiChatCodec, DirectTransport> {
    /// One-call construction against `https://api.openai.com`.
    /// Bundles [`OpenAiChatCodec`] + [`DirectTransport`] +
    /// [`BearerProvider`].
    pub fn openai(api_key: impl Into<SecretString>, model: impl Into<String>) -> Result<Self> {
        let credentials = Arc::new(BearerProvider::new(api_key));
        let transport = DirectTransport::openai(credentials)?;
        Ok(Self::new(OpenAiChatCodec::new(), transport, model))
    }
}

impl ChatModel<GeminiCodec, DirectTransport> {
    /// One-call construction against
    /// `https://generativelanguage.googleapis.com`. Bundles
    /// [`GeminiCodec`] + [`DirectTransport`] + [`BearerProvider`].
    pub fn gemini(api_key: impl Into<SecretString>, model: impl Into<String>) -> Result<Self> {
        let credentials = Arc::new(BearerProvider::new(api_key));
        let transport = DirectTransport::gemini(credentials)?;
        Ok(Self::new(GeminiCodec::new(), transport, model))
    }
}
