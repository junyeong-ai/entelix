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
use crate::ir::{
    ContentPart, JsonSchemaSpec, Message, ModelRequest, ModelResponse, ReasoningEffort,
    ResponseFormat, Role, SystemPrompt, ToolChoice, ToolSpec,
};
use crate::overrides::RunOverrides;
use crate::service::{
    BoxedModelService, BoxedStreamingService, ModelInvocation, ModelStream,
    StreamingModelInvocation,
};
use crate::stream::{StreamDelta, tap_aggregator};
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
    reasoning_effort: Option<ReasoningEffort>,
    /// `complete_typed<O>` retry budget — schema-mismatch failures
    /// reflect the parse error to the model and re-prompt up to
    /// `validation_retries` times before bubbling
    /// [`Error::Serde`]. Default `0` (no retry); operators opt in
    /// per ADR-0090. Distinct from [`crate::Error::Provider`]'s
    /// transport retries (handled by `RetryService`).
    validation_retries: u32,
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
            reasoning_effort: None,
            validation_retries: 0,
        }
    }

    /// `complete_typed<O>` retry budget. Default `0` — the first
    /// schema-mismatch fail surfaces unchanged. Operators that want
    /// the loop to reflect the parse error to the model and ask for
    /// a corrected JSON response set this to `1`–`3`.
    pub const fn validation_retries(&self) -> u32 {
        self.validation_retries
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

    /// Cross-vendor reasoning-effort knob (`None` ⇒ vendor default).
    /// Codecs translate onto their native wire shape per the
    /// per-vendor mapping documented on
    /// [`crate::ir::ReasoningEffort`]; lossy approximations emit
    /// `ModelWarning::LossyEncode`.
    pub const fn reasoning_effort(&self) -> Option<&ReasoningEffort> {
        self.reasoning_effort.as_ref()
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
            reasoning_effort: self.reasoning_effort.clone(),
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

impl<C: Codec + 'static, T: Transport + 'static> Service<StreamingModelInvocation>
    for InnerChatModel<C, T>
{
    type Response = ModelStream;
    type Error = Error;
    type Future = BoxFuture<'static, Result<ModelStream>>;

    #[inline]
    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, invocation: StreamingModelInvocation) -> Self::Future {
        let codec = Arc::clone(&self.codec);
        let transport = Arc::clone(&self.transport);
        Box::pin(async move {
            let StreamingModelInvocation {
                inner: ModelInvocation { request, ctx },
            } = invocation;
            let encoded = codec.encode_streaming(&request)?;
            let warnings = encoded.warnings.clone();
            let stream = transport.send_streaming(encoded, &ctx).await?;
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
            let rate_limit = codec.extract_rate_limit(&stream.headers);
            // The codec's `decode_stream` signature borrows
            // `&'a self`, so we tie the borrow's lifetime to the
            // owned `Arc<C>` by capturing it inside an
            // `async_stream::stream!` generator — the generator
            // becomes a `'static` future state that owns the Arc
            // for as long as the resulting stream lives. Without
            // this re-anchoring, the returned `BoxDeltaStream<'a>`
            // would borrow from a stack-local `codec` binding
            // that drops at the end of `call`.
            let codec_for_stream = Arc::clone(&codec);
            // The Rust 2024 tail-expr-drop-order change is benign
            // here — the `async_stream::stream!` block holds a
            // pinned mutable iterator (`inner`) over the codec's
            // decode pipeline; whether the temporaries inside drop
            // at end-of-block (Edition 2021) or end-of-statement
            // (Edition 2024) does not change observable
            // semantics, since `inner` is fully consumed before
            // the block returns.
            #[allow(tail_expr_drop_order)]
            let codec_stream: BoxDeltaStream<'static> = Box::pin(async_stream::stream! {
                let inner = codec_for_stream.decode_stream(stream.body, warnings);
                futures::pin_mut!(inner);
                while let Some(delta) = inner.next().await {
                    yield delta;
                }
            });
            // Prepend a synthetic `RateLimit` delta when the codec
            // could parse one from the response headers — the
            // aggregator captures it the same way it would a
            // mid-stream `Usage` delta, so the final
            // `ModelResponse` carries the snapshot operators
            // expect on `ModelResponse::rate_limit`.
            let prefixed: BoxDeltaStream<'static> = match rate_limit {
                Some(snapshot) => {
                    let prepend = futures::stream::iter(vec![Ok(StreamDelta::RateLimit(snapshot))]);
                    Box::pin(prepend.chain(codec_stream))
                }
                None => codec_stream,
            };
            Ok(tap_aggregator(prefixed))
        })
    }
}

/// Boxed factory: `InnerChatModel` → layered [`BoxedModelService`].
/// Stored on `ChatModel` so the type of the (potentially deeply
/// nested) layer stack stays opaque to users; one concrete
/// `ChatModel<C, T>` shape regardless of how many layers are
/// attached.
type LayerFactory<C, T> = Arc<dyn Fn(InnerChatModel<C, T>) -> BoxedModelService + Send + Sync>;

/// Boxed factory for the streaming-side spine. Parallel to
/// [`LayerFactory`]; layers attached via [`ChatModel::layer`] are
/// stacked onto this factory the same way they're stacked onto
/// the one-shot factory, so a single `.layer(OtelLayer::new(...))`
/// call wraps both spines with one observability stack.
type StreamingLayerFactory<C, T> =
    Arc<dyn Fn(InnerChatModel<C, T>) -> BoxedStreamingService + Send + Sync>;

/// Configurable chat model — codec + transport + layer stack.
///
/// Cheap to clone (handles are `Arc`-backed). Builder-style
/// configuration via `with_*` methods; layer attachment via
/// [`Self::layer`].
pub struct ChatModel<C: Codec + 'static, T: Transport + 'static> {
    inner: InnerChatModel<C, T>,
    config: ChatModelConfig,
    factory: Option<LayerFactory<C, T>>,
    streaming_factory: Option<StreamingLayerFactory<C, T>>,
}

impl<C: Codec + 'static, T: Transport + 'static> Clone for ChatModel<C, T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            config: self.config.clone(),
            factory: self.factory.clone(),
            streaming_factory: self.streaming_factory.clone(),
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
            streaming_factory: None,
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

    /// Set the [`complete_typed`](Self::complete_typed) validation
    /// retry budget — number of times the loop reflects a
    /// schema-mismatch [`Error::Serde`] back to the model with
    /// corrective hint text before bubbling the error. Default `0`
    /// (no retry). Each retry increments the conversation length by
    /// two messages (assistant's failed reply + retry prompt).
    #[must_use]
    pub const fn with_validation_retries(mut self, n: u32) -> Self {
        self.config.validation_retries = n;
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

    /// Set the cross-vendor reasoning-effort knob. Codecs translate
    /// onto their native wire shape per the table on
    /// [`crate::ir::ReasoningEffort`] — `Off` / `Minimal` / `Low`
    /// / `Medium` / `High` / `Auto` snap to vendor buckets, lossy
    /// approximations emit `ModelWarning::LossyEncode`, and
    /// `VendorSpecific(s)` passes the literal vendor wire value
    /// through.
    #[must_use]
    pub fn with_reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.config.reasoning_effort = Some(effort);
        self
    }

    /// Append a `tower::Layer` to **both** dispatch spines — the
    /// one-shot path (`Service<ModelInvocation, Response =
    /// ModelResponse>`) and the streaming path
    /// (`Service<StreamingModelInvocation, Response =
    /// ModelStream>`). Layers run in registration order around
    /// each leaf — the first-registered layer is *outermost*
    /// (sees the request first, the response last).
    ///
    /// `PolicyLayer` and `OtelLayer` from the policy / otel
    /// crates satisfy both spines, so a single `.layer(...)` call
    /// wraps the agent's complete dispatch fan-out:
    ///
    /// - one-shot: `Service<ModelInvocation, Response = ModelResponse>`
    /// - streaming: `Service<StreamingModelInvocation, Response = ModelStream>`
    /// - tool dispatch (separate, on `ToolRegistry::layer`):
    ///   `Service<ToolInvocation, Response = Value>`
    ///
    /// The streaming-side layer wraps the [`ModelStream`]'s
    /// `completion` future so observability events (cost,
    /// latency, span close) fire only on the `Ok` branch — a
    /// stream that errors mid-flight surfaces the error and
    /// emits no charge (invariant 12).
    ///
    /// Layers that legitimately apply only to the one-shot spine
    /// (e.g. `RetryLayer` — retrying a partially-streamed
    /// response is meaningless) implement a pass-through
    /// `Layer<BoxedStreamingService>` so they satisfy the
    /// constraint without affecting streaming dispatch.
    #[must_use]
    pub fn layer<L>(mut self, layer: L) -> Self
    where
        L: Layer<BoxedModelService> + Layer<BoxedStreamingService> + Clone + Send + Sync + 'static,
        <L as Layer<BoxedModelService>>::Service: Service<ModelInvocation, Response = ModelResponse, Error = Error>
            + Clone
            + Send
            + 'static,
        <<L as Layer<BoxedModelService>>::Service as Service<ModelInvocation>>::Future:
            Send + 'static,
        <L as Layer<BoxedStreamingService>>::Service: Service<StreamingModelInvocation, Response = ModelStream, Error = Error>
            + Clone
            + Send
            + 'static,
        <<L as Layer<BoxedStreamingService>>::Service as Service<StreamingModelInvocation>>::Future:
            Send + 'static,
    {
        let prev = self.factory.take();
        let prev_streaming = self.streaming_factory.take();
        let layer_one_shot = layer.clone();
        let layer_streaming = layer;
        let new_factory: LayerFactory<C, T> = Arc::new(move |inner: InnerChatModel<C, T>| {
            let stacked: BoxedModelService = match &prev {
                Some(prev_factory) => prev_factory(inner),
                None => BoxCloneService::new(inner),
            };
            BoxCloneService::new(<L as Layer<BoxedModelService>>::layer(
                &layer_one_shot,
                stacked,
            ))
        });
        let new_streaming: StreamingLayerFactory<C, T> =
            Arc::new(move |inner: InnerChatModel<C, T>| {
                let stacked: BoxedStreamingService = match &prev_streaming {
                    Some(prev_factory) => prev_factory(inner),
                    None => BoxCloneService::new(inner),
                };
                BoxCloneService::new(<L as Layer<BoxedStreamingService>>::layer(
                    &layer_streaming,
                    stacked,
                ))
            });
        self.factory = Some(new_factory);
        self.streaming_factory = Some(new_streaming);
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

    /// Build the layered [`BoxedStreamingService`] — the streaming
    /// counterpart to [`Self::service`]. Layers attached via
    /// [`Self::layer`] wrap this spine the same way they wrap the
    /// one-shot service; consumers driving the service directly
    /// (rather than through [`Self::stream_deltas`]) drive
    /// `Service<StreamingModelInvocation, Response = ModelStream>`.
    #[must_use]
    pub fn streaming_service(&self) -> BoxedStreamingService {
        match &self.streaming_factory {
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
        let budget = ctx.run_budget();
        if let Some(budget) = &budget {
            // Pre-call axes — request count cap. Token caps fire
            // post-decode below.
            budget.check_pre_request()?;
        }
        let mut request = self.config.build_request(messages);
        apply_run_overrides(&mut request, ctx);
        let invocation = ModelInvocation::new(request, ctx.clone());
        let response = self.service().oneshot(invocation).await?;
        if let Some(budget) = &budget {
            // Post-call accumulation — invariant 12 transactional
            // semantics: only on the `Ok` branch does the budget
            // see usage. The `?` above ensures the error branch
            // never reaches this line.
            budget.observe_usage(&response.usage)?;
        }
        Ok(response)
    }

    /// Send a conversation and return a typed `O` parsed from the
    /// model's structured-output channel.
    ///
    /// The codec emits a `response_format` directive carrying the
    /// schemars-derived schema for `O`; the dispatch shape (native
    /// JSON-Schema vs forced tool call) is the codec's
    /// [`Codec::auto_output_strategy`] for the configured model
    /// when [`crate::ir::OutputStrategy::Auto`] is selected — see
    /// [`crate::ir::OutputStrategy`] and ADR-0079 for the cross-vendor mapping.
    /// Operators that need to override the strategy build their
    /// own [`ResponseFormat`] via
    /// [`ResponseFormat::with_strategy`] and attach it to the
    /// request through a custom flow.
    ///
    /// On `Native` dispatch, the codec produces a single text
    /// `ContentPart` whose body parses as `O`. On `Tool` dispatch,
    /// the codec emits one forced `ContentPart::ToolUse` whose
    /// `input` is the JSON object the model produced; this method
    /// extracts the input and parses it as `O`.
    ///
    /// `O: JsonSchema + DeserializeOwned + Send + 'static` —
    /// schemars derives the JSON Schema at call time (zero-cost
    /// after the first call thanks to schemars' static schema
    /// caching). Production operators that cache the schema across
    /// many calls build the [`JsonSchemaSpec`] once and attach it
    /// to the request directly (no `O` type parameter on the
    /// caller side).
    pub async fn complete_typed<O>(
        &self,
        messages: Vec<Message>,
        ctx: &ExecutionContext,
    ) -> Result<O>
    where
        O: schemars::JsonSchema + serde::de::DeserializeOwned + Send + 'static,
    {
        // Derive the JSON Schema once per call. schemars handles
        // memoisation at the type-level via its `for_value` cache;
        // operators sensitive to per-call overhead pre-build the
        // `JsonSchemaSpec` and route through `complete_full`.
        let schema_value = serde_json::to_value(schemars::schema_for!(O)).map_err(Error::Serde)?;
        let type_name = std::any::type_name::<O>();
        // Strip the module path so the wire-side `name` is short
        // and stable (`entelix_core::ir::request::ModelRequest` →
        // `ModelRequest`); vendors that surface the name in
        // observability ship a readable string.
        let short_name = type_name.rsplit("::").next().unwrap_or(type_name);
        let spec = JsonSchemaSpec::new(short_name, schema_value)?;
        let format = ResponseFormat::strict(spec);

        let mut conversation = messages;
        let max_retries = self.config.validation_retries;
        let mut attempt: u32 = 0;
        loop {
            let budget = ctx.run_budget();
            if let Some(budget) = &budget {
                budget.check_pre_request()?;
            }
            let mut request = self.config.build_request(conversation.clone());
            apply_run_overrides(&mut request, ctx);
            request.response_format = Some(format.clone());

            let invocation = ModelInvocation::new(request, ctx.clone());
            let response = self.service().oneshot(invocation).await?;
            if let Some(budget) = &budget {
                budget.observe_usage(&response.usage)?;
            }
            // Capture the assistant's reply text for the retry path.
            // `parse_typed_response` consumes by value; clone the
            // text-block content first so a parse failure can
            // re-feed the model its own output as context.
            let assistant_text = response_text_for_retry(&response);
            match parse_typed_response::<O>(response) {
                Ok(value) => return Ok(value),
                Err(err) if matches!(err, Error::Serde(_)) && attempt < max_retries => {
                    attempt += 1;
                    let parse_diagnostic = err.to_string();
                    // Echo the assistant's failed turn into the
                    // conversation so the next call sees what it
                    // produced; then add a user-side correction.
                    conversation.push(Message::new(
                        crate::ir::Role::Assistant,
                        vec![ContentPart::Text {
                            text: assistant_text.unwrap_or_default(),
                            cache_control: None,
                        }],
                    ));
                    conversation.push(Message::new(
                        crate::ir::Role::User,
                        vec![ContentPart::Text {
                            text: format!(
                                "Your previous response did not match the required JSON schema for `{short_name}`. \
                                 Parser diagnostic: {parse_diagnostic}\n\
                                 Re-emit the response as a single valid JSON object that conforms to the schema."
                            ),
                            cache_control: None,
                        }],
                    ));
                }
                Err(err) => return Err(err),
            }
        }
    }

    /// Send a conversation, parse the structured-output response as
    /// `O`, and run `validator` against the parsed value. Mirrors
    /// [`Self::complete_typed`] for the schema-mismatch retry path
    /// and additionally catches [`crate::Error::ModelRetry`] raised
    /// from the validator — both error kinds reflect a corrective
    /// hint to the model and re-invoke within the same
    /// [`ChatModelConfig::validation_retries`](crate::ChatModelConfig::validation_retries)
    /// budget.
    ///
    /// The validator surface is sync ([`crate::OutputValidator::validate`])
    /// so simple closures (`|out: &O| -> Result<()>`) compose
    /// without `async-trait` ceremony. Validators that need to
    /// `.await` (DB lookup, external check) compose around the
    /// `complete_typed_validated` call boundary instead — run the
    /// async work after the typed response returns.
    pub async fn complete_typed_validated<O, V>(
        &self,
        messages: Vec<Message>,
        validator: V,
        ctx: &ExecutionContext,
    ) -> Result<O>
    where
        O: schemars::JsonSchema + serde::de::DeserializeOwned + Send + 'static,
        V: crate::output_validator::OutputValidator<O>,
    {
        let schema_value = serde_json::to_value(schemars::schema_for!(O)).map_err(Error::Serde)?;
        let type_name = std::any::type_name::<O>();
        let short_name = type_name.rsplit("::").next().unwrap_or(type_name);
        let spec = JsonSchemaSpec::new(short_name, schema_value)?;
        let format = ResponseFormat::strict(spec);

        let mut conversation = messages;
        let max_retries = self.config.validation_retries;
        let mut attempt: u32 = 0;
        loop {
            let budget = ctx.run_budget();
            if let Some(budget) = &budget {
                budget.check_pre_request()?;
            }
            let mut request = self.config.build_request(conversation.clone());
            apply_run_overrides(&mut request, ctx);
            request.response_format = Some(format.clone());

            let invocation = ModelInvocation::new(request, ctx.clone());
            let response = self.service().oneshot(invocation).await?;
            if let Some(budget) = &budget {
                budget.observe_usage(&response.usage)?;
            }
            let assistant_text = response_text_for_retry(&response);
            let parse_outcome = parse_typed_response::<O>(response);

            // Combine schema-mismatch and validator-driven retries
            // through one match. The retry budget is shared because
            // both paths represent "the model emitted output we
            // can't accept" — distinguishing them at the budget
            // level adds knobs without buying behaviour operators
            // commonly want to vary independently.
            let retry_hint = match parse_outcome {
                Ok(value) => match validator.validate(&value) {
                    Ok(()) => return Ok(value),
                    Err(Error::ModelRetry { hint, .. }) if attempt < max_retries => {
                        Some(hint.into_inner())
                    }
                    Err(err) => return Err(err),
                },
                Err(err) if matches!(err, Error::Serde(_)) && attempt < max_retries => {
                    Some(format!(
                        "Your previous response did not match the required JSON schema for `{short_name}`. \
                         Parser diagnostic: {err}\n\
                         Re-emit the response as a single valid JSON object that conforms to the schema."
                    ))
                }
                Err(err) => return Err(err),
            };

            let Some(hint) = retry_hint else {
                unreachable!()
            };
            attempt += 1;
            conversation.push(Message::new(
                crate::ir::Role::Assistant,
                vec![ContentPart::Text {
                    text: assistant_text.unwrap_or_default(),
                    cache_control: None,
                }],
            ));
            conversation.push(Message::new(
                crate::ir::Role::User,
                vec![ContentPart::Text {
                    text: hint,
                    cache_control: None,
                }],
            ));
        }
    }

    /// Open a streaming model call and return an IR `StreamDelta`
    /// stream.
    ///
    /// Pipeline: `codec.encode_streaming` → `transport.send_streaming`
    /// → `codec.decode_stream` → `tap_aggregator`, all driven
    /// through the same `tower::Service` spine as
    /// [`Self::complete_full`]. Layers attached via
    /// [`Self::layer`] (e.g. `OtelLayer`, `PolicyLayer`) wrap
    /// the streaming dispatch the same way they wrap the one-shot
    /// dispatch; observability events (cost, span close) fire on
    /// the streaming-side completion future's `Ok` branch only —
    /// invariant 12.
    ///
    /// The returned [`ModelStream`] carries both the delta stream
    /// (consumer-visible) and a `completion` future that resolves
    /// to the aggregated [`ModelResponse`] after the consumer
    /// drains the stream. Layers wrap `completion` to gate
    /// observability emission on the `Ok` branch — a stream that
    /// errors mid-flight surfaces the error on both the consumer
    /// side and the completion future, and no charge fires.
    ///
    /// Codecs without true streaming support fall back to a
    /// single-shot pseudo-stream the same way they did before
    /// the spine refactor.
    pub async fn stream_deltas(
        &self,
        messages: Vec<Message>,
        ctx: &ExecutionContext,
    ) -> Result<ModelStream> {
        let budget = ctx.run_budget();
        if let Some(budget) = &budget {
            budget.check_pre_request()?;
        }
        let mut request = self.config.build_request(messages);
        apply_run_overrides(&mut request, ctx);
        let invocation = StreamingModelInvocation::new(ModelInvocation::new(request, ctx.clone()));
        let model_stream = self.streaming_service().oneshot(invocation).await?;
        let ModelStream { stream, completion } = model_stream;
        // Wrap completion to observe usage on the streaming-side
        // `Ok` branch — invariant 12 transactional semantics: a
        // stream that errors mid-flight resolves `completion` to
        // `Err` and never reaches the budget. Mirrors the
        // OtelLayer / PolicyLayer streaming wrap from G-1.
        let budget_for_completion = budget.clone();
        let user_facing = async move {
            let result = completion.await;
            if let (Ok(response), Some(budget)) = (&result, budget_for_completion.as_ref()) {
                budget.observe_usage(&response.usage)?;
            }
            result
        };
        Ok(ModelStream {
            stream,
            completion: Box::pin(user_facing),
        })
    }

    /// Streaming sibling of [`Self::complete_typed`]. Returns a
    /// [`TypedModelStream<O>`] whose `stream` field exposes raw
    /// [`StreamDelta`]s (text fragments operators echo to the user
    /// during generation) and whose `completion` future resolves to
    /// the aggregated, parsed `O`.
    ///
    /// `response_format = ResponseFormat::strict(JsonSchemaSpec::for::<O>())`
    /// is set on the request so the model emits the typed JSON
    /// payload natively (Native strategy via `text` deltas) or
    /// through a `tool_use` block (Tool strategy). The aggregator
    /// behind [`Self::stream_deltas`] already collects both shapes
    /// into the final [`ModelResponse`]; `stream_typed` parses the
    /// completion the same way [`Self::complete_typed`] does.
    ///
    /// `O: JsonSchema + DeserializeOwned + Send + 'static` —
    /// schemars derives the JSON Schema at call time.
    ///
    /// # Streaming + retry tradeoff
    ///
    /// `stream_typed` does **not** retry on parse failure: by the
    /// time `completion` resolves the deltas have already been
    /// surfaced to the consumer, so re-invoking with a corrective
    /// hint would emit a divergent second stream. Operators wanting
    /// the [`ChatModelConfig::validation_retries`] loop call
    /// [`Self::complete_typed`] / [`Self::complete_typed_validated`]
    /// instead — a parse failure there is fully recoverable because
    /// no partial output was surfaced.
    ///
    /// A validator that needs to inspect the parsed `O` runs after
    /// `completion` resolves: `let value = stream.completion.await?;
    /// validator.validate(&value)?;`. The validator's `Err` does not
    /// flow back into the stream — it surfaces alongside the typed
    /// completion at the call site.
    pub async fn stream_typed<O>(
        &self,
        messages: Vec<Message>,
        ctx: &ExecutionContext,
    ) -> Result<TypedModelStream<O>>
    where
        O: schemars::JsonSchema + serde::de::DeserializeOwned + Send + 'static,
    {
        if self.config.validation_retries > 0 {
            // Surface the contract divergence — operators wiring
            // with_validation_retries on a ChatModel and then calling
            // stream_typed almost certainly expect the retry loop to
            // also cover the streaming path. ADR-0096 records the
            // design decision (no-retry on streaming because deltas
            // were already surfaced to the consumer); a debug log
            // makes the silent ignore visible at run time.
            tracing::debug!(
                validation_retries = self.config.validation_retries,
                "ChatModel::stream_typed ignores validation_retries — \
                 streaming + retry would emit a divergent second stream \
                 over already-surfaced deltas. Use complete_typed for \
                 the unified retry budget. ADR-0096."
            );
        }
        let schema_value = serde_json::to_value(schemars::schema_for!(O)).map_err(Error::Serde)?;
        let type_name = std::any::type_name::<O>();
        let short_name = type_name.rsplit("::").next().unwrap_or(type_name);
        let spec = JsonSchemaSpec::new(short_name, schema_value)?;
        let format = ResponseFormat::strict(spec);

        let budget = ctx.run_budget();
        if let Some(budget) = &budget {
            budget.check_pre_request()?;
        }
        let mut request = self.config.build_request(messages);
        apply_run_overrides(&mut request, ctx);
        request.response_format = Some(format);

        let invocation = StreamingModelInvocation::new(ModelInvocation::new(request, ctx.clone()));
        let model_stream = self.streaming_service().oneshot(invocation).await?;
        let ModelStream { stream, completion } = model_stream;

        let budget_for_completion = budget.clone();
        let typed_completion = async move {
            let response = completion.await?;
            if let Some(budget) = &budget_for_completion {
                budget.observe_usage(&response.usage)?;
            }
            parse_typed_response::<O>(response)
        };
        Ok(TypedModelStream {
            stream,
            completion: Box::pin(typed_completion),
        })
    }
}

/// Streaming counterpart to a `complete_typed` call — raw deltas on
/// `stream`, the aggregated typed payload on `completion`.
///
/// The `stream` field is identical to [`ModelStream::stream`]: raw
/// [`StreamDelta`]s the consumer surfaces to the user (text
/// fragments, thinking deltas, …) as the model emits them. The
/// `completion` future resolves to the typed `O` parsed from the
/// aggregated final response — operators await it once the stream
/// has been drained.
///
/// Mirrors [`crate::service::ModelStream`] for the typed-output
/// path; the underlying aggregator is the same `tap_aggregator`
/// used by [`ChatModel::stream_deltas`].
pub struct TypedModelStream<O> {
    /// Raw delta stream surfaced to the consumer.
    pub stream: BoxDeltaStream<'static>,
    /// Future resolving to the typed `O` after the stream has been
    /// consumed to its terminal `Stop`. Drops the `TypedModelStream`
    /// before draining the stream to surface an `Err` on completion.
    pub completion: BoxFuture<'static, Result<O>>,
}

impl<O> std::fmt::Debug for TypedModelStream<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TypedModelStream")
            .field("stream", &"<BoxDeltaStream>")
            .field(
                "completion",
                &format_args!("<BoxFuture<Result<{}>>>", std::any::type_name::<O>()),
            )
            .finish()
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

/// Parse a [`ModelResponse`] produced by a `complete_typed`
/// dispatch into the operator's typed `O`. The two dispatch
/// shapes both surface the JSON object the model produced —
/// `Native` strategy lands as a single text content part whose
/// body is the JSON document; `Tool` strategy lands as a single
/// `ContentPart::ToolUse` whose `input` is the JSON object. The
/// helper tries the tool path first (Anthropic / Bedrock-Anthropic
/// default), then falls through to the text path (OpenAI /
/// Gemini default).
// `response` is taken by value to express ownership of the
// response payload — the function consumes the response's
// stop reason / warnings semantics from the caller's
// perspective even though only `content` is read. Clippy's
// `needless_pass_by_value` would have us borrow, but a borrow
// would let the caller continue to inspect `response.warnings`
// (etc.) after `parse_typed_response` succeeded, masking the
// "this response has been consumed" intent the typed surface
// communicates.
/// Concatenate every `ContentPart::Text` block in `response` for
/// the retry-path conversation echo. Returns `None` when the model
/// surfaced no textual content (in that case the retry loop seeds
/// the assistant turn with an empty string — the diagnostic message
/// alone is enough context for the model to course-correct).
fn response_text_for_retry(response: &ModelResponse) -> Option<String> {
    let mut out = String::new();
    for part in &response.content {
        if let ContentPart::Text { text, .. } = part {
            out.push_str(text);
        }
    }
    if out.is_empty() { None } else { Some(out) }
}

#[allow(clippy::needless_pass_by_value)]
fn parse_typed_response<O>(response: ModelResponse) -> Result<O>
where
    O: serde::de::DeserializeOwned,
{
    for part in &response.content {
        if let ContentPart::ToolUse { input, .. } = part {
            return serde_json::from_value(input.clone()).map_err(Error::Serde);
        }
    }
    for part in &response.content {
        if let ContentPart::Text { text, .. } = part {
            return serde_json::from_str(text).map_err(Error::Serde);
        }
    }
    Err(Error::invalid_request(
        "complete_typed: model response carried neither a `tool_use` block nor a text \
         block — the configured `OutputStrategy` did not produce typed output",
    ))
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
