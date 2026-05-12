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
use crate::overrides::{RequestOverrides, RunOverrides};
use crate::service::{
    BoxedModelService, BoxedStreamingService, ModelInvocation, ModelStream, NamedLayer,
    StreamingModelInvocation,
};
use crate::stream::{StreamDelta, tap_aggregator};
use crate::transports::Transport;

/// Patch `request` with any [`RunOverrides`] and [`RequestOverrides`]
/// attached to `ctx`. Both `complete_full` and `stream_deltas` route
/// through this helper so the override semantics stay identical
/// across the two surfaces.
///
/// `RunOverrides` patches agent-loop-owned fields (`model`,
/// `system`); `RequestOverrides` patches `ModelRequest`-shaped
/// sampling knobs (`temperature`, `top_p`, `max_tokens`,
/// `stop_sequences`, `reasoning_effort`, `tool_choice`,
/// `response_format`). Either, both, or neither may be present.
fn apply_overrides(request: &mut ModelRequest, ctx: &ExecutionContext) {
    if let Some(run) = ctx.extension::<RunOverrides>() {
        if let Some(model) = run.model() {
            model.clone_into(&mut request.model);
        }
        if let Some(system) = run.system_prompt() {
            request.system = system.clone();
        }
        if let Some(specs) = run.tool_specs() {
            request.tools = Arc::clone(specs);
        }
    }
    if let Some(req) = ctx.extension::<RequestOverrides>() {
        if let Some(t) = req.temperature() {
            request.temperature = Some(t);
        }
        if let Some(p) = req.top_p() {
            request.top_p = Some(p);
        }
        if let Some(k) = req.top_k() {
            request.top_k = Some(k);
        }
        if let Some(n) = req.max_tokens() {
            request.max_tokens = Some(n);
        }
        if let Some(sequences) = req.stop_sequences() {
            request.stop_sequences = sequences.to_vec();
        }
        if let Some(effort) = req.reasoning_effort() {
            request.reasoning_effort = Some(effort.clone());
        }
        if let Some(choice) = req.tool_choice() {
            request.tool_choice = choice.clone();
        }
        if let Some(format) = req.response_format() {
            request.response_format = Some(format.clone());
        }
        if let Some(parallel) = req.parallel_tool_calls() {
            request.parallel_tool_calls = Some(parallel);
        }
        if let Some(user_id) = req.end_user_id() {
            request.end_user_id = Some(user_id.to_owned());
        }
        if let Some(seed) = req.seed() {
            request.seed = Some(seed);
        }
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
    top_k: Option<u32>,
    stop_sequences: Vec<String>,
    tools: Arc<[ToolSpec]>,
    tool_choice: ToolChoice,
    reasoning_effort: Option<ReasoningEffort>,
    /// `complete_typed<O>` retry budget — schema-mismatch and
    /// [`crate::OutputValidator`] failures both reflect their hint
    /// to the model and re-prompt up to `validation_retries` times
    /// before surfacing [`Error::ModelRetry`] (invariant 20).
    /// Default `0` (no retry). Distinct from
    /// [`crate::Error::Provider`]'s transport retries (handled by
    /// `RetryService`).
    validation_retries: u32,
    /// Operator-supplied token counter for pre-flight budget checks
    /// and content-economy estimation. `None` (the default) means
    /// the SDK relies on the vendor's post-flight `Usage` block;
    /// pre-flight enforcement (refusing a call that would exceed
    /// the configured `RunBudget` ceiling before sending) requires
    /// an explicit counter. Concrete counters ship as companion
    /// crates (`entelix-tokenizer-tiktoken`,
    /// `entelix-tokenizer-hf`, locale-aware companions);
    /// [`crate::ByteCountTokenCounter`] is the zero-dependency
    /// English-biased default.
    token_counter: Option<std::sync::Arc<dyn crate::tokens::TokenCounter>>,
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
            top_k: None,
            stop_sequences: Vec::new(),
            tools: Arc::from([]),
            tool_choice: ToolChoice::default(),
            reasoning_effort: None,
            validation_retries: 0,
            token_counter: None,
        }
    }

    /// Borrow the configured token counter, if any. Returns `None`
    /// when the operator has not wired one — pre-flight budget
    /// enforcement falls back to vendor `Usage` post-response.
    #[must_use]
    pub fn token_counter(&self) -> Option<&std::sync::Arc<dyn crate::tokens::TokenCounter>> {
        self.token_counter.as_ref()
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

    /// Top-k sampling parameter (`None` ⇒ vendor default). Codec
    /// support follows the IR mapping documented on
    /// [`crate::ir::ModelRequest::top_k`].
    pub const fn top_k(&self) -> Option<u32> {
        self.top_k
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
    /// [`ModelRequest`]. Only the fields the config carries are
    /// projected; per-request knobs and provider extensions stay at
    /// their `Default` (i.e. unset) and flow in via `RequestOverrides`
    /// or direct `ExecutionContext::add_extension` instead.
    #[must_use]
    pub fn build_request(&self, messages: Vec<Message>) -> ModelRequest {
        ModelRequest {
            model: self.model.clone(),
            messages,
            system: self.system.clone(),
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            stop_sequences: self.stop_sequences.clone(),
            tools: Arc::clone(&self.tools),
            tool_choice: self.tool_choice.clone(),
            reasoning_effort: self.reasoning_effort.clone(),
            ..ModelRequest::default()
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
    /// Diagnostic stack — names captured at [`Self::layer`] compose
    /// time, in insertion order (innermost composed first → outermost
    /// composed last). The wrapped service stacks last-composed
    /// outermost, so this `Vec` reads bottom-to-top relative to
    /// request flow.
    layer_names: Vec<&'static str>,
}

impl<C: Codec + 'static, T: Transport + 'static> Clone for ChatModel<C, T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            config: self.config.clone(),
            factory: self.factory.clone(),
            streaming_factory: self.streaming_factory.clone(),
            layer_names: self.layer_names.clone(),
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
            layer_names: Vec::new(),
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
    /// schema-mismatch or [`crate::OutputValidator`] failure back to
    /// the model with corrective hint text before surfacing the
    /// terminal [`Error::ModelRetry`]. Default `0` (no retry). Each
    /// retry increments the conversation length by two messages
    /// (assistant's failed reply + retry prompt). Both retry shapes
    /// share one budget and route through `Error::ModelRetry`
    /// (invariant 20).
    #[must_use]
    pub const fn with_validation_retries(mut self, n: u32) -> Self {
        self.config.validation_retries = n;
        self
    }

    /// Wire an operator-supplied [`TokenCounter`](crate::TokenCounter)
    /// for pre-flight budget checks and content-economy estimation.
    /// Each call replaces the configured counter — the last call wins.
    ///
    /// Vendor-accurate counters (tiktoken, HuggingFace, ko-mecab)
    /// ship as companion crates so the core stays
    /// zero-dependency. [`crate::ByteCountTokenCounter`] is the
    /// English-biased zero-dependency default for development
    /// scaffolding.
    #[must_use]
    pub fn with_token_counter(
        mut self,
        counter: std::sync::Arc<dyn crate::tokens::TokenCounter>,
    ) -> Self {
        self.config.token_counter = Some(counter);
        self
    }

    /// Set nucleus sampling parameter.
    #[must_use]
    pub const fn with_top_p(mut self, p: f32) -> Self {
        self.config.top_p = Some(p);
        self
    }

    /// Set top-k sampling parameter. Native on Anthropic, Gemini,
    /// and Bedrock-Anthropic; OpenAI codecs surface as
    /// `LossyEncode`.
    #[must_use]
    pub const fn with_top_k(mut self, k: u32) -> Self {
        self.config.top_k = Some(k);
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

    /// Replace the advertised tools list. Accepts anything that
    /// converts into `Arc<[ToolSpec]>` — `Vec<ToolSpec>` and
    /// `[ToolSpec; N]` literals both qualify, so caller ergonomics
    /// match the previous `Vec` shape while per-dispatch
    /// `build_request` clones become an atomic refcount bump rather
    /// than a deep walk of every tool's JSON schema.
    #[must_use]
    pub fn with_tools(mut self, tools: impl Into<Arc<[ToolSpec]>>) -> Self {
        self.config.tools = tools.into();
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
    /// ModelStream>`). Each `.layer(L)` wraps `L` *around* the
    /// already-composed stack, so the **last-registered layer is
    /// outermost** (sees the request first, the response last) and
    /// the first-registered layer sits innermost against the leaf
    /// `InnerChatModel`.
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
    ///
    /// ## Introspection
    ///
    /// Every layer is recorded in [`Self::layer_names`] at compose
    /// time via [`NamedLayer::layer_name`]. First-party entelix
    /// layers (`PolicyLayer`, `OtelLayer`) implement
    /// [`NamedLayer`] directly; external `tower::Layer` middleware
    /// wraps through [`crate::service::WithName`] to participate:
    ///
    /// ```ignore
    /// use entelix_core::WithName;
    ///
    /// chat_model
    ///     .layer(PolicyLayer::new(registry))
    ///     .layer(OtelLayer::new("anthropic"))
    ///     .layer(WithName::new("concurrency", tower::limit::ConcurrencyLimitLayer::new(10)));
    /// // chat_model.layer_names() == ["policy", "otel", "concurrency"]
    /// ```
    #[must_use]
    pub fn layer<L>(mut self, layer: L) -> Self
    where
        L: Layer<BoxedModelService>
            + Layer<BoxedStreamingService>
            + NamedLayer
            + Clone
            + Send
            + Sync
            + 'static,
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
        // Capture the static name BEFORE the type-erasing closure
        // swallows `L`. The factory chain (Arc<dyn Fn(InnerChatModel)
        // -> BoxedModelService>) discards concrete layer types, so
        // introspection has to happen here at compose time.
        self.layer_names.push(layer.layer_name());

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

    /// Compose a `tower::Layer` whose only difference from a
    /// first-party entelix layer is the missing [`NamedLayer`]
    /// impl. Equivalent to `self.layer(WithName::new(name, layer))`
    /// — the wrapper supplies the identity surfaced through
    /// [`Self::layer_names`].
    ///
    /// Use this for external middleware (`tower::limit`,
    /// `tower::timeout`, operator-defined wrappers) without
    /// implementing [`NamedLayer`] yourself. First-party entelix
    /// layers already implement [`NamedLayer`] and should reach for
    /// [`Self::layer`] directly so the canonical role name
    /// (e.g. `"policy"`, `"otel"`, `"retry"`) ships at the call
    /// site.
    #[must_use]
    pub fn layer_named<L>(self, name: &'static str, layer: L) -> Self
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
        self.layer(crate::service::WithName::new(name, layer))
    }

    /// Diagnostic snapshot of the composed layer stack, in
    /// registration order: `[0]` is the first `.layer(...)` call
    /// (innermost, against the leaf `InnerChatModel`); the last
    /// element is the most recent registration (outermost, sees
    /// requests first). Empty for a `ChatModel` with no layers
    /// composed.
    ///
    /// Surfaced for boot-time wiring assertions, debug dashboards,
    /// and conditional-layer audits. The values are
    /// [`NamedLayer::layer_name`] outputs — `&'static str` and
    /// patch-version-stable per the trait's contract.
    #[must_use]
    pub fn layer_names(&self) -> &[&'static str] {
        &self.layer_names
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
        apply_overrides(&mut request, ctx);
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
    /// [`crate::ir::OutputStrategy`] and for the cross-vendor mapping.
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
        self.complete_typed_validated(messages, |_: &O| Ok(()), ctx)
            .await
    }

    /// Send a conversation, parse the structured-output response as
    /// `O`, and run `validator` against the parsed value.
    ///
    /// Both failure modes — schema-mismatch (the model emitted JSON
    /// the deserialiser couldn't bind to `O`) and validator failure
    /// (the deserialised value broke a semantic invariant) — route
    /// through one channel: [`crate::Error::ModelRetry`]. The retry
    /// loop catches the variant, reflects the hint to the model as
    /// a corrective user message, and re-invokes within the same
    /// [`ChatModelConfig::validation_retries`](crate::ChatModelConfig::validation_retries)
    /// budget (invariant 20).
    ///
    /// [`Self::complete_typed`] is the no-validator shortcut — it
    /// calls into this method with an always-`Ok` validator so the
    /// schema-mismatch retry path stays uniform.
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
        use crate::llm_facing::RenderedForLlm;

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
            apply_overrides(&mut request, ctx);
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

            // Both failure modes arrive as `Error::ModelRetry`:
            // schema-mismatch is wrapped inside `parse_typed_response`
            // at the parse site, and validator failures already
            // construct `Error::ModelRetry` themselves. A shared
            // budget governs both because they reflect the same
            // condition — the model emitted output the harness
            // cannot accept — and distinguishing them at the budget
            // level would add knobs without buying behaviour
            // operators commonly want to vary independently
            // (invariant 20).
            let retry_hint: RenderedForLlm<String> =
                match parse_typed_response::<O>(short_name, response) {
                    Ok(value) => match validator.validate(&value) {
                        Ok(()) => return Ok(value),
                        Err(Error::ModelRetry { hint, .. }) => hint,
                        Err(err) => return Err(err),
                    },
                    Err(Error::ModelRetry { hint, .. }) => hint,
                    Err(err) => return Err(err),
                };

            if attempt >= max_retries {
                return Err(Error::model_retry(retry_hint, attempt));
            }
            attempt += 1;

            // Echo the assistant's failed turn into the conversation
            // so the next call sees what it produced, then push the
            // corrective user message carrying the rendered hint.
            conversation.push(Message::new(
                crate::ir::Role::Assistant,
                vec![ContentPart::Text {
                    text: assistant_text.unwrap_or_default(),
                    cache_control: None,
                    provider_echoes: Vec::new(),
                }],
            ));
            conversation.push(Message::new(
                crate::ir::Role::User,
                vec![ContentPart::Text {
                    text: retry_hint.into_inner(),
                    cache_control: None,
                    provider_echoes: Vec::new(),
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
        apply_overrides(&mut request, ctx);
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
            // also cover the streaming path. records the
            // design decision (no-retry on streaming because deltas
            // were already surfaced to the consumer); a debug log
            // makes the silent ignore visible at run time.
            tracing::debug!(
                validation_retries = self.config.validation_retries,
                "ChatModel::stream_typed ignores validation_retries — \
                 streaming + retry would emit a divergent second stream \
                 over already-surfaced deltas. Use complete_typed for \
                 the unified retry budget."
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
        apply_overrides(&mut request, ctx);
        request.response_format = Some(format);

        let invocation = StreamingModelInvocation::new(ModelInvocation::new(request, ctx.clone()));
        let model_stream = self.streaming_service().oneshot(invocation).await?;
        let ModelStream { stream, completion } = model_stream;

        let budget_for_completion = budget.clone();
        let short_name_owned = short_name.to_owned();
        let typed_completion = async move {
            let response = completion.await?;
            if let Some(budget) = &budget_for_completion {
                budget.observe_usage(&response.usage)?;
            }
            parse_typed_response::<O>(&short_name_owned, response)
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
            .field("layers", &self.layer_names)
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

/// Parse a structured-output response into `O`, wrapping any
/// schema-mismatch directly into [`Error::ModelRetry`] at the parse
/// site so the unified retry loop never sees a raw [`Error::Serde`]
/// for schema-mismatch failures (invariant 20). The hint flows through
/// the [`crate::llm_facing::RenderedForLlm`] funnel so the model
/// receives a corrective message routed by the operator (invariant 16).
///
/// [`Error::invalid_request`] still surfaces unchanged when the
/// response carried no parseable content — that is an `OutputStrategy`
/// configuration error, not a retry condition.
#[allow(clippy::needless_pass_by_value)]
fn parse_typed_response<O>(short_name: &str, response: ModelResponse) -> Result<O>
where
    O: serde::de::DeserializeOwned,
{
    use crate::llm_facing::LlmRenderable;
    let wrap = |e: serde_json::Error| -> Error {
        Error::model_retry(schema_mismatch_diagnostic(short_name, &e).for_llm(), 0)
    };
    for part in &response.content {
        if let ContentPart::ToolUse { input, .. } = part {
            return serde_json::from_value(input.clone()).map_err(wrap);
        }
    }
    for part in &response.content {
        if let ContentPart::Text { text, .. } = part {
            return serde_json::from_str(text).map_err(wrap);
        }
    }
    Err(Error::invalid_request(
        "complete_typed: model response carried neither a `tool_use` block nor a text \
         block — the configured `OutputStrategy` did not produce typed output",
    ))
}

/// Render a schema-mismatch parse error into a model-actionable hint.
/// Strips `serde_json::Error`'s trailing `at line N column M` position
/// noise — line / column offsets reference the raw bytes the parser
/// scanned and cannot help the model correct its output, but they leak
/// internal parser state into the LLM channel (invariant 16).
///
/// Schema-mismatch and validator-driven retries converge on the
/// returned text wrapped in the `RenderedForLlm` carrier (invariant 20).
fn schema_mismatch_diagnostic(short_name: &str, err: &serde_json::Error) -> String {
    let raw = err.to_string();
    let trimmed = raw
        .split(" at line ")
        .next()
        .unwrap_or(raw.as_str())
        .trim_end_matches('.');
    format!(
        "Your previous response did not match the required JSON schema for `{short_name}`. \
         Parser diagnostic: {trimmed}.\n\
         Re-emit the response as a single valid JSON object that conforms to the schema."
    )
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
