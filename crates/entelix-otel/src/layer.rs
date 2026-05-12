//! `OtelLayer` — `tower::Layer<S>` middleware emitting OpenTelemetry
//! GenAI semconv events around model and tool calls.
//!
//! Single layer struct; two `Service` impls — `Service<ModelInvocation>`
//! and `Service<ToolInvocation>` — emit the right `gen_ai.*` events
//! per pipeline. Compose via `ChatModel::layer(OtelLayer::new(...))`
//! and `ToolRegistry::layer(OtelLayer::new(...))`.
//!
//! Events are emitted via `tracing::event!` with dotted `gen_ai.*`
//! field names. When `tracing-opentelemetry` is wired (see
//! [`crate::init`]) those events become real OTel span events;
//! without OTel they remain plain tracing events visible to any
//! subscriber.

use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Instant;

use futures::future::BoxFuture;
use serde_json::Value;
use tower::{Layer, Service, ServiceExt};

use entelix_core::cost::{CostCalculator, ToolCostCalculator};
use entelix_core::error::{Error, Result};
use entelix_core::ir::{ModelResponse, StopReason};
use entelix_core::service::{
    ModelInvocation, ModelStream, StreamingModelInvocation, ToolInvocation,
};

use crate::metrics::{GenAiMetrics, OperationKind};

/// Default truncation cap for tool I/O capture (4 KiB). Chosen so
/// the typical `{"task": "..."}` / `{"output": "..."}` payload
/// surfaces in full while a 50 KB HTTP-fetch response truncates
/// before ballooning the span. Operators override via
/// [`OtelLayer::with_tool_io_capture`].
pub const DEFAULT_TOOL_IO_TRUNCATION: usize = 4096;

/// How the tool side of the layer surfaces `gen_ai.tool.input`
/// and `gen_ai.tool.output` on the dispatch span events.
///
/// Tool I/O routinely carries either large payloads (HTTP fetch
/// responses, tool-output transcripts) or sensitive content (PII,
/// API keys captured from elicitation, raw user inputs). Emitting
/// every byte to OTel by default would balloon span size in the
/// common case and leak in the sensitive case.
///
/// The default is [`Self::Truncated`] with a `4096`-byte cap —
/// operators get useful debug visibility without the worst-case
/// span bloat. Deployments with stricter PII policies switch to
/// [`Self::Off`]; deployments running their own PII redaction
/// upstream switch to [`Self::Full`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ToolIoCaptureMode {
    /// Omit `gen_ai.tool.input` / `gen_ai.tool.output` from the
    /// span events. The fields appear as `<omitted>` so operators
    /// reading the span see the policy was applied (vs. the
    /// attribute being absent because it was forgotten).
    Off,
    /// Truncate input / output to at most `max_bytes` bytes,
    /// appending a `… [truncated, N bytes total]` marker when
    /// the cap fires. The truncation is byte-based to keep the
    /// cap predictable; the marker is appended on UTF-8 char
    /// boundaries so the rendered string stays valid.
    Truncated {
        /// Inclusive byte cap before the truncation marker is
        /// appended.
        max_bytes: usize,
    },
    /// Emit input / output verbatim. Use when an upstream layer
    /// already redacts PII or when payload size is bounded by
    /// the deployment.
    Full,
}

impl Default for ToolIoCaptureMode {
    fn default() -> Self {
        Self::Truncated {
            max_bytes: DEFAULT_TOOL_IO_TRUNCATION,
        }
    }
}

/// Layer that emits GenAI semconv events around the wrapped
/// service. Cloning is cheap (`Arc`s internally).
#[derive(Clone)]
pub struct OtelLayer {
    /// `gen_ai.system` attribute value (provider name).
    system: Arc<str>,
    /// Optional metric handles. When `None` the layer only emits
    /// tracing events.
    metrics: Option<GenAiMetrics>,
    /// Optional cost calculator. When attached, the layer emits
    /// `gen_ai.usage.cost` on the response event for successful
    /// model invocations whose model has a pricing row.
    cost_calculator: Option<Arc<dyn CostCalculator>>,
    /// Optional tool cost calculator. When attached, the layer
    /// emits `gen_ai.tool.cost` on the success event for tool
    /// dispatches whose `(tenant, tool_name, output)` resolves to
    /// a price.
    tool_cost_calculator: Option<Arc<dyn ToolCostCalculator>>,
    /// How tool input / output payloads surface on the span
    /// events. Defaults to [`ToolIoCaptureMode::Truncated`] with
    /// the `4096`-byte cap from [`DEFAULT_TOOL_IO_TRUNCATION`].
    tool_io_capture: ToolIoCaptureMode,
}

impl OtelLayer {
    /// Patch-version-stable identifier surfaced through
    /// [`entelix_core::ChatModel::layer_names`] /
    /// `ToolRegistry::layer_names`. Renaming this constant is a
    /// breaking change for dashboards keyed off the value.
    pub const NAME: &'static str = "otel";

    /// Build with a system name (provider identifier — typically
    /// `"anthropic"`, `"openai"`, etc.).
    #[must_use]
    pub fn new(system: impl Into<Arc<str>>) -> Self {
        Self {
            system: system.into(),
            metrics: None,
            cost_calculator: None,
            tool_cost_calculator: None,
            tool_io_capture: ToolIoCaptureMode::default(),
        }
    }

    /// Attach a metric instrument set.
    #[must_use]
    pub fn with_metrics(mut self, metrics: GenAiMetrics) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Attach a [`CostCalculator`]. Once set, the layer emits
    /// `gen_ai.usage.cost` on every successful response event whose
    /// model resolves to a pricing row. Unknown models surface as
    /// the calculator's `None` and the attribute is omitted —
    /// silent zero would hide a missing-pricing-row deployment bug.
    ///
    /// The calculator is invoked **after** `inner.call` returns Ok,
    /// so a transport failure never produces a phantom cost.
    #[must_use]
    pub fn with_cost_calculator(mut self, calculator: Arc<dyn CostCalculator>) -> Self {
        self.cost_calculator = Some(calculator);
        self
    }

    /// Attach a [`ToolCostCalculator`]. Once set, the layer emits
    /// `gen_ai.tool.cost` on every successful tool dispatch whose
    /// `(tenant, tool_name, output)` resolves to a price. Tools
    /// with no pricing row produce `None` and the attribute is
    /// omitted, matching the model-cost discipline.
    #[must_use]
    pub fn with_tool_cost_calculator(mut self, calculator: Arc<dyn ToolCostCalculator>) -> Self {
        self.tool_cost_calculator = Some(calculator);
        self
    }

    /// Override the policy that governs how tool input / output
    /// surfaces on `gen_ai.tool.start` / `gen_ai.tool.end` span
    /// events. See [`ToolIoCaptureMode`] for the three modes
    /// (`Off`, `Truncated`, `Full`); the default is
    /// `Truncated { max_bytes: 4096 }`.
    #[must_use]
    pub const fn with_tool_io_capture(mut self, mode: ToolIoCaptureMode) -> Self {
        self.tool_io_capture = mode;
        self
    }
}

impl std::fmt::Debug for OtelLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OtelLayer")
            .field("system", &self.system.as_ref())
            .field("metrics_attached", &self.metrics.is_some())
            .field("cost_calculator_attached", &self.cost_calculator.is_some())
            .field(
                "tool_cost_calculator_attached",
                &self.tool_cost_calculator.is_some(),
            )
            .field("tool_io_capture", &self.tool_io_capture)
            .finish()
    }
}

impl entelix_core::NamedLayer for OtelLayer {
    fn layer_name(&self) -> &'static str {
        Self::NAME
    }
}

impl<S> Layer<S> for OtelLayer {
    type Service = OtelService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        OtelService {
            inner,
            system: Arc::clone(&self.system),
            metrics: self.metrics.clone(),
            cost_calculator: self.cost_calculator.clone(),
            tool_cost_calculator: self.tool_cost_calculator.clone(),
            tool_io_capture: self.tool_io_capture,
        }
    }
}

/// `Service` produced by [`OtelLayer`]. Specialised
/// `Service<ModelInvocation>` and `Service<ToolInvocation>` impls
/// below.
#[derive(Clone)]
pub struct OtelService<S> {
    inner: S,
    system: Arc<str>,
    metrics: Option<GenAiMetrics>,
    cost_calculator: Option<Arc<dyn CostCalculator>>,
    tool_cost_calculator: Option<Arc<dyn ToolCostCalculator>>,
    tool_io_capture: ToolIoCaptureMode,
}

impl<S> Service<ModelInvocation> for OtelService<S>
where
    S: Service<ModelInvocation, Response = ModelResponse, Error = Error> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = ModelResponse;
    type Error = Error;
    type Future = BoxFuture<'static, Result<ModelResponse>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<()>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, invocation: ModelInvocation) -> Self::Future {
        let inner = self.inner.clone();
        let system = Arc::clone(&self.system);
        let metrics = self.metrics.clone();
        let cost_calculator = self.cost_calculator.clone();
        Box::pin(async move {
            let request_model = invocation.request.model.clone();
            let max_tokens = invocation.request.max_tokens;
            let temperature = invocation.request.temperature;
            let top_p = invocation.request.top_p;
            let tenant = invocation.ctx.tenant_id().to_owned();
            let thread = invocation.ctx.thread_id().map(str::to_owned);
            let run = invocation.ctx.run_id().map(str::to_owned);
            let cancel_token = invocation.ctx.cancellation().clone();
            // Snapshot the ctx so the cost calculator can read tenant
            // / run-id even after `invocation` moves into `oneshot`.
            // `ExecutionContext: Clone` is cheap (Arc-backed cancel token).
            let ctx_for_cost = invocation.ctx.clone();
            let started_at = Instant::now();
            tracing::event!(
                target: "gen_ai",
                tracing::Level::INFO,
                gen_ai.system = %system,
                gen_ai.operation.name = OperationKind::Chat.as_str(),
                gen_ai.request.model = %request_model,
                gen_ai.request.max_tokens = max_tokens,
                gen_ai.request.temperature = temperature,
                gen_ai.request.top_p = top_p,
                entelix.tenant_id = %tenant,
                entelix.thread_id = thread,
                entelix.run_id = run,
                "gen_ai.request"
            );

            let result = inner.oneshot(invocation).await;
            let duration = started_at.elapsed();
            let duration_ms = u64::try_from(duration.as_millis()).unwrap_or(u64::MAX);
            let cancelled = cancel_token.is_cancelled();

            match &result {
                Ok(response) => {
                    let finish_reason = stop_reason_label(&response.stop_reason);
                    // Compute cost AFTER the response arrives — never
                    // on the error branch — so a failed call cannot
                    // produce a phantom charge in telemetry. The
                    // calculator receives the request `ctx` directly
                    // so multi-tenant calculators can read
                    // `tenant_id` and select per-tenant pricing.
                    let cost = if let Some(calc) = &cost_calculator {
                        calc.compute_cost(&response.model, &response.usage, &ctx_for_cost)
                            .await
                    } else {
                        None
                    };
                    tracing::event!(
                        target: "gen_ai",
                        tracing::Level::INFO,
                        gen_ai.system = %system,
                        gen_ai.operation.name = OperationKind::Chat.as_str(),
                        gen_ai.response.id = %response.id,
                        gen_ai.response.model = %response.model,
                        gen_ai.response.finish_reasons = finish_reason,
                        gen_ai.usage.input_tokens = response.usage.input_tokens,
                        gen_ai.usage.output_tokens = response.usage.output_tokens,
                        gen_ai.usage.cached_input_tokens = response.usage.cached_input_tokens,
                        gen_ai.usage.cache_creation_input_tokens =
                            response.usage.cache_creation_input_tokens,
                        gen_ai.usage.reasoning_tokens = response.usage.reasoning_tokens,
                        gen_ai.usage.cost = cost,
                        duration_ms,
                        entelix.cancelled = cancelled,
                        entelix.tenant_id = %tenant,
                        entelix.thread_id = thread,
                        entelix.run_id = run,
                        "gen_ai.response"
                    );
                    if let Some(metrics) = &metrics {
                        metrics.record_call(
                            &system,
                            OperationKind::Chat,
                            &request_model,
                            &response.model,
                            &response.usage,
                            duration,
                        );
                    }
                }
                Err(err) => {
                    tracing::event!(
                        target: "gen_ai",
                        tracing::Level::ERROR,
                        gen_ai.system = %system,
                        gen_ai.operation.name = OperationKind::Chat.as_str(),
                        gen_ai.request.model = %request_model,
                        error.message = %err,
                        duration_ms,
                        entelix.cancelled = cancelled,
                        entelix.tenant_id = %tenant,
                        entelix.thread_id = thread,
                        entelix.run_id = run,
                        "gen_ai.error"
                    );
                }
            }
            result
        })
    }
}

impl<S> Service<StreamingModelInvocation> for OtelService<S>
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

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<()>> {
        self.inner.poll_ready(cx)
    }

    // The streaming-side `Service::call` interleaves request-side
    // observability (vendor identifier emit + metric attribute
    // setup) with the post-stream completion-future wrap (cost
    // computation + token-usage histogram + terminal Ok / Err
    // event emit). Splitting the body into two helpers would
    // separate the metric-name SSoT from the call-site, which the
    // semconv discipline (entelix-otel CLAUDE.md) wants colocated.
    // The 127-line body is intentional.
    #[allow(clippy::too_many_lines)]
    fn call(&mut self, invocation: StreamingModelInvocation) -> Self::Future {
        let inner = self.inner.clone();
        let system = Arc::clone(&self.system);
        let metrics = self.metrics.clone();
        let cost_calculator = self.cost_calculator.clone();
        Box::pin(async move {
            let request_model = invocation.request().model.clone();
            let max_tokens = invocation.request().max_tokens;
            let temperature = invocation.request().temperature;
            let top_p = invocation.request().top_p;
            let tenant = invocation.ctx().tenant_id().to_owned();
            let thread = invocation.ctx().thread_id().map(str::to_owned);
            let run = invocation.ctx().run_id().map(str::to_owned);
            let cancel_token = invocation.ctx().cancellation().clone();
            let ctx_for_cost = invocation.ctx().clone();
            let started_at = Instant::now();
            tracing::event!(
                target: "gen_ai",
                tracing::Level::INFO,
                gen_ai.system = %system,
                gen_ai.operation.name = OperationKind::Chat.as_str(),
                gen_ai.request.model = %request_model,
                gen_ai.request.max_tokens = max_tokens,
                gen_ai.request.temperature = temperature,
                gen_ai.request.top_p = top_p,
                entelix.streaming = true,
                entelix.tenant_id = %tenant,
                entelix.thread_id = thread,
                entelix.run_id = run,
                "gen_ai.request"
            );

            // Open the stream first so initial-connection failures
            // surface as a typed `gen_ai.error` event before we
            // hand a `ModelStream` to the caller.
            let model_stream = match inner.oneshot(invocation).await {
                Ok(s) => s,
                Err(err) => {
                    let duration_ms =
                        u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX);
                    tracing::event!(
                        target: "gen_ai",
                        tracing::Level::ERROR,
                        gen_ai.system = %system,
                        gen_ai.operation.name = OperationKind::Chat.as_str(),
                        gen_ai.request.model = %request_model,
                        entelix.streaming = true,
                        error.message = %err,
                        duration_ms,
                        entelix.cancelled = cancel_token.is_cancelled(),
                        entelix.tenant_id = %tenant,
                        entelix.thread_id = thread,
                        entelix.run_id = run,
                        "gen_ai.error"
                    );
                    return Err(err);
                }
            };
            let ModelStream { stream, completion } = model_stream;

            // Wrap the completion future so cost / response events
            // emit on the `Ok` branch only. The contract: callers
            // that consume the stream to its terminal `Stop` MUST
            // await `completion` to receive both the aggregated
            // `ModelResponse` and the observability emission tied
            // to it. Layers do not spawn background tasks — that
            // would couple `OtelLayer` to a specific runtime
            // (tokio vs others) and silently produce events whose
            // ordering relative to caller drain is unstable.
            //
            // Callers that drop the stream without awaiting
            // `completion` lose the post-stream cost emission —
            // mirroring what `complete_full` would do if its
            // `oneshot(invocation)` future were dropped before
            // resolving. The transactional guarantee (invariant 12
            // — "no charge on the error branch") still holds: the
            // wrapped future never emits cost on `Err`, and
            // dropping the future before it resolves emits
            // nothing.
            let user_facing = async move {
                let result = completion.await;
                let duration_ms =
                    u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX);
                let cancelled = cancel_token.is_cancelled();
                match &result {
                    Ok(response) => {
                        let finish_reason = stop_reason_label(&response.stop_reason);
                        let cost = if let Some(calc) = &cost_calculator {
                            calc.compute_cost(&response.model, &response.usage, &ctx_for_cost)
                                .await
                        } else {
                            None
                        };
                        tracing::event!(
                            target: "gen_ai",
                            tracing::Level::INFO,
                            gen_ai.system = %system,
                            gen_ai.operation.name = OperationKind::Chat.as_str(),
                            gen_ai.response.id = %response.id,
                            gen_ai.response.model = %response.model,
                            gen_ai.response.finish_reasons = finish_reason,
                            gen_ai.usage.input_tokens = response.usage.input_tokens,
                            gen_ai.usage.output_tokens = response.usage.output_tokens,
                            gen_ai.usage.cached_input_tokens = response.usage.cached_input_tokens,
                            gen_ai.usage.cache_creation_input_tokens =
                                response.usage.cache_creation_input_tokens,
                            gen_ai.usage.reasoning_tokens = response.usage.reasoning_tokens,
                            gen_ai.usage.cost = cost,
                            entelix.streaming = true,
                            duration_ms,
                            entelix.cancelled = cancelled,
                            entelix.tenant_id = %tenant,
                            entelix.thread_id = thread,
                            entelix.run_id = run,
                            "gen_ai.response"
                        );
                        if let Some(metrics) = &metrics {
                            metrics.record_call(
                                &system,
                                OperationKind::Chat,
                                &request_model,
                                &response.model,
                                &response.usage,
                                started_at.elapsed(),
                            );
                        }
                    }
                    Err(err) => {
                        tracing::event!(
                            target: "gen_ai",
                            tracing::Level::ERROR,
                            gen_ai.system = %system,
                            gen_ai.operation.name = OperationKind::Chat.as_str(),
                            gen_ai.request.model = %request_model,
                            entelix.streaming = true,
                            error.message = %err,
                            duration_ms,
                            entelix.cancelled = cancelled,
                            entelix.tenant_id = %tenant,
                            entelix.thread_id = thread,
                            entelix.run_id = run,
                            "gen_ai.error"
                        );
                    }
                }
                result
            };
            Ok(ModelStream {
                stream,
                completion: Box::pin(user_facing),
            })
        })
    }
}

impl<S> Service<ToolInvocation> for OtelService<S>
where
    S: Service<ToolInvocation, Response = Value, Error = Error> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = Value;
    type Error = Error;
    type Future = BoxFuture<'static, Result<Value>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<()>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, invocation: ToolInvocation) -> Self::Future {
        let inner = self.inner.clone();
        let system = Arc::clone(&self.system);
        let tool_cost_calculator = self.tool_cost_calculator.clone();
        let io_capture = self.tool_io_capture;
        Box::pin(async move {
            let metadata = Arc::clone(&invocation.metadata);
            let tool_name = metadata.name.clone();
            let tool_version = metadata.version.clone();
            let tool_effect = metadata.effect.as_wire();
            let tool_idempotent = metadata.idempotent;
            let tool_retry_attempts = metadata.retry_hint.map(|h| h.max_attempts);
            let tool_use_id = invocation.tool_use_id.clone();
            let tenant = invocation.ctx.tenant_id().to_owned();
            let run = invocation.ctx.run_id().map(str::to_owned);
            let cancel_token = invocation.ctx.cancellation().clone();
            let input_str = capture_tool_payload(&invocation.input.to_string(), io_capture);
            // Snapshot ctx for the calculator path; `invocation`
            // moves into oneshot below.
            let ctx_for_cost = invocation.ctx.clone();
            tracing::event!(
                target: "gen_ai",
                tracing::Level::INFO,
                gen_ai.system = %system,
                gen_ai.operation.name = OperationKind::ExecuteTool.as_str(),
                gen_ai.tool.name = %tool_name,
                gen_ai.tool.version = tool_version,
                gen_ai.tool.effect = tool_effect,
                gen_ai.tool.idempotent = tool_idempotent,
                gen_ai.tool.retry.max_attempts = tool_retry_attempts,
                gen_ai.tool.call.id = %tool_use_id,
                gen_ai.tool.input = %input_str,
                entelix.tenant_id = %tenant,
                entelix.run_id = run,
                "gen_ai.tool.start"
            );
            let started_at = Instant::now();
            let result = inner.oneshot(invocation).await;
            let duration_ms = u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX);
            let cancelled = cancel_token.is_cancelled();
            match &result {
                Ok(output) => {
                    let output_str = capture_tool_payload(&output.to_string(), io_capture);
                    // Compute tool cost on the success path only —
                    // a failed tool call must never produce a
                    // phantom charge in telemetry.
                    let cost = if let Some(calc) = &tool_cost_calculator {
                        calc.compute_cost(&tool_name, output, &ctx_for_cost).await
                    } else {
                        None
                    };
                    tracing::event!(
                        target: "gen_ai",
                        tracing::Level::INFO,
                        gen_ai.system = %system,
                        gen_ai.operation.name = OperationKind::ExecuteTool.as_str(),
                        gen_ai.tool.name = %tool_name,
                        gen_ai.tool.effect = tool_effect,
                        gen_ai.tool.call.id = %tool_use_id,
                        gen_ai.tool.output = %output_str,
                        gen_ai.tool.cost = cost,
                        output_kind = output_kind_label(output),
                        duration_ms,
                        entelix.cancelled = cancelled,
                        entelix.tenant_id = %tenant,
                        entelix.run_id = run,
                        "gen_ai.tool.end"
                    );
                }
                Err(err) => {
                    tracing::event!(
                        target: "gen_ai",
                        tracing::Level::ERROR,
                        gen_ai.system = %system,
                        gen_ai.operation.name = OperationKind::ExecuteTool.as_str(),
                        gen_ai.tool.name = %tool_name,
                        gen_ai.tool.effect = tool_effect,
                        gen_ai.tool.call.id = %tool_use_id,
                        error.message = %err,
                        duration_ms,
                        entelix.cancelled = cancelled,
                        entelix.tenant_id = %tenant,
                        entelix.run_id = run,
                        "gen_ai.tool.error"
                    );
                }
            }
            result
        })
    }
}

fn stop_reason_label(reason: &StopReason) -> &str {
    match reason {
        StopReason::EndTurn => "end_turn",
        StopReason::MaxTokens => "max_tokens",
        StopReason::StopSequence { .. } => "stop_sequence",
        StopReason::ToolUse => "tool_use",
        StopReason::Refusal { .. } => "refusal",
        _ => "other",
    }
}

fn output_kind_label(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Apply [`ToolIoCaptureMode`] to a tool payload string.
///
/// `Off` returns the literal `"<omitted>"` so the attribute
/// stays present (operators reading the span see the policy
/// applied vs. the field being silently absent). `Truncated`
/// caps at the supplied byte count, snapping back to the
/// nearest UTF-8 char boundary so the marker concatenates
/// cleanly. `Full` returns the input verbatim.
fn capture_tool_payload(raw: &str, mode: ToolIoCaptureMode) -> String {
    match mode {
        ToolIoCaptureMode::Off => "<omitted>".to_owned(),
        ToolIoCaptureMode::Full => raw.to_owned(),
        ToolIoCaptureMode::Truncated { max_bytes } => {
            if raw.len() <= max_bytes {
                return raw.to_owned();
            }
            let mut cut = max_bytes;
            while cut > 0 && !raw.is_char_boundary(cut) {
                cut -= 1;
            }
            let truncated_total = raw.len();
            format!(
                "{prefix}… [truncated, {truncated_total} bytes total]",
                prefix = &raw[..cut]
            )
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use entelix_core::TenantId;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::task::Context as TaskContext;

    use entelix_core::ExecutionContext;
    use entelix_core::ir::{ContentPart, Message, ModelRequest, Usage};
    use serde_json::json;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::{Layer as TLayer, Registry};

    use super::*;

    #[test]
    fn capture_tool_payload_full_returns_input_verbatim() {
        let s = capture_tool_payload(r#"{"x":1}"#, ToolIoCaptureMode::Full);
        assert_eq!(s, r#"{"x":1}"#);
    }

    #[test]
    fn capture_tool_payload_off_returns_omitted_marker() {
        let s = capture_tool_payload(r#"{"sensitive":"value"}"#, ToolIoCaptureMode::Off);
        assert_eq!(s, "<omitted>");
    }

    #[test]
    fn capture_tool_payload_truncated_under_cap_passes_through() {
        let s = capture_tool_payload("small", ToolIoCaptureMode::Truncated { max_bytes: 100 });
        assert_eq!(s, "small");
    }

    #[test]
    fn capture_tool_payload_truncated_over_cap_appends_marker() {
        let raw = "x".repeat(20);
        let s = capture_tool_payload(&raw, ToolIoCaptureMode::Truncated { max_bytes: 5 });
        assert!(s.starts_with("xxxxx"));
        assert!(s.contains("[truncated, 20 bytes total]"));
    }

    #[test]
    fn capture_tool_payload_truncated_snaps_to_utf8_boundary() {
        // 4-byte char "🚀" at offset 5 — cap of 6 bytes lands
        // mid-char and must snap back to byte 5.
        let raw = "hello🚀world";
        let s = capture_tool_payload(raw, ToolIoCaptureMode::Truncated { max_bytes: 6 });
        // Truncation marker still appended; prefix is "hello"
        // (5 bytes), not "hello\xF0" which would be invalid.
        assert!(s.starts_with("hello…"));
        assert!(s.contains("bytes total"));
    }

    #[test]
    fn default_tool_io_capture_mode_is_truncated_4096() {
        match ToolIoCaptureMode::default() {
            ToolIoCaptureMode::Truncated { max_bytes } => {
                assert_eq!(max_bytes, DEFAULT_TOOL_IO_TRUNCATION);
                assert_eq!(DEFAULT_TOOL_IO_TRUNCATION, 4096);
            }
            other => panic!("default mode must be Truncated, got {other:?}"),
        }
    }

    /// Capturing tracing layer used to assert events fired.
    #[derive(Clone, Default)]
    struct Capture(Arc<Mutex<Vec<String>>>);

    impl<S: tracing::Subscriber> TLayer<S> for Capture {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            struct V<'a>(&'a mut String);
            impl tracing::field::Visit for V<'_> {
                fn record_debug(
                    &mut self,
                    field: &tracing::field::Field,
                    value: &dyn std::fmt::Debug,
                ) {
                    use std::fmt::Write;
                    let _ = write!(self.0, " {}={value:?}", field.name());
                }
            }
            let mut buf = format!("[{}]", event.metadata().target());
            event.record(&mut V(&mut buf));
            self.0.lock().unwrap().push(buf);
        }
    }

    fn set_capture() -> (Capture, tracing::subscriber::DefaultGuard) {
        let cap = Capture::default();
        let subscriber = Registry::default().with(cap.clone());
        let guard = tracing::subscriber::set_default(subscriber);
        (cap, guard)
    }

    #[derive(Clone)]
    struct CountingModelService(Arc<AtomicU32>, ModelResponse);

    impl Service<ModelInvocation> for CountingModelService {
        type Response = ModelResponse;
        type Error = Error;
        type Future = BoxFuture<'static, Result<ModelResponse>>;
        fn poll_ready(&mut self, _: &mut TaskContext<'_>) -> Poll<Result<()>> {
            Poll::Ready(Ok(()))
        }
        fn call(&mut self, _: ModelInvocation) -> Self::Future {
            self.0.fetch_add(1, Ordering::SeqCst);
            let resp = self.1.clone();
            Box::pin(async move { Ok(resp) })
        }
    }

    fn make_request(model: &str) -> ModelRequest {
        ModelRequest {
            model: model.into(),
            messages: vec![Message::user("hi")],
            max_tokens: Some(64),
            temperature: Some(0.7),
            ..ModelRequest::default()
        }
    }

    fn make_response() -> ModelResponse {
        ModelResponse {
            id: "r1".into(),
            model: "claude-opus-4-7".into(),
            stop_reason: StopReason::EndTurn,
            content: vec![ContentPart::text("ok")],
            usage: Usage::new(100, 50),
            rate_limit: None,
            warnings: Vec::new(),
            provider_echoes: Vec::new(),
        }
    }

    #[tokio::test]
    async fn model_layer_emits_request_and_response_events() {
        let (cap, _g) = set_capture();
        let layer = OtelLayer::new("anthropic");
        let leaf = CountingModelService(Arc::new(AtomicU32::new(0)), make_response());
        let svc = layer.layer(leaf);
        let inv = ModelInvocation::new(
            make_request("claude-opus-4-7"),
            ExecutionContext::new()
                .with_tenant_id(TenantId::new("acme"))
                .with_thread_id("conv-1"),
        );
        let _ = tower::ServiceExt::oneshot(svc, inv).await.unwrap();
        let combined = cap.0.lock().unwrap().join("\n");
        assert!(combined.contains("gen_ai.request"), "{combined}");
        assert!(combined.contains("gen_ai.response"), "{combined}");
        assert!(combined.contains("anthropic"), "{combined}");
    }

    /// Stub calculator that returns a fixed cost for a known model
    /// and `None` otherwise — exercises both branches without a
    /// pricing-table dependency.
    struct StubCostCalculator {
        known_model: String,
        cost: f64,
    }

    #[async_trait::async_trait]
    impl entelix_core::cost::CostCalculator for StubCostCalculator {
        async fn compute_cost(
            &self,
            model: &str,
            _usage: &entelix_core::ir::Usage,
            _ctx: &entelix_core::ExecutionContext,
        ) -> Option<f64> {
            (model == self.known_model).then_some(self.cost)
        }
    }

    #[tokio::test]
    async fn model_layer_emits_gen_ai_usage_cost_when_calculator_is_attached() {
        let (cap, _g) = set_capture();
        let calc = Arc::new(StubCostCalculator {
            known_model: "claude-opus-4-7".to_owned(),
            cost: 0.0125,
        });
        let layer = OtelLayer::new("anthropic").with_cost_calculator(calc);
        let leaf = CountingModelService(Arc::new(AtomicU32::new(0)), make_response());
        let svc = layer.layer(leaf);
        let inv = ModelInvocation::new(
            make_request("claude-opus-4-7"),
            ExecutionContext::new().with_tenant_id(TenantId::new("acme")),
        );
        let _ = tower::ServiceExt::oneshot(svc, inv).await.unwrap();
        let combined = cap.0.lock().unwrap().join("\n");
        assert!(combined.contains("gen_ai.response"), "{combined}");
        assert!(combined.contains("gen_ai.usage.cost"), "{combined}");
        assert!(combined.contains("0.0125"), "{combined}");
    }

    #[tokio::test]
    async fn model_layer_omits_cost_attribute_for_unknown_model() {
        // Calculator is attached but does not know this model — the
        // attribute must NOT appear (silent zero would hide a
        // missing-pricing-row deployment bug).
        let (cap, _g) = set_capture();
        let calc = Arc::new(StubCostCalculator {
            known_model: "different-model".to_owned(),
            cost: 99.0,
        });
        let layer = OtelLayer::new("anthropic").with_cost_calculator(calc);
        let leaf = CountingModelService(Arc::new(AtomicU32::new(0)), make_response());
        let svc = layer.layer(leaf);
        let inv = ModelInvocation::new(
            make_request("claude-opus-4-7"),
            ExecutionContext::new().with_tenant_id(TenantId::new("acme")),
        );
        let _ = tower::ServiceExt::oneshot(svc, inv).await.unwrap();
        let combined = cap.0.lock().unwrap().join("\n");
        assert!(combined.contains("gen_ai.response"), "{combined}");
        assert!(
            !combined.contains("gen_ai.usage.cost=99"),
            "unknown model must not be charged: {combined}"
        );
    }

    fn lookup_invocation(tool_use_id: &str, input: Value, ctx: ExecutionContext) -> ToolInvocation {
        ToolInvocation::new(
            tool_use_id.to_owned(),
            Arc::new(entelix_core::tools::ToolMetadata::function(
                "lookup",
                "look up a record",
                json!({"type": "object"}),
            )),
            input,
            ctx,
        )
    }

    #[derive(Clone)]
    struct EchoToolService;

    impl Service<ToolInvocation> for EchoToolService {
        type Response = Value;
        type Error = Error;
        type Future = BoxFuture<'static, Result<Value>>;
        fn poll_ready(&mut self, _: &mut TaskContext<'_>) -> Poll<Result<()>> {
            Poll::Ready(Ok(()))
        }
        fn call(&mut self, inv: ToolInvocation) -> Self::Future {
            Box::pin(async move { Ok(inv.input) })
        }
    }

    #[tokio::test]
    async fn tool_layer_emits_start_and_end_events() {
        let (cap, _g) = set_capture();
        let layer = OtelLayer::new("anthropic");
        let svc = layer.layer(EchoToolService);
        let inv = lookup_invocation(
            "tool_use_1",
            json!({"id": "x"}),
            ExecutionContext::new().with_tenant_id(TenantId::new("acme")),
        );
        let _ = tower::ServiceExt::oneshot(svc, inv).await.unwrap();
        let combined = cap.0.lock().unwrap().join("\n");
        assert!(combined.contains("gen_ai.tool.start"), "{combined}");
        assert!(combined.contains("gen_ai.tool.end"), "{combined}");
        assert!(combined.contains("execute_tool"), "{combined}");
        assert!(combined.contains("lookup"), "{combined}");
    }

    #[tokio::test]
    async fn model_layer_emits_run_id_attribute() {
        let (cap, _g) = set_capture();
        let layer = OtelLayer::new("anthropic");
        let leaf = CountingModelService(Arc::new(AtomicU32::new(0)), make_response());
        let svc = layer.layer(leaf);
        let inv = ModelInvocation::new(
            make_request("claude-opus-4-7"),
            ExecutionContext::new()
                .with_tenant_id(TenantId::new("acme"))
                .with_run_id("run-abc-123"),
        );
        let _ = tower::ServiceExt::oneshot(svc, inv).await.unwrap();
        let combined = cap.0.lock().unwrap().join("\n");
        assert!(
            combined.contains("entelix.run_id") && combined.contains("run-abc-123"),
            "{combined}"
        );
    }

    #[derive(Clone)]
    struct FailingModelService;

    impl Service<ModelInvocation> for FailingModelService {
        type Response = ModelResponse;
        type Error = Error;
        type Future = BoxFuture<'static, Result<ModelResponse>>;
        fn poll_ready(&mut self, _: &mut TaskContext<'_>) -> Poll<Result<()>> {
            Poll::Ready(Ok(()))
        }
        fn call(&mut self, _inv: ModelInvocation) -> Self::Future {
            Box::pin(async move { Err(Error::config("provider unavailable")) })
        }
    }

    #[tokio::test]
    async fn model_layer_emits_duration_ms_on_error_branch() {
        let (cap, _g) = set_capture();
        let layer = OtelLayer::new("anthropic");
        let svc = layer.layer(FailingModelService);
        let inv = ModelInvocation::new(
            make_request("claude-opus-4-7"),
            ExecutionContext::new().with_tenant_id(TenantId::new("acme")),
        );
        let _ = tower::ServiceExt::oneshot(svc, inv).await.unwrap_err();
        let combined = cap.0.lock().unwrap().join("\n");
        assert!(combined.contains("gen_ai.error"), "{combined}");
        assert!(combined.contains("duration_ms"), "{combined}");
    }

    #[derive(Clone)]
    struct FailingToolService;

    impl Service<ToolInvocation> for FailingToolService {
        type Response = Value;
        type Error = Error;
        type Future = BoxFuture<'static, Result<Value>>;
        fn poll_ready(&mut self, _: &mut TaskContext<'_>) -> Poll<Result<()>> {
            Poll::Ready(Ok(()))
        }
        fn call(&mut self, _inv: ToolInvocation) -> Self::Future {
            Box::pin(async move { Err(Error::config("tool blew up")) })
        }
    }

    #[tokio::test]
    async fn tool_layer_emits_duration_ms_on_both_branches() {
        // Success branch
        let (ok_cap, _ok_g) = set_capture();
        let layer = OtelLayer::new("anthropic");
        let svc = layer.layer(EchoToolService);
        let inv = lookup_invocation(
            "tu-ok",
            json!({}),
            ExecutionContext::new().with_tenant_id(TenantId::new("acme")),
        );
        let _ = tower::ServiceExt::oneshot(svc, inv).await.unwrap();
        let combined_ok = ok_cap.0.lock().unwrap().join("\n");
        assert!(
            combined_ok.contains("gen_ai.tool.end") && combined_ok.contains("duration_ms"),
            "{combined_ok}"
        );

        // Error branch
        let (err_cap, _err_g) = set_capture();
        let layer = OtelLayer::new("anthropic");
        let svc = layer.layer(FailingToolService);
        let inv = lookup_invocation(
            "tu-err",
            json!({}),
            ExecutionContext::new().with_tenant_id(TenantId::new("acme")),
        );
        let _ = tower::ServiceExt::oneshot(svc, inv).await.unwrap_err();
        let combined_err = err_cap.0.lock().unwrap().join("\n");
        assert!(
            combined_err.contains("gen_ai.tool.error") && combined_err.contains("duration_ms"),
            "{combined_err}"
        );
    }

    /// Stub tool-cost calculator that returns a fixed cost for a
    /// known tool name and `None` for everything else.
    struct StubToolCostCalculator {
        known_tool: String,
        cost: f64,
    }

    #[async_trait::async_trait]
    impl entelix_core::cost::ToolCostCalculator for StubToolCostCalculator {
        async fn compute_cost(
            &self,
            tool_name: &str,
            _output: &Value,
            _ctx: &entelix_core::ExecutionContext,
        ) -> Option<f64> {
            (tool_name == self.known_tool).then_some(self.cost)
        }
    }

    #[tokio::test]
    async fn tool_layer_emits_gen_ai_tool_cost_when_calculator_attached() {
        let (cap, _g) = set_capture();
        let calc = Arc::new(StubToolCostCalculator {
            known_tool: "lookup".to_owned(),
            cost: 0.0007,
        });
        let layer = OtelLayer::new("anthropic").with_tool_cost_calculator(calc);
        let svc = layer.layer(EchoToolService);
        let inv = lookup_invocation(
            "tu-cost-ok",
            json!({"x": 1}),
            ExecutionContext::new().with_tenant_id(TenantId::new("acme")),
        );
        let _ = tower::ServiceExt::oneshot(svc, inv).await.unwrap();
        let combined = cap.0.lock().unwrap().join("\n");
        assert!(combined.contains("gen_ai.tool.end"), "{combined}");
        assert!(combined.contains("gen_ai.tool.cost"), "{combined}");
        assert!(combined.contains("0.0007"), "{combined}");
    }

    #[tokio::test]
    async fn tool_layer_omits_cost_attribute_for_unknown_tool() {
        // Calculator attached but the dispatched tool is not in its
        // pricing — the attribute must NOT appear (silent zero
        // would hide a missing-pricing-row deployment bug).
        let (cap, _g) = set_capture();
        let calc = Arc::new(StubToolCostCalculator {
            known_tool: "different-tool".to_owned(),
            cost: 99.0,
        });
        let layer = OtelLayer::new("anthropic").with_tool_cost_calculator(calc);
        let svc = layer.layer(EchoToolService);
        let inv = lookup_invocation(
            "tu-cost-unknown",
            json!({}),
            ExecutionContext::new().with_tenant_id(TenantId::new("acme")),
        );
        let _ = tower::ServiceExt::oneshot(svc, inv).await.unwrap();
        let combined = cap.0.lock().unwrap().join("\n");
        assert!(combined.contains("gen_ai.tool.end"), "{combined}");
        assert!(
            !combined.contains("gen_ai.tool.cost=99"),
            "unknown tool must not be charged: {combined}"
        );
    }

    #[tokio::test]
    async fn model_layer_does_not_emit_cost_on_error_path() {
        // Symmetric to `tool_layer_does_not_emit_cost_on_error_path`:
        // a failing model invocation must never produce a cost
        // attribute, regardless of whether a cost calculator is
        // attached. Phantom charges on errors are exactly what
        // Invariant 12 (transactional cost charging) prohibits.
        let (cap, _g) = set_capture();
        let calc = Arc::new(StubCostCalculator {
            known_model: "claude-opus-4-7".to_owned(),
            cost: 0.0125,
        });
        let layer = OtelLayer::new("anthropic").with_cost_calculator(calc);
        let svc = layer.layer(FailingModelService);
        let inv = ModelInvocation::new(
            make_request("claude-opus-4-7"),
            ExecutionContext::new().with_tenant_id(TenantId::new("acme")),
        );
        let _ = tower::ServiceExt::oneshot(svc, inv).await.unwrap_err();
        let combined = cap.0.lock().unwrap().join("\n");
        assert!(combined.contains("gen_ai.error"), "{combined}");
        assert!(
            !combined.contains("gen_ai.usage.cost"),
            "no cost attribute on error path: {combined}"
        );
    }

    #[tokio::test]
    async fn tool_layer_does_not_emit_cost_on_error_path() {
        // A failing tool dispatch must never produce a cost
        // attribute — phantom charges on errors are exactly what
        // F4 (transactional cost charging) prohibits.
        let (cap, _g) = set_capture();
        let calc = Arc::new(StubToolCostCalculator {
            known_tool: "lookup".to_owned(),
            cost: 0.0007,
        });
        let layer = OtelLayer::new("anthropic").with_tool_cost_calculator(calc);
        let svc = layer.layer(FailingToolService);
        let inv = lookup_invocation(
            "tu-cost-err",
            json!({}),
            ExecutionContext::new().with_tenant_id(TenantId::new("acme")),
        );
        let _ = tower::ServiceExt::oneshot(svc, inv).await.unwrap_err();
        let combined = cap.0.lock().unwrap().join("\n");
        assert!(combined.contains("gen_ai.tool.error"), "{combined}");
        assert!(
            !combined.contains("gen_ai.tool.cost"),
            "no cost attribute on error path: {combined}"
        );
    }

    #[tokio::test]
    async fn tool_layer_emits_run_id_and_tool_use_id_attributes() {
        let (cap, _g) = set_capture();
        let layer = OtelLayer::new("anthropic");
        let svc = layer.layer(EchoToolService);
        let inv = lookup_invocation(
            "tu-7",
            json!({}),
            ExecutionContext::new()
                .with_tenant_id(TenantId::new("acme"))
                .with_run_id("run-xyz"),
        );
        let _ = tower::ServiceExt::oneshot(svc, inv).await.unwrap();
        let combined = cap.0.lock().unwrap().join("\n");
        assert!(combined.contains("entelix.run_id") && combined.contains("run-xyz"));
        assert!(combined.contains("gen_ai.tool.call.id") && combined.contains("tu-7"));
    }
}
