//! `Agent<S>` — production agent runtime wrapping any `Runnable<S, S>`
//! with an event sink, execution mode, and lifecycle observers.
//!
//! `Agent<S>` is the surface every production caller targets:
//!
//! ```ignore
//! let agent = create_react_agent(model, tools)?;
//! let final_state = agent.execute(initial, &ctx).await?;  // sync drain
//!
//! let mut stream = agent.execute_stream(initial, &ctx);
//! while let Some(event) = stream.next().await {
//!     match event? { /* AgentEvent variants */ }
//! }
//! ```
//!
//! The runtime is deliberately a *thin wrapper* over the inner
//! `Runnable<S, S>` (typically a `CompiledGraph<S>`):
//!
//! - **`execute`** drives the inner runnable via
//!   [`Runnable::invoke`] and emits a `Started` / `Complete` pair
//!   on the sink for observability — returns the terminal state.
//! - **`execute_stream`** uses the same lifecycle and inner
//!   primitive (`Runnable::invoke`) but returns a stream of
//!   `AgentEvent` values for callers wiring to SSE or other
//!   incremental consumers. Both surfaces observe one canonical
//!   `Started` → `Complete(state)` sequence regardless of which
//!   the caller picks.
//!
//! ## Composability
//!
//! `Agent<S>` itself implements `Runnable<S, S>`, so an agent is a
//! valid node in a parent `StateGraph<ParentState>` that maps state
//! across the boundary. Recursive sub-agent dispatch follows the
//! same pattern as any other `Runnable` composition.
//!
//! ## Information density
//!
//! Events emitted to the sink carry the full observability surface
//! (ids, durations, classifications). The agent's own re-feed path
//! to the model — when consuming a `ToolComplete` to synthesize the
//! next message — reads only the LLM-facing fields (`output`,
//! `delta`, lean `error_message`). Test fixtures verify both
//! surfaces independently.

mod approval_layer;
mod approver;
mod event;
mod mode;
mod observer;
mod result;
mod sink;
mod tool_event_layer;
mod tool_hook_layer;

use std::sync::Arc;

use async_trait::async_trait;
use entelix_core::context::ExecutionContext;
use entelix_core::error::{Error, Result};
use entelix_runnable::Runnable;
use entelix_runnable::stream::BoxStream;
use tracing::Instrument;

pub use self::approval_layer::{
    ApprovalLayer, ApprovalService, EffectGate, ToolApprovalEventSink, ToolApprovalEventSinkHandle,
};
pub use self::approver::{
    AlwaysApprove, ApprovalDecision, ApprovalRequest, Approver, ChannelApprover,
    ChannelApproverConfig, PendingApproval,
};
pub use self::event::AgentEvent;
pub use self::mode::ExecutionMode;
pub use self::observer::{AgentObserver, DynObserver};
pub use self::result::AgentRunResult;
pub use self::sink::{
    AgentEventSink, BroadcastSink, CaptureSink, ChannelSink, DroppingSink, FailOpenSink, FanOutSink,
};
pub use self::tool_event_layer::{ToolEventLayer, ToolEventService};
pub use self::tool_hook_layer::{
    ToolHook, ToolHookDecision, ToolHookLayer, ToolHookRegistry, ToolHookRequest, ToolHookService,
};

/// Production agent runtime.
///
/// Construct via [`Agent::builder`]; finalize with
/// [`AgentBuilder::build`]. See module docs for the abstraction
/// model and information-density discipline.
pub struct Agent<S>
where
    S: Clone + Send + Sync + 'static,
{
    name: String,
    runnable: Arc<dyn Runnable<S, S>>,
    sink: Arc<dyn AgentEventSink<S>>,
    observers: Vec<DynObserver<S>>,
    execution_mode: ExecutionMode,
    approver: Option<Arc<dyn Approver>>,
}

impl<S> Agent<S>
where
    S: Clone + Send + Sync + 'static,
{
    /// Start a fluent builder.
    #[must_use]
    pub fn builder() -> AgentBuilder<S> {
        AgentBuilder::default()
    }

    /// Borrow the agent's stable name. Always non-empty —
    /// `AgentBuilder::build` rejects empty / unset names so trace
    /// correlation never silently breaks.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Borrow the underlying runnable. Useful when an agent is
    /// embedded as a node in a larger graph and the parent needs
    /// direct access to the inner shape (e.g. for checkpointing).
    #[must_use]
    pub fn inner(&self) -> &Arc<dyn Runnable<S, S>> {
        &self.runnable
    }

    /// Run to completion, returning the terminal state.
    ///
    /// Emits a `Started{run_id}` opener, fires every registered
    /// observer at the appropriate lifecycle point, then emits
    /// either `Complete{run_id, state}` or `Failed{run_id, error}`
    /// on the sink — every run produces exactly one terminal event
    ///.
    ///
    /// The `run_id` is inherited from `ctx.run_id()` when present,
    /// otherwise a fresh UUID v7 is generated and propagated to the
    /// inner runnable through a cloned context.
    pub async fn execute(&self, input: S, ctx: &ExecutionContext) -> Result<AgentRunResult<S>> {
        let (run_id, scoped_ctx) = Self::scoped_run_context(ctx);
        let parent_run_id = scoped_ctx.parent_run_id().map(str::to_owned);
        let ctx = &scoped_ctx;

        // Attach the type-erased approval-event sink handle so any
        // `ApprovalLayer` deeper in the dispatch stack can emit
        // `ToolCallApproved` / `ToolCallDenied` through the agent's
        // typed sink without taking it as a constructor arg
        // (which would tie the layer to a specific `S`). The
        // attachment is unconditional — operators without an
        // `ApprovalLayer` pay only the cost of an unused
        // `Extensions` slot.
        let scoped_with_sink =
            ctx.clone()
                .add_extension(ToolApprovalEventSinkHandle::for_agent_sink(Arc::clone(
                    &self.sink,
                )));

        self.execute_inner(input, run_id, parent_run_id, &scoped_with_sink)
            .await
    }

    /// Convenience entry that attaches a [`entelix_core::RunOverrides`] extension
    /// to the context for the duration of the call. Equivalent to
    /// `agent.execute(input, &ctx.add_extension(overrides))` —
    /// shorter at the call site and signals the per-call shape at
    /// a glance.
    ///
    /// **Asymmetric by design** — `RunOverrides` is the
    /// agent-loop carrier (model / system prompt / max iterations
    /// owned by the loop the `Agent` itself drives) so a typed
    /// convenience belongs on `Agent`. `RequestOverrides`
    /// (temperature / top_p / top_k / max_tokens / stop_sequences /
    /// reasoning_effort / tool_choice / response_format /
    /// parallel_tool_calls) is `ChatModel`-shaped — the dispatch
    /// layer downstream picks it up via
    /// `ExecutionContext::add_extension(RequestOverrides::new()…)`,
    /// no agent-side convenience is needed (and adding one would
    /// duplicate the orthogonality that the carrier split S99
    /// established).
    ///
    /// Operators threading multiple per-call extensions stay on
    /// [`Self::execute`] with their own `add_extension` chain.
    pub async fn execute_with(
        &self,
        input: S,
        overrides: entelix_core::RunOverrides,
        ctx: &ExecutionContext,
    ) -> Result<AgentRunResult<S>> {
        let scoped = ctx.clone().add_extension(overrides);
        self.execute(input, &scoped).await
    }

    async fn execute_inner(
        &self,
        input: S,
        run_id: String,
        parent_run_id: Option<String>,
        ctx: &ExecutionContext,
    ) -> Result<AgentRunResult<S>> {
        self.sink
            .send(AgentEvent::Started {
                run_id: run_id.clone(),
                parent_run_id: parent_run_id.clone(),
                agent: self.name.clone(),
            })
            .await?;

        // The model + tool service-layer spans fire inside
        // `run_inner`. Instrumenting *that* future with the
        // agent-run span makes those layer spans children of the
        // agent root in the OTel trace tree — operators see one
        // tree per agent run instead of layer spans floating
        // side by side. `Started` / `Failed` / `Complete` are
        // sink emissions, not tracing spans, so they live
        // outside the instrumented future without losing
        // observability.
        let outcome = self
            .run_inner(input, run_id.clone(), ctx)
            .instrument(self.run_span(&run_id, ctx))
            .await;
        match outcome {
            Ok(result) => {
                self.sink
                    .send(AgentEvent::Complete {
                        run_id: result.run_id.clone(),
                        state: result.state.clone(),
                        usage: result.usage,
                    })
                    .await?;
                Ok(result)
            }
            Err(err) => {
                if let entelix_core::Error::UsageLimitExceeded(breach) = &err
                    && let Some(handle) = ctx.audit_sink()
                {
                    handle.as_sink().record_usage_limit_exceeded(breach);
                }
                // Best-effort `Failed` emission — if the sink itself
                // errors (dropped receiver), swallow the secondary
                // error so the original surfaces unchanged.
                let _ = self
                    .sink
                    .send(AgentEvent::Failed {
                        run_id,
                        error: err.to_string(),
                        wire_code: err.wire_code(),
                        wire_class: err.wire_class(),
                    })
                    .await;
                Err(err)
            }
        }
    }

    /// Build the `entelix.agent.run` tracing span for one
    /// `execute` / `execute_stream` invocation. Span fields use
    /// the `gen_ai.agent.*` / `entelix.*` namespaces that match
    /// the rest of the OTel surface so dashboards joining on
    /// run_id stay consistent across the layer / agent stack.
    ///
    /// The six `gen_ai.usage.*` / `entelix.usage.*` fields are
    /// declared as [`tracing::field::Empty`] placeholders and
    /// populated by [`Self::run_inner`] at the
    /// [`AgentRunResult::usage`] freeze point — a `RunBudget`
    /// attached to the [`ExecutionContext`] surfaces its frozen
    /// counters as span attributes so OTel dashboards filter
    /// per-run consumption without consumers having to harvest
    /// the envelope return value. Runs without a budget keep the
    /// fields `Empty`; the `tracing-opentelemetry` bridge omits
    /// empty fields from the exported span attributes.
    fn run_span(&self, run_id: &str, ctx: &ExecutionContext) -> tracing::Span {
        tracing::info_span!(
            target: "gen_ai",
            "entelix.agent.run",
            gen_ai.agent.name = %self.name,
            entelix.run_id = %run_id,
            entelix.tenant_id = %ctx.tenant_id(),
            entelix.thread_id = ctx.thread_id(),
            gen_ai.usage.input_tokens = tracing::field::Empty,
            gen_ai.usage.output_tokens = tracing::field::Empty,
            gen_ai.usage.total_tokens = tracing::field::Empty,
            entelix.agent.usage.cost = tracing::field::Empty,
            entelix.usage.requests = tracing::field::Empty,
            entelix.usage.tool_calls = tracing::field::Empty,
        )
    }

    /// Drive observers + inner runnable as one cohesive unit so the
    /// outer terminal-event branching (`Complete` vs `Failed`) only
    /// matches once.
    ///
    /// The [`AgentRunResult::usage`] snapshot is captured *between*
    /// `runnable.invoke` returning and `on_complete` firing —
    /// observer dispatches may themselves consume budget through
    /// downstream `ChatModel` calls (memory consolidation, summary
    /// writes), and the envelope must reflect the agent run only
    ///.
    async fn run_inner(
        &self,
        input: S,
        run_id: String,
        ctx: &ExecutionContext,
    ) -> Result<AgentRunResult<S>> {
        for observer in &self.observers {
            observer.pre_turn(&input, ctx).await?;
        }
        let state = match self.runnable.invoke(input, ctx).await {
            Ok(state) => state,
            Err(err) => {
                // HITL pause-and-resume is a control signal, not a
                // failure — observers wanting interrupt observation
                // consume `AgentEvent::Interrupted` from the sink.
                if !matches!(err, Error::Interrupted { .. }) {
                    for observer in &self.observers {
                        // Failure-path observability is one-way:
                        // observer errors raised from on_error get
                        // dropped (with a tracing warn) so they don't
                        // replace the original error in flight.
                        // Mirrors the audit-sink contract from
                        // invariant 18 /
                        if let Err(observer_err) = observer.on_error(&err, ctx).await {
                            tracing::warn!(
                                observer = %observer.name(),
                                source = %observer_err,
                                "AgentObserver::on_error returned an error; dropping"
                            );
                        }
                    }
                }
                return Err(err);
            }
        };
        let usage = ctx.run_budget().map(|budget| budget.snapshot());
        // Mirror the frozen snapshot onto the `entelix.agent.run`
        // span declared in `run_span` — `Span::current()` here is
        // that root span (this future is `.instrument`-ed by
        // `execute_inner` / `book_end_stream`). Runs without a
        // budget leave the fields as `tracing::field::Empty`, which
        // the `tracing-opentelemetry` bridge drops from the
        // exported span (no zero-valued attributes ride through).
        if let Some(snapshot) = usage {
            let span = tracing::Span::current();
            span.record("gen_ai.usage.input_tokens", snapshot.input_tokens);
            span.record("gen_ai.usage.output_tokens", snapshot.output_tokens);
            span.record("gen_ai.usage.total_tokens", snapshot.total_tokens());
            // `Decimal` lacks a `tracing::Value` impl; serialise to
            // string. OTel exporters round-trip the cost as a
            // string-typed attribute (the `Decimal` representation
            // is the most accurate; consumers parse on read).
            //
            // Attribute name `entelix.agent.usage.cost` is distinct
            // from `OtelLayer`'s per-model-call `gen_ai.usage.cost`
            // — that one is a per-charge increment (`f64`), this one
            // is the per-run cumulative roll-up (`Decimal` Display).
            // Same key would conflate two different metrics on the
            // dashboard; the namespace split mirrors the existing
            // `entelix.usage.requests` / `entelix.usage.tool_calls`
            // pattern (per-run aggregates ride the `entelix.*`
            // namespace; per-call vendor signals ride `gen_ai.*`).
            span.record(
                "entelix.agent.usage.cost",
                tracing::field::display(&snapshot.cost_usd),
            );
            span.record("entelix.usage.requests", snapshot.requests);
            span.record("entelix.usage.tool_calls", snapshot.tool_calls);
        }
        // on_complete fires before any terminal event so observers
        // can mutate side-channel state (vector store writes,
        // summary persistence) and have those writes reflected in
        // the same audit trail row.
        for observer in &self.observers {
            observer.on_complete(&state, ctx).await?;
        }
        Ok(AgentRunResult::new(state, run_id, usage))
    }

    /// Compute `(run_id, ctx_with_run_id_when_minted)` for an entry
    /// Mint a fresh run id for this `Agent::execute` and rebase the
    /// context so the inner runnable sees `(run_id = fresh,
    /// parent_run_id = Some(caller's run_id))`. Top-level runs land
    /// with `parent_run_id = None`; sub-agent dispatches preserve
    /// the parent's id under `parent_run_id` so `(run_id,
    /// parent_run_id)` edges reconstruct the trace tree.
    ///
    /// Always mints — never reuses the caller's id. Recipes that
    /// pre-allocated a `run_id` for an external reservation see
    /// their id flow through as `parent_run_id` of the agent's run,
    /// keeping a deterministic correlation without flattening the
    /// hierarchy.
    fn scoped_run_context(ctx: &ExecutionContext) -> (String, ExecutionContext) {
        let fresh = uuid::Uuid::now_v7().to_string();
        let mut scoped = ctx.clone().with_run_id(fresh.clone());
        if let Some(parent) = ctx.run_id() {
            scoped = scoped.with_parent_run_id(parent.to_owned());
        }
        (fresh, scoped)
    }

    /// Borrow the configured execution mode.
    #[must_use]
    pub const fn execution_mode(&self) -> ExecutionMode {
        self.execution_mode
    }

    /// Borrow the configured approver (`None` in `Auto` mode).
    #[must_use]
    pub fn approver(&self) -> Option<&Arc<dyn Approver>> {
        self.approver.as_ref()
    }

    /// Number of registered lifecycle observers.
    #[must_use]
    pub fn observer_count(&self) -> usize {
        self.observers.len()
    }

    /// Run with `AgentEvent` book-ends as a stream. Sinks
    /// attached at construction time receive the same events for
    /// fan-out telemetry.
    ///
    /// The returned stream is the caller-facing view; the sink
    /// is the observability-facing view. **Both observe the same
    /// `Started` → `Complete(state)` sequence** that
    /// [`Self::execute`] produces — the only difference is the
    /// return shape (a stream of events vs the awaited terminal
    /// state).
    ///
    /// Drives the inner runnable via [`Runnable::invoke`] rather
    /// than [`Runnable::stream`] so the lifecycle is identical to
    /// `execute` and the `Complete` event always fires on
    /// successful runs (a previous design routed through
    /// `Runnable::stream`'s `Updates` mode and could silently skip
    /// `Complete` for runnables that emit no per-node updates).
    ///
    /// Construction is synchronous and infallible — every event
    /// (including the initial `Started`) yields lazily as the
    /// stream is polled. Callers consume with `.next().await` like
    /// any `Stream`; no extra `.await` on the constructor itself.
    #[must_use]
    pub fn execute_stream<'a>(
        &'a self,
        input: S,
        ctx: &'a ExecutionContext,
    ) -> BoxStream<'a, Result<AgentEvent<S>>> {
        Box::pin(self.book_end_stream(input, ctx))
    }

    /// Async-stream body for `execute_stream`. Mirrors
    /// [`Self::execute`] exactly so observers / sinks see one
    /// canonical lifecycle regardless of which surface the caller
    /// picked.
    #[allow(clippy::redundant_async_block)]
    fn book_end_stream<'a>(
        &'a self,
        input: S,
        ctx: &'a ExecutionContext,
    ) -> impl futures::Stream<Item = Result<AgentEvent<S>>> + Send + 'a {
        async_stream::stream! {
            let (run_id, scoped) = Self::scoped_run_context(ctx);
            let parent_run_id = scoped.parent_run_id().map(str::to_owned);
            // Keep `scoped` alive across the `await` boundaries below
            // so the run-id-stamped child context lives for the whole call.
            let inner_ctx: &ExecutionContext = &scoped;

            // Started book-end (sink + caller stream).
            let started = AgentEvent::Started {
                run_id: run_id.clone(),
                parent_run_id,
                agent: self.name.clone(),
            };
            self.sink.send(started.clone()).await?;
            yield Ok(started);

            // Same instrument pattern as `execute` — model +
            // tool layer spans inside `run_inner` nest under the
            // agent-run span. Sink emissions (book-end events)
            // stay outside the span; they're sink-only, not
            // tracing events.
            let outcome = self
                .run_inner(input, run_id.clone(), inner_ctx)
                .instrument(self.run_span(&run_id, inner_ctx))
                .await;
            // `scoped` lives at least until here — the borrow above is
            // valid for the whole stream body.
            drop(scoped);
            match outcome {
                Ok(result) => {
                    let complete = AgentEvent::Complete {
                        run_id: result.run_id,
                        state: result.state,
                        usage: result.usage,
                    };
                    self.sink.send(complete.clone()).await?;
                    yield Ok(complete);
                }
                Err(err) => {
                    let failed = AgentEvent::Failed {
                        run_id,
                        error: err.to_string(),
                        wire_code: err.wire_code(),
                        wire_class: err.wire_class(),
                    };
                    let _ = self.sink.send(failed.clone()).await;
                    yield Ok(failed);
                    yield Err(err);
                }
            }
        }
    }
}

impl<S> std::fmt::Debug for Agent<S>
where
    S: Clone + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

/// `Agent<S>` is itself a [`Runnable<S, S>`] so it composes inside
/// larger graphs (recursive sub-agent dispatch).
///
/// The composition contract is `S → S`, so the [`AgentRunResult`]
/// envelope is unwrapped here — composing graphs see only the
/// terminal state. Callers that need the per-run `UsageSnapshot`
/// or `run_id` go through [`Agent::execute`] directly.
#[async_trait]
impl<S> Runnable<S, S> for Agent<S>
where
    S: Clone + Send + Sync + 'static,
{
    async fn invoke(&self, input: S, ctx: &ExecutionContext) -> Result<S> {
        self.execute(input, ctx)
            .await
            .map(AgentRunResult::into_state)
    }
}

/// Fluent builder for [`Agent<S>`].
///
/// Required fields (build fails otherwise):
/// - `name` — non-empty, surfaces in `AgentEvent::Started { agent }`
///   and `OTel` spans for trace correlation
/// - `runnable` — the inner state machine the agent drives
///
/// Optional fields with sensible defaults:
/// - `sink`: [`DroppingSink`] (telemetry-free)
/// - `observers`: empty (no lifecycle hooks fire)
/// - `execution_mode`: [`ExecutionMode::Auto`]
/// - `approver`: `None` (only meaningful in `Supervised` mode)
pub struct AgentBuilder<S>
where
    S: Clone + Send + Sync + 'static,
{
    name: Option<String>,
    runnable: Option<Arc<dyn Runnable<S, S>>>,
    sinks: Vec<Arc<dyn AgentEventSink<S>>>,
    observers: Vec<DynObserver<S>>,
    execution_mode: ExecutionMode,
    approver: Option<Arc<dyn Approver>>,
}

impl<S> Default for AgentBuilder<S>
where
    S: Clone + Send + Sync + 'static,
{
    fn default() -> Self {
        Self {
            name: None,
            runnable: None,
            sinks: Vec::new(),
            observers: Vec::new(),
            execution_mode: ExecutionMode::default(),
            approver: None,
        }
    }
}

impl<S> AgentBuilder<S>
where
    S: Clone + Send + Sync + 'static,
{
    /// Set the agent's stable identifier — required, surfaces in
    /// `AgentEvent::Started { agent }` and `OTel` spans for trace
    /// correlation. Must be non-empty after `Into::into`; the
    /// build call returns `Error::Config` otherwise.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Required — the agent's underlying [`Runnable<S, S>`]. The
    /// build call returns `Error::Config` otherwise.
    #[must_use]
    pub fn with_runnable<R>(mut self, runnable: R) -> Self
    where
        R: Runnable<S, S> + 'static,
    {
        self.runnable = Some(Arc::new(runnable));
        self
    }

    /// Reuse an `Arc<dyn Runnable<S, S>>` directly — useful when
    /// the inner has already been boxed elsewhere (recipes do this
    /// to share a `CompiledGraph<S>` between the agent and other
    /// composition sites).
    #[must_use]
    pub fn with_runnable_arc(mut self, runnable: Arc<dyn Runnable<S, S>>) -> Self {
        self.runnable = Some(runnable);
        self
    }

    /// Append an event sink. Multiple calls accumulate — the agent
    /// dispatches each emitted event to every registered sink in
    /// registration order via an internal [`FanOutSink`]. Sinks
    /// added earlier run first; a sink that returns `Err` halts the
    /// run before later sinks see the event, so operators add
    /// must-succeed sinks (audit, compliance) first and wrap
    /// best-effort sinks (telemetry, embedding indexer) with
    /// [`FailOpenSink`]. Empty registrations resolve to
    /// [`DroppingSink`] at build time.
    #[must_use]
    pub fn add_sink<K>(mut self, sink: K) -> Self
    where
        K: AgentEventSink<S> + 'static,
    {
        self.sinks.push(Arc::new(sink));
        self
    }

    /// Append a pre-erased `Arc<dyn AgentEventSink<S>>` — useful
    /// when the sink has already been boxed elsewhere.
    #[must_use]
    pub fn add_sink_arc(mut self, sink: Arc<dyn AgentEventSink<S>>) -> Self {
        self.sinks.push(sink);
        self
    }

    /// Register a lifecycle observer. Observers are appended in
    /// registration order; the agent fires them in that order at
    /// each lifecycle event. Multiple observers are supported via
    /// repeated calls.
    #[must_use]
    pub fn with_observer<O>(mut self, observer: O) -> Self
    where
        O: AgentObserver<S> + 'static,
    {
        self.observers.push(Arc::new(observer));
        self
    }

    /// Register an `Arc<dyn AgentObserver<S>>` directly — useful
    /// when the observer is also held by another consumer (e.g.
    /// the same observer drives both an agent and an HTTP route
    /// for direct inspection).
    #[must_use]
    pub fn with_observer_arc(mut self, observer: DynObserver<S>) -> Self {
        self.observers.push(observer);
        self
    }

    /// Defaults to [`ExecutionMode::Auto`]. `Supervised` requires an
    /// [`Approver`] (set via [`Self::with_approver`]); the call to
    /// [`Self::build`] returns `Error::Config` if mode is
    /// `Supervised` and no approver was registered.
    #[must_use]
    pub const fn with_execution_mode(mut self, mode: ExecutionMode) -> Self {
        self.execution_mode = mode;
        self
    }

    /// Attach the approver used in `Supervised` mode. Has no
    /// effect in `Auto` mode but is preserved across builder
    /// calls so a single fluent chain can configure both modes
    /// before deciding.
    #[must_use]
    pub fn with_approver<A>(mut self, approver: A) -> Self
    where
        A: Approver + 'static,
    {
        self.approver = Some(Arc::new(approver));
        self
    }

    /// Reuse an `Arc<dyn Approver>` directly.
    #[must_use]
    pub fn with_approver_arc(mut self, approver: Arc<dyn Approver>) -> Self {
        self.approver = Some(approver);
        self
    }

    /// Finalize. Returns [`entelix_core::Error::Config`] when:
    /// - `name` was not set or is empty (every agent must be
    ///   identifiable in traces — empty-string defaults silently
    ///   destroy correlation), or
    /// - `runnable` was not set, or
    /// - `execution_mode` is `Supervised` but no `approver` was
    ///   registered (supervised mode without an approver is a
    ///   programming error — there is no decision-maker).
    pub fn build(self) -> Result<Agent<S>> {
        let name = self.name.filter(|n| !n.is_empty()).ok_or_else(|| {
            entelix_core::Error::config(
                "AgentBuilder::build: name is required and must be non-empty \
                     (call .with_name(...) — surfaces in AgentEvent::Started and OTel spans)",
            )
        })?;
        let runnable = self.runnable.ok_or_else(|| {
            entelix_core::Error::config(
                "AgentBuilder::build: runnable is required (call .with_runnable(...) or .with_runnable_arc(...))",
            )
        })?;
        if self.execution_mode.requires_approval() && self.approver.is_none() {
            return Err(entelix_core::Error::config(
                "AgentBuilder::build: ExecutionMode::Supervised requires an Approver \
                 (call .with_approver(...) or .with_approver_arc(...))",
            ));
        }
        let sink: Arc<dyn AgentEventSink<S>> = match self.sinks.len() {
            0 => Arc::new(DroppingSink),
            1 => self
                .sinks
                .into_iter()
                .next()
                .unwrap_or_else(|| unreachable!("len()==1 guarantees a value")),
            _ => {
                let mut fan = FanOutSink::<S>::new();
                for sink in self.sinks {
                    fan = fan.push(sink);
                }
                Arc::new(fan)
            }
        };
        Ok(Agent {
            name,
            runnable,
            sink,
            observers: self.observers,
            execution_mode: self.execution_mode,
            approver: self.approver,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
mod tests {
    use entelix_runnable::RunnableLambda;
    use futures::StreamExt;

    use super::*;

    fn echo_runnable() -> impl Runnable<i32, i32> {
        RunnableLambda::new(|n: i32, _ctx| async move { Ok::<_, _>(n + 1) })
    }

    #[tokio::test]
    async fn build_requires_name() {
        let err = Agent::<i32>::builder()
            .with_runnable(echo_runnable())
            .build()
            .unwrap_err();
        assert!(format!("{err}").contains("name is required"));
    }

    #[tokio::test]
    async fn build_rejects_empty_name() {
        // Empty string defaults destroy trace correlation — guard
        // against the silent-failure mode at build time.
        let err = Agent::<i32>::builder()
            .with_name("")
            .with_runnable(echo_runnable())
            .build()
            .unwrap_err();
        assert!(format!("{err}").contains("name is required"));
    }

    #[tokio::test]
    async fn build_requires_runnable() {
        let err = Agent::<i32>::builder()
            .with_name("needs-runnable")
            .build()
            .unwrap_err();
        assert!(format!("{err}").contains("runnable is required"));
    }

    #[tokio::test]
    async fn execute_drives_inner_and_emits_book_ends() {
        let sink = CaptureSink::<i32>::new();
        let agent = Agent::<i32>::builder()
            .with_name("test-agent")
            .with_runnable(echo_runnable())
            .add_sink(sink.clone())
            .build()
            .unwrap();

        let result = agent.execute(41, &ExecutionContext::new()).await.unwrap();
        assert_eq!(result.state, 42);
        assert!(!result.run_id.is_empty(), "run_id must be minted");
        assert!(
            result.usage.is_none(),
            "no RunBudget on ctx → envelope.usage is None"
        );
        let events = sink.events();
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], AgentEvent::Started { agent, .. } if agent == "test-agent"));
        assert!(matches!(events[1], AgentEvent::Complete { state: 42, .. }));
    }

    #[tokio::test]
    async fn execute_envelope_carries_frozen_usage_snapshot_when_budget_is_attached() {
        // With a RunBudget on the context, the envelope's `usage`
        // is `Some(snapshot)` — frozen at the moment the inner
        // runnable returned. Subsequent budget mutations (which
        // would happen in a real downstream layer) MUST NOT be
        // reflected in the snapshot we already returned.
        use entelix_core::RunBudget;

        let sink = CaptureSink::<i32>::new();
        let agent = Agent::<i32>::builder()
            .with_name("budgeted-agent")
            .with_runnable(echo_runnable())
            .add_sink(sink.clone())
            .build()
            .unwrap();

        // `unlimited()` plus a non-`None` `request_limit` so the
        // pre-call CAS actually increments — `check_pre_request`
        // early-returns when no cap is attached and never touches
        // the counter (the runtime path that consumes a budget
        // always sets at least one cap, so this is the realistic
        // shape).
        let budget = RunBudget::unlimited().with_request_limit(100);
        budget.check_pre_request().unwrap();
        let ctx = ExecutionContext::new().with_run_budget(budget.clone());

        let result = agent.execute(0, &ctx).await.unwrap();
        let snapshot = result.usage.expect("budget attached → usage Some");
        assert_eq!(snapshot.requests, 1, "snapshot reflects pre-stamped count");

        // Mutate the budget after the run — the snapshot must NOT
        // change. This is the frozen-at-terminal contract.
        budget.check_pre_request().unwrap();
        assert_eq!(
            snapshot.requests, 1,
            "snapshot is frozen — not Arc-shared with live counter",
        );

        // The matching `Complete` sink event carries the same
        // snapshot — operators wiring telemetry through the sink
        // see the same artifact as direct callers (one-shot vs
        // streaming surfaces match).
        let events = sink.events();
        let complete = events
            .iter()
            .find_map(|event| match event {
                AgentEvent::Complete { usage, .. } => Some(*usage),
                _ => None,
            })
            .expect("Complete event must be emitted");
        assert_eq!(complete, Some(snapshot));
    }

    #[tokio::test]
    async fn agent_is_runnable_so_it_composes() {
        // Demonstrate that Agent<S>: Runnable<S, S> — the agent is
        // itself usable as a node in a larger composition.
        let inner = Agent::<i32>::builder()
            .with_name("composed-inner")
            .with_runnable(echo_runnable())
            .build()
            .unwrap();
        let composed: Arc<dyn Runnable<i32, i32>> = Arc::new(inner);
        let result = composed.invoke(10, &ExecutionContext::new()).await.unwrap();
        assert_eq!(result, 11);
    }

    #[tokio::test]
    async fn execute_stream_emits_started_and_complete() {
        let sink = CaptureSink::<i32>::new();
        let agent = Agent::<i32>::builder()
            .with_name("streamer")
            .with_runnable(echo_runnable())
            .add_sink(sink.clone())
            .build()
            .unwrap();

        let ctx = ExecutionContext::new();
        let mut stream = agent.execute_stream(7, &ctx);
        let mut received = Vec::new();
        while let Some(event) = stream.next().await {
            received.push(event.unwrap());
        }

        // Events should be: Started → Complete(8). Both surfaces
        // (caller stream + sink) see the same sequence.
        assert!(matches!(received[0], AgentEvent::Started { .. }));
        assert!(matches!(
            received.last(),
            Some(AgentEvent::Complete {
                state: 8,
                usage: None,
                ..
            })
        ));
        assert_eq!(received.len(), sink.len());
    }

    #[tokio::test]
    async fn execute_stream_with_dropping_sink_does_not_block() {
        // Caller-facing stream still works when sink is a no-op.
        let agent = Agent::<i32>::builder()
            .with_name("dropping-sink")
            .with_runnable(echo_runnable())
            .build()
            .unwrap();
        let ctx = ExecutionContext::new();
        let mut stream = agent.execute_stream(0, &ctx);
        let mut count = 0;
        while stream.next().await.is_some() {
            count += 1;
        }
        assert!(count >= 2, "expected at least Started + Complete");
    }
}
