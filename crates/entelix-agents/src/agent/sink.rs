//! `AgentEventSink<S>` — consumer trait for [`AgentEvent<S>`] emissions.
//!
//! All sinks share one `async fn send(&self, event)` signature so the
//! agent runtime can drive any of them uniformly:
//!
//! - [`DroppingSink`] — silently discards. Default for tests and
//!   CLI flows that only consume the awaited result of
//!   `Agent::execute`.
//! - [`ChannelSink`] — `tokio::sync::mpsc` backed; the consumer
//!   owns the receiver. Bounded capacity gives backpressure; if
//!   the consumer falls behind, `send` returns an error.
//! - [`BroadcastSink`] — `tokio::sync::broadcast` backed for
//!   multi-consumer subscribe-shaped fan-out (`SSE` clients +
//!   `OTel` exporter via `subscribe()`). Slow consumers receive
//!   `Lagged` errors but the agent never blocks.
//! - [`CaptureSink`] — captures every event into an
//!   `Arc<Mutex<Vec<_>>>` for assertions in integration tests.
//! - [`FanOutSink`] — callback-shaped composition primitive that
//!   dispatches every event to a fixed set of inner sinks
//!   sequentially. Stops at the first failing sink — wrap
//!   best-effort sinks with [`FailOpenSink`] to keep
//!   higher-priority sinks downstream.
//! - [`FailOpenSink`] — composition adapter that swallows the
//!   inner sink's `Err` (logs once via `tracing::warn!`) and
//!   always returns `Ok(())`. Lifts an observe-only sink
//!   (embedding indexer, billing, anomaly detector) into the
//!   `must-succeed` semantics that bare [`AgentEventSink::send`]
//!   contract demands.
//!
//! Composition pattern for production deployments — audit must
//! succeed, telemetry is best-effort:
//!
//! ```ignore
//! use std::sync::Arc;
//! use entelix_agents::{FanOutSink, FailOpenSink};
//!
//! let sink = FanOutSink::<MyState>::new()
//!     .push(audit_sink)                          // must succeed → propagates Err
//!     .push(Arc::new(FailOpenSink::new(otel)))   // observe-only → swallowed
//!     .push(Arc::new(FailOpenSink::new(billing))); // observe-only → swallowed
//! ```
//!
//! [`AgentEvent<S>`]: crate::agent::event::AgentEvent

use std::sync::Arc;

use async_trait::async_trait;
use entelix_core::error::{Error, Result};
use parking_lot::Mutex;
use tokio::sync::{broadcast, mpsc};

use crate::agent::event::AgentEvent;

/// Consumer trait the agent calls for every emitted event.
///
/// Implementations should be cheap to clone — the agent runtime
/// holds the sink as `Arc<dyn AgentEventSink<S>>` and may share it
/// across nested sub-agents and observers.
#[async_trait]
pub trait AgentEventSink<S>: Send + Sync
where
    S: Clone + Send + Sync + 'static,
{
    /// Consume a single event. Returning `Err` halts the agent —
    /// sinks that want best-effort semantics should swallow their
    /// own errors and return `Ok(())`.
    async fn send(&self, event: AgentEvent<S>) -> Result<()>;
}

/// No-op sink — the agent runs to completion without surfacing
/// per-event telemetry. Default when no sink is configured.
///
/// To surface the misconfiguration in real deployments without
/// adding overhead to test-only paths, the first event a process
/// drops emits a single `tracing::debug!` naming the alternative
/// sinks. One line is enough for an operator grepping the output
/// to discover that telemetry isn't wired.
///
/// Operators wiring OpenTelemetry / SSE / log forwarding without
/// also calling `Agent::builder().add_sink(...)` end up here by
/// accident: the agent runs, telemetry is silently discarded,
/// alerts never fire because no data arrives.
#[derive(Clone, Copy, Debug, Default)]
pub struct DroppingSink;

#[async_trait]
impl<S> AgentEventSink<S> for DroppingSink
where
    S: Clone + Send + Sync + 'static,
{
    async fn send(&self, _event: AgentEvent<S>) -> Result<()> {
        warn_dropped_first_event();
        Ok(())
    }
}

/// Process-wide one-shot guard for the dropped-event diagnostic.
/// Using a `static AtomicBool` (rather than a field on
/// [`DroppingSink`]) keeps the type `Copy` so existing call sites
/// that pass it by value continue to compile, and means a process
/// running 50 agents with `DroppingSink` defaults sees one log
/// line, not fifty.
fn warn_dropped_first_event() {
    use std::sync::atomic::{AtomicBool, Ordering};
    static WARNED: AtomicBool = AtomicBool::new(false);
    if WARNED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
        .is_ok()
    {
        tracing::debug!(
            target: "entelix_agents",
            "DroppingSink dropped first agent event — telemetry is not wired. \
             Pass an explicit sink via Agent::builder().add_sink(...) — see \
             ChannelSink, BroadcastSink, CaptureSink, FanOutSink, or wire OtelLayer."
        );
    }
}

/// Single-consumer mpsc sink. Construct via [`Self::new`] — the
/// caller keeps the [`tokio::sync::mpsc::Receiver`] for downstream
/// consumption (HTTP SSE driver, file logger, etc.).
///
/// Bounded capacity provides backpressure: when the buffer fills,
/// `send` waits for the consumer to drain. If the consumer is
/// dropped, `send` returns [`Error::Cancelled`] so the agent can
/// shut down cleanly.
pub struct ChannelSink<S>
where
    S: Clone + Send + Sync + 'static,
{
    tx: mpsc::Sender<AgentEvent<S>>,
}

impl<S> ChannelSink<S>
where
    S: Clone + Send + Sync + 'static,
{
    /// Build a sink with an mpsc channel of the given capacity.
    /// Returns the sink and the matching receiver.
    #[must_use]
    pub fn new(capacity: usize) -> (Self, mpsc::Receiver<AgentEvent<S>>) {
        let (tx, rx) = mpsc::channel(capacity);
        (Self { tx }, rx)
    }
}

impl<S> Clone for ChannelSink<S>
where
    S: Clone + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
        }
    }
}

impl<S> std::fmt::Debug for ChannelSink<S>
where
    S: Clone + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChannelSink")
            .field("capacity", &self.tx.max_capacity())
            .field("closed", &self.tx.is_closed())
            .finish()
    }
}

#[async_trait]
impl<S> AgentEventSink<S> for ChannelSink<S>
where
    S: Clone + Send + Sync + 'static,
{
    async fn send(&self, event: AgentEvent<S>) -> Result<()> {
        self.tx.send(event).await.map_err(|_| Error::Cancelled)
    }
}

/// Multi-consumer broadcast sink. Each subscriber gets its own
/// `Receiver`; slow consumers receive `Lagged` errors but the
/// agent never blocks. Suitable for `SSE` fan-out + `OTel` exporter
/// + recording sink simultaneously.
pub struct BroadcastSink<S>
where
    S: Clone + Send + Sync + 'static,
{
    tx: broadcast::Sender<AgentEvent<S>>,
}

impl<S> BroadcastSink<S>
where
    S: Clone + Send + Sync + 'static,
{
    /// Build a broadcast sink with the given per-subscriber buffer
    /// depth. Subscribers register via [`Self::subscribe`].
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let (tx, _rx_drop) = broadcast::channel(capacity);
        Self { tx }
    }

    /// Register a new subscriber. Each receiver sees every event
    /// emitted *after* its registration.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<AgentEvent<S>> {
        self.tx.subscribe()
    }

    /// Number of currently registered subscribers.
    #[must_use]
    pub fn receiver_count(&self) -> usize {
        self.tx.receiver_count()
    }
}

impl<S> Clone for BroadcastSink<S>
where
    S: Clone + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
        }
    }
}

impl<S> std::fmt::Debug for BroadcastSink<S>
where
    S: Clone + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BroadcastSink")
            .field("subscribers", &self.tx.receiver_count())
            .finish()
    }
}

#[async_trait]
impl<S> AgentEventSink<S> for BroadcastSink<S>
where
    S: Clone + Send + Sync + 'static,
{
    async fn send(&self, event: AgentEvent<S>) -> Result<()> {
        // `send` returns Err when *all* receivers have been dropped.
        // That is not a hard failure — the agent continues. A
        // crashed or detached consumer would otherwise spam one
        // debug line per event for the rest of the run, so a
        // process-wide one-shot guard collapses the storm to a
        // single observable signal.
        if self.tx.send(event).is_err() {
            warn_no_subscribers_once();
        }
        Ok(())
    }
}

/// Process-wide one-shot guard for the no-subscribers diagnostic.
/// A `static AtomicBool` keeps `BroadcastSink<S>` zero-cost (no
/// extra field per instance, no `S`-monomorphisation tax) and
/// means N agents observing N consumer drops produce one log
/// line, not N × N. Operators see one alert and investigate.
fn warn_no_subscribers_once() {
    use std::sync::atomic::{AtomicBool, Ordering};
    static WARNED: AtomicBool = AtomicBool::new(false);
    if WARNED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
        .is_ok()
    {
        tracing::debug!(
            target: "entelix_agents::sink",
            "BroadcastSink: no active subscribers; event dropped (further drops will be silent — \
             investigate whether the consumer crashed or was detached)"
        );
    }
}

/// In-memory capture sink for integration tests. Every emitted
/// event is appended to an `Arc<Mutex<Vec<_>>>`; tests inspect the
/// vector to assert ordering and content.
pub struct CaptureSink<S>
where
    S: Clone + Send + Sync + 'static,
{
    events: Arc<Mutex<Vec<AgentEvent<S>>>>,
}

impl<S> CaptureSink<S>
where
    S: Clone + Send + Sync + 'static,
{
    /// Empty capture sink.
    #[must_use]
    pub fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Snapshot of the events captured so far. Cheap clone of the
    /// internal vector; subsequent emissions do not affect the
    /// returned snapshot.
    #[must_use]
    pub fn events(&self) -> Vec<AgentEvent<S>> {
        self.events.lock().clone()
    }

    /// Number of events captured.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.lock().len()
    }

    /// Whether the capture is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.lock().is_empty()
    }
}

impl<S> Default for CaptureSink<S>
where
    S: Clone + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<S> Clone for CaptureSink<S>
where
    S: Clone + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        Self {
            events: Arc::clone(&self.events),
        }
    }
}

impl<S> std::fmt::Debug for CaptureSink<S>
where
    S: Clone + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CaptureSink")
            .field("captured", &self.len())
            .finish()
    }
}

#[async_trait]
impl<S> AgentEventSink<S> for CaptureSink<S>
where
    S: Clone + Send + Sync + 'static,
{
    async fn send(&self, event: AgentEvent<S>) -> Result<()> {
        self.events.lock().push(event);
        Ok(())
    }
}

/// Callback-shaped fan-out — every emitted event reaches every
/// inner sink in registration order.
///
/// Stops at the first failing sink: subsequent sinks do **not** see
/// the event. Operators add sinks in priority order — highest-priority
/// (must-succeed: audit, compliance) first, lower-priority (telemetry,
/// embedding indexer) wrapped in [`FailOpenSink`] later. A failing
/// must-succeed sink halts the run before the lower-priority work
/// runs at all.
///
/// Sequential, not parallel — events arrive at sinks in stable
/// registration order so debugging traces stay reproducible across
/// runs. Fan-out across many sinks is bounded by the slowest sink;
/// observe-only sinks that may stall (HTTP exporter, remote DB)
/// belong behind a [`ChannelSink`] in front of [`FanOutSink`] so the
/// agent loop never blocks on their I/O.
///
/// Distinct from [`BroadcastSink`]: `FanOutSink` is callback-shaped
/// (every sink runs synchronously inside `send`), `BroadcastSink` is
/// subscribe-shaped (consumers pull from their own `Receiver`).
/// Compose freely — a `FanOutSink` whose pushed sinks include a
/// `BroadcastSink` gives both shapes simultaneously.
pub struct FanOutSink<S>
where
    S: Clone + Send + Sync + 'static,
{
    sinks: Vec<Arc<dyn AgentEventSink<S>>>,
}

impl<S> FanOutSink<S>
where
    S: Clone + Send + Sync + 'static,
{
    /// Empty fan-out — `send` is a no-op until at least one inner
    /// sink is pushed.
    #[must_use]
    pub fn new() -> Self {
        Self { sinks: Vec::new() }
    }

    /// Append an inner sink. Builder-style — chains naturally with
    /// `Arc::new(MySink::new()).into()`.
    #[must_use]
    pub fn push(mut self, sink: Arc<dyn AgentEventSink<S>>) -> Self {
        self.sinks.push(sink);
        self
    }

    /// Number of registered inner sinks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sinks.len()
    }

    /// Whether no inner sinks are registered (`send` is a no-op).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sinks.is_empty()
    }
}

impl<S> Default for FanOutSink<S>
where
    S: Clone + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<S> Clone for FanOutSink<S>
where
    S: Clone + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        Self {
            sinks: self.sinks.clone(),
        }
    }
}

impl<S> std::fmt::Debug for FanOutSink<S>
where
    S: Clone + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FanOutSink")
            .field("sinks", &self.sinks.len())
            .finish()
    }
}

#[async_trait]
impl<S> AgentEventSink<S> for FanOutSink<S>
where
    S: Clone + Send + Sync + 'static,
{
    async fn send(&self, event: AgentEvent<S>) -> Result<()> {
        for sink in &self.sinks {
            sink.send(event.clone()).await?;
        }
        Ok(())
    }
}

/// Composition adapter that swallows the inner sink's errors and
/// always returns `Ok(())`.
///
/// `AgentEventSink::send` returning `Err` halts the agent (the
/// trait's contract). For sinks whose failure should NOT propagate
/// — embedding indexer, billing meter, anomaly detector,
/// best-effort telemetry — wrap in `FailOpenSink` so a transient
/// downstream failure logs once and the run continues.
///
/// The lift is one-way and explicit: there is no `is_observe_only`
/// flag on the sink itself. Operators express intent at the
/// composition site (`FailOpenSink::new(my_sink)`) so the dispatch
/// loop stays simple — every sink is treated identically.
pub struct FailOpenSink<S>
where
    S: Clone + Send + Sync + 'static,
{
    inner: Arc<dyn AgentEventSink<S>>,
}

impl<S> FailOpenSink<S>
where
    S: Clone + Send + Sync + 'static,
{
    /// Wrap an inner sink — its errors will be logged via
    /// `tracing::warn!` and never propagated to the agent runtime.
    #[must_use]
    pub fn new(inner: Arc<dyn AgentEventSink<S>>) -> Self {
        Self { inner }
    }
}

impl<S> Clone for FailOpenSink<S>
where
    S: Clone + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<S> std::fmt::Debug for FailOpenSink<S>
where
    S: Clone + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FailOpenSink").finish_non_exhaustive()
    }
}

#[async_trait]
impl<S> AgentEventSink<S> for FailOpenSink<S>
where
    S: Clone + Send + Sync + 'static,
{
    async fn send(&self, event: AgentEvent<S>) -> Result<()> {
        if let Err(err) = self.inner.send(event).await {
            tracing::warn!(
                target: "entelix_agents::sink",
                error = %err,
                "FailOpenSink: inner sink errored — discarding event and continuing"
            );
        }
        Ok(())
    }
}

/// Adapter that erases an agent's state type so a single
/// [`AgentEventSink<()>`] can fan in from heterogeneous agents
/// (`Agent<ReActState>`, `Agent<SupervisorState>`, operator-defined
/// state types) in a multi-agent system.
///
/// The wrapper takes any `Arc<dyn AgentEventSink<()>>` (the canonical
/// state-agnostic sink shape) and produces an [`AgentEventSink<S>`]
/// for any `S: Send + Sync + 'static`. The adapter calls
/// [`AgentEvent::erase_state`] on every dispatch, replacing the
/// agent's terminal state with `()` while every other field
/// (`run_id`, `tenant_id`, `parent_run_id`, tool I/O, error envelope,
/// usage snapshot, approval decisions) reaches the sink unchanged.
///
/// Wire pattern:
///
/// ```ignore
/// let audit: Arc<dyn AgentEventSink<()>> = Arc::new(MyAuditSink);
///
/// // ReAct agent — own state-typed sink built from the same audit.
/// let react = ReActAgentBuilder::new(model)
///     .add_sink(Arc::new(StateErasureSink::new(Arc::clone(&audit))))
///     .build()?;
///
/// // Supervisor agent — sharing the same audit pipeline.
/// let supervisor = create_supervisor_agent(...)
///     .add_sink(Arc::new(StateErasureSink::new(audit)))
///     .build()?;
/// ```
pub struct StateErasureSink<S> {
    inner: Arc<dyn AgentEventSink<()>>,
    _phantom: std::marker::PhantomData<fn(S)>,
}

impl<S> StateErasureSink<S> {
    /// Wrap a state-agnostic sink for attachment to an
    /// `Agent<S>`-typed event stream.
    #[must_use]
    pub fn new(inner: Arc<dyn AgentEventSink<()>>) -> Self {
        Self {
            inner,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<S> Clone for StateErasureSink<S> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<S> std::fmt::Debug for StateErasureSink<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StateErasureSink").finish_non_exhaustive()
    }
}

#[async_trait]
impl<S> AgentEventSink<S> for StateErasureSink<S>
where
    S: Clone + Send + Sync + 'static,
{
    async fn send(&self, event: AgentEvent<S>) -> Result<()> {
        self.inner.send(event.erase_state()).await
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    type TestEvent = AgentEvent<i32>;

    fn started(agent: impl Into<String>) -> TestEvent {
        TestEvent::Started {
            run_id: "test-run".into(),
            tenant_id: entelix_core::TenantId::new("t-test"),
            parent_run_id: None,
            agent: agent.into(),
        }
    }

    fn complete(state: i32) -> TestEvent {
        TestEvent::Complete {
            run_id: "test-run".into(),
            tenant_id: entelix_core::TenantId::new("t-test"),
            state,
            usage: None,
        }
    }

    #[tokio::test]
    async fn dropping_sink_silently_discards_events() {
        let sink = DroppingSink;
        for i in 0..10 {
            sink.send(started(format!("a{i}"))).await.unwrap();
        }
    }

    #[tokio::test]
    async fn channel_sink_round_trips_in_order() {
        let (sink, mut rx) = ChannelSink::<i32>::new(4);
        for i in 0..3 {
            sink.send(started(format!("a{i}"))).await.unwrap();
        }
        drop(sink);
        let mut received = Vec::new();
        while let Some(event) = rx.recv().await {
            received.push(event);
        }
        assert_eq!(received.len(), 3);
        assert!(matches!(&received[0], AgentEvent::Started { agent, .. } if agent == "a0"));
    }

    #[tokio::test]
    async fn channel_sink_returns_cancelled_when_receiver_dropped() {
        let (sink, rx) = ChannelSink::<i32>::new(1);
        sink.send(started("a")).await.unwrap();
        drop(rx);
        let err = sink.send(complete(0)).await.unwrap_err();
        assert!(matches!(err, Error::Cancelled));
    }

    #[tokio::test]
    async fn broadcast_sink_fans_out_to_multiple_subscribers() {
        let sink = BroadcastSink::<i32>::new(8);
        let mut a = sink.subscribe();
        let mut b = sink.subscribe();
        assert_eq!(sink.receiver_count(), 2);
        sink.send(complete(7)).await.unwrap();
        let ea = a.recv().await.unwrap();
        let eb = b.recv().await.unwrap();
        assert!(matches!(ea, AgentEvent::Complete { state: 7, .. }));
        assert!(matches!(eb, AgentEvent::Complete { state: 7, .. }));
    }

    #[tokio::test]
    async fn broadcast_sink_no_subscribers_does_not_error() {
        let sink = BroadcastSink::<i32>::new(8);
        sink.send(complete(0)).await.unwrap();
    }

    #[tokio::test]
    async fn capture_sink_preserves_order_and_content() {
        let sink = CaptureSink::<i32>::new();
        sink.send(started("test")).await.unwrap();
        sink.send(complete(42)).await.unwrap();

        assert_eq!(sink.len(), 2);
        let events = sink.events();
        assert!(matches!(events[0], AgentEvent::Started { .. }));
        assert!(matches!(events[1], AgentEvent::Complete { state: 42, .. }));
    }

    #[tokio::test]
    async fn capture_sink_clones_share_underlying_buffer() {
        let sink_a = CaptureSink::<i32>::new();
        let sink_b = sink_a.clone();
        sink_a.send(complete(1)).await.unwrap();
        sink_b.send(complete(2)).await.unwrap();
        assert_eq!(sink_a.len(), 2);
        assert_eq!(sink_b.len(), 2);
    }

    /// Sink that always returns `Err` for `FailOpenSink` and
    /// `FanOutSink` regression coverage.
    #[derive(Default)]
    struct FailingSink {
        calls: parking_lot::Mutex<u32>,
    }

    #[async_trait]
    impl AgentEventSink<i32> for FailingSink {
        async fn send(&self, _event: AgentEvent<i32>) -> Result<()> {
            *self.calls.lock() += 1;
            Err(Error::Cancelled)
        }
    }

    #[tokio::test]
    async fn fan_out_sink_dispatches_in_registration_order() {
        let a = CaptureSink::<i32>::new();
        let b = CaptureSink::<i32>::new();
        let fan = FanOutSink::<i32>::new()
            .push(Arc::new(a.clone()))
            .push(Arc::new(b.clone()));
        fan.send(complete(1)).await.unwrap();
        fan.send(complete(2)).await.unwrap();
        assert_eq!(a.len(), 2);
        assert_eq!(b.len(), 2);
        assert!(matches!(
            a.events()[0],
            AgentEvent::Complete { state: 1, .. }
        ));
    }

    #[tokio::test]
    async fn fan_out_sink_propagates_first_error_and_stops() {
        let recorded = CaptureSink::<i32>::new();
        let failing = Arc::new(FailingSink::default());
        let fan = FanOutSink::<i32>::new()
            .push(Arc::clone(&failing) as Arc<dyn AgentEventSink<i32>>)
            .push(Arc::new(recorded.clone()));
        let err = fan.send(complete(1)).await.unwrap_err();
        assert!(matches!(err, Error::Cancelled));
        assert_eq!(*failing.calls.lock(), 1, "failing sink saw the event");
        assert_eq!(
            recorded.len(),
            0,
            "downstream sinks must not see the event after an upstream failure"
        );
    }

    #[tokio::test]
    async fn fail_open_sink_swallows_inner_error() {
        let failing: Arc<dyn AgentEventSink<i32>> = Arc::new(FailingSink::default());
        let lifted = FailOpenSink::new(failing);
        // Three failing sends — every one returns Ok.
        for _ in 0..3 {
            lifted.send(complete(0)).await.unwrap();
        }
    }

    #[tokio::test]
    async fn fail_open_sink_lifts_into_fan_out_to_isolate_observe_only() {
        let recorded = CaptureSink::<i32>::new();
        let failing: Arc<dyn AgentEventSink<i32>> = Arc::new(FailingSink::default());
        let fan = FanOutSink::<i32>::new()
            .push(Arc::new(FailOpenSink::new(failing))) // observe-only — lifted
            .push(Arc::new(recorded.clone())); // must-succeed — sees the event
        fan.send(complete(7)).await.unwrap();
        assert_eq!(
            recorded.len(),
            1,
            "lifting the failing sink with FailOpenSink must keep downstream sinks reachable"
        );
    }

    #[tokio::test]
    async fn state_erasure_sink_fans_in_heterogeneous_agents_to_one_unit_sink() {
        // Two agents with distinct state types (i32, String) share a
        // single `AgentEventSink<()>` audit pipeline through
        // StateErasureSink wrappers. The audit sees every event from
        // both agents with `Complete::state == ()` and every header
        // field (run_id / tenant_id / parent_run_id) preserved.
        let audit: Arc<CaptureSink<()>> = Arc::new(CaptureSink::<()>::new());
        let audit_dyn: Arc<dyn AgentEventSink<()>> = audit.clone();

        // Agent A: state-typed `i32`.
        let a_adapter: StateErasureSink<i32> = StateErasureSink::new(Arc::clone(&audit_dyn));
        a_adapter
            .send(AgentEvent::Started {
                run_id: "a-run".into(),
                tenant_id: entelix_core::TenantId::new("t-shared"),
                parent_run_id: None,
                agent: "agent-a".into(),
            })
            .await
            .unwrap();
        a_adapter
            .send(AgentEvent::Complete {
                run_id: "a-run".into(),
                tenant_id: entelix_core::TenantId::new("t-shared"),
                state: 42_i32,
                usage: None,
            })
            .await
            .unwrap();

        // Agent B: state-typed `String`.
        let b_adapter: StateErasureSink<String> = StateErasureSink::new(audit_dyn);
        b_adapter
            .send(AgentEvent::Started {
                run_id: "b-run".into(),
                tenant_id: entelix_core::TenantId::new("t-shared"),
                parent_run_id: Some("a-run".into()),
                agent: "agent-b".into(),
            })
            .await
            .unwrap();
        b_adapter
            .send(AgentEvent::Complete {
                run_id: "b-run".into(),
                tenant_id: entelix_core::TenantId::new("t-shared"),
                state: "done".to_owned(),
                usage: None,
            })
            .await
            .unwrap();

        // Audit captured all four events with Complete::state == ().
        let events = audit.events();
        assert_eq!(events.len(), 4);

        // Headers survived erasure.
        match &events[0] {
            AgentEvent::Started {
                run_id,
                tenant_id,
                parent_run_id,
                agent,
            } => {
                assert_eq!(run_id, "a-run");
                assert_eq!(tenant_id.as_str(), "t-shared");
                assert_eq!(parent_run_id.as_deref(), None);
                assert_eq!(agent, "agent-a");
            }
            other => panic!("unexpected event: {other:?}"),
        }
        match &events[2] {
            AgentEvent::Started {
                parent_run_id,
                agent,
                ..
            } => {
                assert_eq!(parent_run_id.as_deref(), Some("a-run"));
                assert_eq!(agent, "agent-b");
            }
            other => panic!("unexpected event: {other:?}"),
        }

        // Complete events lost their typed state via erasure — the
        // single-unit audit sink observes `state: ()` from both.
        match &events[1] {
            AgentEvent::Complete { state, .. } => assert_eq!(*state, ()),
            other => panic!("unexpected event: {other:?}"),
        }
        match &events[3] {
            AgentEvent::Complete { state, .. } => assert_eq!(*state, ()),
            other => panic!("unexpected event: {other:?}"),
        }
    }
}
