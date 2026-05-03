//! `AgentEventSink<S>` — consumer trait for [`AgentEvent<S>`] emissions.
//!
//! Three production patterns + one test capture, all behind a single
//! `async fn send(&self, event)` signature so the agent runtime can
//! drive any sink uniformly:
//!
//! - [`DroppingSink`] — silently discards. Default for tests and
//!   CLI flows that only consume the awaited result of
//!   `Agent::execute`.
//! - [`ChannelSink`] — `tokio::sync::mpsc` backed; the consumer
//!   owns the receiver. Bounded capacity gives backpressure; if
//!   the consumer falls behind, `send` returns an error.
//! - [`BroadcastSink`] — `tokio::sync::broadcast` backed for
//!   multi-consumer fan-out (e.g. `SSE` clients + `OTel` exporter).
//!   Slow consumers receive `Lagged` errors but the agent never
//!   blocks.
//! - [`CaptureSink`] — captures every event into an
//!   `Arc<Mutex<Vec<_>>>` for assertions in integration tests.
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
/// Operators wiring OpenTelemetry / SSE / log forwarding without
/// also calling `Agent::builder().with_sink(...)` end up here by
/// accident: the agent runs, telemetry is silently discarded,
/// alerts never fire because no data arrives.
///
/// To surface the misconfiguration in real deployments without
/// adding overhead to test-only paths, the first event a process
/// drops emits a single `tracing::debug!` naming the alternative
/// sinks. One line is enough for an operator grepping the output
/// to discover that telemetry isn't wired.
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
            target: "entelix.agents",
            "DroppingSink dropped first agent event — telemetry is not wired. \
             Pass an explicit sink via Agent::builder().with_sink(...) — see \
             ChannelSink, BroadcastSink, CaptureSink, or wire OtelLayer."
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
            target: "entelix.agents::sink",
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    type TestEvent = AgentEvent<i32>;

    fn started(agent: impl Into<String>) -> TestEvent {
        TestEvent::Started {
            run_id: "test-run".into(),
            agent: agent.into(),
        }
    }

    fn complete(state: i32) -> TestEvent {
        TestEvent::Complete {
            run_id: "test-run".into(),
            state,
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
}
