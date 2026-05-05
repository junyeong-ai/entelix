//! `AgentRunResult<S>` — terminal envelope returned by
//! [`crate::Agent::execute`] / [`crate::Agent::execute_with`].
//!
//! Pairs the agent's final state with the run's stable correlation
//! id and a frozen [`UsageSnapshot`] of the
//! [`entelix_core::RunBudget`] counters at terminal — operators see
//! how much of the cap each axis consumed without holding the live
//! `Arc` (and without observing further mutations from sibling
//! sub-agent dispatches that share the same budget through the
//! `ExecutionContext`).
//!
//! ## Why an envelope, not just an event
//!
//! [`crate::AgentEvent::Complete`] also carries the snapshot, but
//! caller-facing `execute` returns one value rather than a stream.
//! Forcing the caller to thread their own `CaptureSink` to read
//! per-run usage would re-introduce stateful telemetry coupling
//! that the agent runtime exists to avoid (and does not work for
//! the `Auto`-mode `execute` path that does not own a sink loop).
//!
//! ## Frozen at terminal — invariant
//!
//! `usage` is captured *after* the inner runnable returns
//! successfully and *before* observers fire — observers can mutate
//! side-channel state (vector store writes, summary persistence)
//! whose dispatches may themselves consume budget through layered
//! `ChatModel` calls; freezing the snapshot ahead of that point
//! reflects exactly the agent run's own cost. ADR-0080 documents
//! the budget; ADR-0081 documents this envelope.

use entelix_core::UsageSnapshot;

/// Terminal artifact of one agent run.
///
/// Returned by [`crate::Agent::execute`] and
/// [`crate::Agent::execute_with`]. Constructed only by the agent
/// runtime — public field access lets callers destructure
/// ergonomically (`let AgentRunResult { state, usage, .. } = ...`)
/// without a fresh constructor surface.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct AgentRunResult<S> {
    /// Final state produced by the inner runnable.
    pub state: S,
    /// Per-run correlation id (UUID v7 by default, or the value
    /// pre-stamped by the caller via
    /// [`entelix_core::ExecutionContext::with_run_id`]).
    pub run_id: String,
    /// Frozen [`UsageSnapshot`] of the [`entelix_core::RunBudget`]
    /// counters at the moment the inner runnable returned.
    /// `None` when no budget was attached to the
    /// [`entelix_core::ExecutionContext`] (the common opt-out
    /// shape — operators that do not cap usage pay zero overhead).
    pub usage: Option<UsageSnapshot>,
}

impl<S> AgentRunResult<S> {
    /// Constructor used by [`crate::Agent`]. Crate-private so the
    /// envelope's provenance stays inside the runtime — third
    /// parties cannot synthesize a `RunResult` that did not come
    /// from a real run.
    pub(crate) const fn new(state: S, run_id: String, usage: Option<UsageSnapshot>) -> Self {
        Self {
            state,
            run_id,
            usage,
        }
    }

    /// Consume the envelope, returning the inner state. Convenience
    /// for callers wiring an agent into a parent graph that only
    /// needs the typed state — the same pattern the
    /// `Runnable<S, S>` impl on [`crate::Agent`] uses internally.
    #[must_use]
    pub fn into_state(self) -> S {
        self.state
    }
}
