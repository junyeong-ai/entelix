//! `Command<S>` — caller's instruction for `CompiledGraph::resume_with`.
//!
//! After an interrupt the caller has four reasonable next moves:
//! - `Resume` as-is from the saved next-node and saved state.
//! - `Update(s)` the state (e.g. inject the human's reply) and continue
//!   from the saved next-node.
//! - `GoTo(node)` skip ahead to a different node, keeping the saved state.
//! - `ApproveTool { tool_use_id, decision }` thread an out-of-band
//!   approval decision back into the graph for an `AwaitExternal`
//!   pause raised by an approval layer (`entelix-agents::ApprovalLayer`).
//!   The resume path attaches the decision to `ExecutionContext` via
//!   the typed `entelix_core::PendingApprovalDecisions` extension so
//!   the layer's override-lookup short-circuits the approver on
//!   re-entry.

use entelix_core::ApprovalDecision;

/// Resume directive supplied to `CompiledGraph::resume_with`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Command<S>
where
    S: Clone + Send + Sync + 'static,
{
    /// Continue from the saved checkpoint exactly as it stands.
    Resume,
    /// Replace the saved state, then continue from the saved next-node.
    Update(S),
    /// Continue with the saved state but jump to `node` next.
    GoTo(String),
    /// Resume an `AwaitExternal` pause with the operator's eventual
    /// decision for the named `tool_use_id`. The resume path
    /// attaches the decision to `ExecutionContext` via
    /// `PendingApprovalDecisions` so the agent's approval layer
    /// short-circuits the approver on re-entry; the pending tool
    /// dispatches with `decision` applied. Saved state is kept
    /// intact (combine with `Update(s)` outside `resume_with` if
    /// state mutation is also needed).
    ///
    /// This variant does not depend on the enum's `S` parameter —
    /// the tool-use id and decision are state-agnostic. The `S`
    /// generic is part of the enum's shape (other variants need
    /// it); operators reach this variant via the same
    /// `Command::<S>::ApproveTool { ... }` path as any other.
    ApproveTool {
        /// Tool-use id matching the originating `ContentPart::ToolUse`
        /// — the same id carried in the `Error::Interrupted::payload`
        /// emitted by the layer.
        tool_use_id: String,
        /// Operator's decision (`Approve` to fire the dispatch,
        /// `Reject { reason }` to short-circuit). `AwaitExternal`
        /// is rejected at runtime — pausing again on resume is
        /// almost certainly an operator bug.
        decision: ApprovalDecision,
    },
}
