//! Tool-dispatch approval primitives shared between the agent
//! runtime (`entelix-agents::ApprovalLayer`) and the graph runtime
//! (`entelix-graph::Command::ApproveTool` resume path).
//!
//! These types live in `entelix-core` so both downstream crates can
//! reference them without violating the workspace DAG (`entelix-graph`
//! depends on `entelix-core`; `entelix-agents` depends on both).
//!
//! Operators implement the `Approver` trait in `entelix-agents` and
//! return [`ApprovalDecision`] from `decide`. The HITL pause-and-
//! resume flow uses [`Command::ApproveTool`](entelix-graph) carrying
//! an `ApprovalDecision` directly — the SDK threads the decision to
//! the resumed dispatch through an internal context extension, so
//! operators never construct the carrier manually.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Outcome of a single tool-dispatch approval decision.
///
/// Returned by `entelix_agents::Approver::decide`. The agent
/// runtime maps each variant onto the dispatch outcome:
///
/// - `Approve` → the inner tool service runs.
/// - `Reject { reason }` → the dispatch short-circuits with
///   `Error::InvalidRequest` carrying `reason` (the model receives
///   the lean text).
/// - `AwaitExternal` → the dispatch raises `Error::Interrupted`
///   (kind = [`INTERRUPT_KIND_APPROVAL_PENDING`]). The graph
///   dispatch loop persists a checkpoint and bubbles the typed
///   error to the caller. Resume via
///   `entelix_graph::Command::ApproveTool { tool_use_id, decision }`
///   threads the operator's eventual decision back through; the
///   approver is not re-asked for that `tool_use_id`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ApprovalDecision {
    /// Proceed with the pending tool call.
    Approve,
    /// Refuse; the lean `reason` is fed back to the model.
    Reject {
        /// Human-readable reason. Forwarded to the LLM verbatim,
        /// so keep it short and prose-shaped (no Rust types, no IDs).
        reason: String,
    },
    /// The agent should pause (graph `interrupt`) and wait for an
    /// external decision delivered via
    /// `entelix_graph::Command::ApproveTool` on the resume call.
    /// Supports out-of-band human review (web UI, Slack, e-mail).
    AwaitExternal,
}

/// Discriminator string carried inside `Error::Interrupted::payload`
/// when an approval layer pauses on an `AwaitExternal` decision.
///
/// Operators inspecting the payload can match on this constant to
/// recognise approval pauses without parsing the full payload
/// shape. The typed resume primitive is
/// `entelix_graph::Command::ApproveTool` — operators that thread
/// resumes through `Command` rarely need the discriminator string
/// directly.
pub const INTERRUPT_KIND_APPROVAL_PENDING: &str = "approval_pending";

/// Resume-side mapping from `tool_use_id` to the operator's
/// decision. Attached to `ExecutionContext::extension` so the
/// agent's approval layer reads the decision during a re-fired
/// dispatch and short-circuits the approver for any `tool_use_id`
/// whose decision is present.
///
/// Two attachment paths:
///
/// - **Typed (recommended)**: `Command::ApproveTool { tool_use_id,
///   decision }` on `CompiledGraph::resume_with` constructs the
///   carrier internally. Operators using the graph-resume path
///   never touch this type directly.
/// - **Direct (advanced)**: operators dispatching through the
///   raw `ToolRegistry` (no graph, no checkpointer) attach the
///   carrier on the request `ExecutionContext`:
///
///   ```ignore
///   let mut pending = PendingApprovalDecisions::new();
///   pending.insert("tu-1", ApprovalDecision::Approve);
///   let ctx = ExecutionContext::new().add_extension(pending);
///   registry.dispatch("tu-1", "echo", input, &ctx).await?;
///   ```
///
///   The direct path is the canonical mechanism the SDK exposes
///   for non-graph dispatch (e.g. tests, custom embedded loops).
///   `Command::ApproveTool` is the higher-level convenience for
///   graph-driven agents.
#[derive(Clone, Debug, Default)]
pub struct PendingApprovalDecisions {
    by_tool_use_id: HashMap<String, ApprovalDecision>,
}

impl PendingApprovalDecisions {
    /// Empty mapping.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one decision for the given `tool_use_id`.
    pub fn insert(&mut self, tool_use_id: impl Into<String>, decision: ApprovalDecision) {
        self.by_tool_use_id.insert(tool_use_id.into(), decision);
    }

    /// Look up the decision for `tool_use_id`. `None` falls through
    /// to the configured `Approver`.
    #[must_use]
    pub fn get(&self, tool_use_id: &str) -> Option<&ApprovalDecision> {
        self.by_tool_use_id.get(tool_use_id)
    }
}
