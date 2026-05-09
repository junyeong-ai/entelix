//! Single human-in-the-loop primitive for the entelix runtime.
//!
//! Every paused dispatch flows through one shape:
//!
//! ```text
//! Error::Interrupted { kind: InterruptionKind, payload: serde_json::Value }
//! ```
//!
//! - **`kind`** — typed reason. SDK-defined variants like
//!   [`InterruptionKind::ApprovalPending`] carry the structured data
//!   the resumer needs (e.g. `tool_use_id`); operator-defined pauses
//!   use [`InterruptionKind::Custom`].
//! - **`payload`** — operator free-form data, passed straight through
//!   to the resumer. For typed kinds (`ApprovalPending`) the payload
//!   is usually `Value::Null`; for `Custom` it carries whatever the
//!   tool / node decided to surface.
//!
//! Mid-tool, mid-node, and middleware-layer pauses all use the same
//! primitive — return [`Err`] from `Tool::execute`, a graph node's
//! `Runnable<S, S>::invoke`, or any `tower::Layer`'s `call` future.
//! The dispatch loop catches the error, persists a checkpoint, and
//! returns it to the caller; the caller resumes via
//! `entelix_graph::CompiledGraph::resume_with(Command, &ctx)`.

use serde::{Deserialize, Serialize};

use crate::error::Error;

/// Phase at which a graph-scheduled pause fires — before the marked
/// node runs (`Before`) or after it returns Ok (`After`).
///
/// `non_exhaustive` matches the workspace-wide pub-enum hygiene gate
/// (`cargo xtask surface-hygiene`). The `Before` / `After` partition
/// of node-boundary timing is conceptually closed, so operator match
/// sites typically use a `_ => unreachable!("…")` fall-through arm
/// — the marker exists as future-proofing against a deliberate SDK
/// addition (e.g. wrapper phases for fan-out joins) rather than a
/// signal of an open variant set.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum InterruptionPhase {
    /// Pause before the marked node executes; resume re-runs the
    /// node from saved pre-state.
    Before,
    /// Pause after the marked node returns Ok; resume continues
    /// forward, skipping a re-run of the just-completed node.
    After,
}

/// Why a dispatch paused. SDK variants carry typed structured data
/// the resumer needs to thread the decision back; operator-defined
/// pauses surface as [`Self::Custom`] with arbitrary payload.
///
/// `non_exhaustive` so post-1.0 SDK variants ship as MINOR; operator
/// match sites should always carry a fall-through `_ =>` arm for
/// future-proofing.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum InterruptionKind {
    /// Operator-defined pause. The associated `payload` on
    /// [`Error::Interrupted`] carries whatever the tool, node, or
    /// layer surfaced. Default kind for [`crate::interrupt`].
    Custom,
    /// A tool-dispatch approval is pending. The agent runtime's
    /// `ApprovalLayer` raises this kind when an `Approver` returns
    /// `ApprovalDecision::AwaitExternal`. Resume via
    /// `entelix_graph::Command::ApproveTool { tool_use_id, decision }`
    /// — the SDK threads the decision to the resumed dispatch.
    ApprovalPending {
        /// The pending tool-use id awaiting an external decision.
        /// Identical to the `tool_use_id` carried on the upstream
        /// `ContentPart::ToolUse` block; the resumer routes the
        /// `ApprovalDecision` to the correct in-flight call.
        tool_use_id: String,
    },
    /// A graph-scheduled pause point fired
    /// (`StateGraph::interrupt_before` / `interrupt_after`). The
    /// `phase` and `node` together identify which node the pause
    /// fired around — the resumer reads them to surface
    /// `"paused before <node-name>"` / `"paused after <node-name>"`
    /// without inspecting the payload's free-form structure.
    ScheduledPause {
        /// Whether the pause fired before or after the node ran.
        phase: InterruptionPhase,
        /// Name of the node the pause is anchored to.
        node: String,
    },
}

/// Pause the current dispatch with a [`InterruptionKind::Custom`]
/// reason. The most common HITL primitive — wrap any tool-body
/// branch, node body, or layer hook that needs to hand control back
/// to the caller for human review.
///
/// Returns `Err(Error::Interrupted { kind: Custom, payload })` so
/// call sites read as `return interrupt(value);`. For typed kinds,
/// reach for [`interrupt_with`].
pub fn interrupt<T>(payload: serde_json::Value) -> Result<T, Error> {
    Err(Error::Interrupted {
        kind: InterruptionKind::Custom,
        payload,
    })
}

/// Pause the current dispatch with a typed [`InterruptionKind`].
/// Used by SDK-internal sites (the agent runtime's `ApprovalLayer`
/// raises [`InterruptionKind::ApprovalPending`] this way) and by
/// operators with structured pause-reasons of their own.
///
/// `payload` is operator free-form context that survives round-trip
/// through the `SessionLog`. Pass `Value::Null` when the typed kind
/// already carries everything the resumer needs.
pub fn interrupt_with<T>(kind: InterruptionKind, payload: serde_json::Value) -> Result<T, Error> {
    Err(Error::Interrupted { kind, payload })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn interrupt_returns_custom_kind() {
        let err = interrupt::<()>(json!({"need": "review"})).unwrap_err();
        match err {
            Error::Interrupted { kind, payload } => {
                assert!(matches!(kind, InterruptionKind::Custom));
                assert_eq!(payload, json!({"need": "review"}));
            }
            other => panic!("expected Interrupted, got {other:?}"),
        }
    }

    #[test]
    fn interrupt_with_carries_typed_kind() {
        let err = interrupt_with::<()>(
            InterruptionKind::ApprovalPending {
                tool_use_id: "tu-1".into(),
            },
            serde_json::Value::Null,
        )
        .unwrap_err();
        match err {
            Error::Interrupted { kind, .. } => {
                assert_eq!(
                    kind,
                    InterruptionKind::ApprovalPending {
                        tool_use_id: "tu-1".into()
                    }
                );
            }
            other => panic!("expected Interrupted, got {other:?}"),
        }
    }

    #[test]
    fn interruption_kind_serializes_with_typed_tag() {
        let custom = serde_json::to_value(&InterruptionKind::Custom).unwrap();
        assert_eq!(custom, json!({"type": "custom"}));
        let approval = serde_json::to_value(&InterruptionKind::ApprovalPending {
            tool_use_id: "tu-1".into(),
        })
        .unwrap();
        assert_eq!(
            approval,
            json!({"type": "approval_pending", "tool_use_id": "tu-1"})
        );
    }
}
