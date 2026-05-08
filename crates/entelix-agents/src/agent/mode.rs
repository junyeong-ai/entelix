//! `ExecutionMode` — agent-level switch between immediate and
//! human-supervised tool execution.
//!
//! - [`ExecutionMode::Auto`] (default) — every tool call dispatched
//!   immediately; the agent surfaces `ToolStart` / `ToolComplete`
//!   events but the operator has no in-flight veto.
//! - [`ExecutionMode::Supervised`] — every tool call is gated by
//!   an [`Approver`](crate::agent::approver::Approver) before the
//!   recipe dispatches it. The recipe pauses on
//!   `AwaitExternal` via the underlying graph's
//!   `interrupt(payload)` until the operator drives a decision
//!   through the approver's reply channel — a single state
//!   machine, no parallel runtime.
//!
//! Marked `#[non_exhaustive]` so future modes (e.g. `Plan` —
//! produce a plan transcript without executing) are not breaking
//! changes.

/// Agent-level switch between immediate and human-supervised tool
/// execution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExecutionMode {
    /// Tool calls execute immediately when the model requests them.
    /// Default for unattended / background agents.
    #[default]
    Auto,
    /// Every tool call is routed through an
    /// [`Approver`](crate::agent::approver::Approver) before
    /// execution. Operators drive `AwaitExternal` decisions
    /// through the approver's reply channel
    /// (`mpsc<PendingApproval>` for `ChannelApprover`).
    Supervised,
}

impl ExecutionMode {
    /// Whether tool calls in this mode require approver consent
    /// before dispatch.
    #[must_use]
    pub const fn requires_approval(self) -> bool {
        matches!(self, Self::Supervised)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_is_default_and_does_not_require_approval() {
        let mode: ExecutionMode = ExecutionMode::default();
        assert_eq!(mode, ExecutionMode::Auto);
        assert!(!mode.requires_approval());
    }

    #[test]
    fn supervised_requires_approval() {
        assert!(ExecutionMode::Supervised.requires_approval());
    }
}
