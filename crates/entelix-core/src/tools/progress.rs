//! Tool-phase reporting primitives.
//!
//! Long-running tools surface in-flight status to operators between
//! the `ToolStart` and `ToolComplete` events by calling
//! [`crate::AgentContext::record_phase`] /
//! [`crate::ExecutionContext::record_phase`] at meaningful work
//! boundaries — schema lookup, vector search, validation, retry
//! arm — and the runtime fans the transition into the
//! [`ToolProgressSink`] attached on the request scope.
//!
//! Two markers ride [`crate::ExecutionContext`] extensions to make the
//! emit path zero-allocation in the absent-sink case:
//!
//! - [`ToolProgressSinkHandle`] — operator wires this once at the
//!   request boundary; tools never construct it.
//! - [`CurrentToolInvocation`] — the layer that brackets tool
//!   dispatch (`ToolEventLayer` in `entelix-agents`) attaches this on
//!   entry so phase emissions correlate to a stable
//!   `(tool_use_id, tool_name, started_at)` triple without the tool
//!   author plumbing identity by hand.
//!
//! When either marker is missing the emit is a silent no-op — the
//! `tracing::info!` discipline (no subscriber → no work). This keeps
//! library tools cost-free to call regardless of whether the embedder
//! cares about phase telemetry.

use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::Result;
use crate::identity::validate_request_identifier;

/// Status transition of one named phase inside a tool dispatch.
///
/// Phases form a state machine per `(tool_use_id, phase)` pair —
/// `Started` opens the phase, optional `Running` updates flow while
/// the phase is in progress, and exactly one terminal `Completed` or
/// `Failed` closes it. Sinks may flatten or drop intermediate
/// `Running` updates; the contract is per-tool-author.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ToolProgressStatus {
    /// The phase has begun. The first transition emitted for a phase.
    Started,
    /// The phase is still running. Intermediate progress (percent
    /// complete, item count, partial result count) flows here.
    Running,
    /// The phase finished successfully. Terminal transition.
    Completed,
    /// The phase finished with a failure. Terminal transition. The
    /// tool may still return its own error through `Tool::execute`.
    Failed,
}

/// One phase transition emitted by a tool.
///
/// Sinks reconstruct per-phase wall-clock duration from successive
/// transitions on the same `(tool_use_id, phase)`. The SDK does not
/// hold phase-state on behalf of the tool — the `dispatch_elapsed_ms`
/// marker is a timeline reference, not a per-phase length.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolProgress {
    /// Per-run correlation id. Echoes
    /// [`crate::ExecutionContext::run_id`] so sinks can join phase
    /// telemetry against `AgentEvent` streams. Empty string when the
    /// dispatch did not originate inside an agent run.
    pub run_id: String,
    /// Stable tool-use id matching the originating model `ToolUse`
    /// block. Pairs with `AgentEvent::ToolStart` / `ToolComplete` /
    /// `ToolError` for the same dispatch.
    pub tool_use_id: String,
    /// Tool name being dispatched.
    pub tool_name: String,
    /// Phase identifier — `schema_lookup`, `vector_search`,
    /// `validation`, `correction_retry`, ... The tool author picks the
    /// vocabulary; sinks treat it as an opaque key.
    pub phase: String,
    /// Status transition for the phase.
    pub status: ToolProgressStatus,
    /// Wall-clock elapsed since the tool dispatch began. Acts as a
    /// timeline marker across every phase in the same dispatch —
    /// per-phase duration is the difference between successive
    /// transitions on the same `(tool_use_id, phase)` pair.
    pub dispatch_elapsed_ms: u64,
    /// Optional structured metadata for UIs and telemetry sinks.
    /// `Value::Null` when the tool author chose not to attach
    /// anything.
    pub metadata: Value,
}

/// Marker attached to [`crate::ExecutionContext`] extensions while a
/// tool dispatch is in flight. The layer that brackets dispatch
/// (`ToolEventLayer` in `entelix-agents`) is the only producer; tools
/// read it indirectly through the `record_phase` helpers and never
/// construct it themselves.
#[derive(Clone, Debug)]
pub struct CurrentToolInvocation {
    tool_use_id: String,
    tool_name: String,
    started_at: Instant,
}

impl CurrentToolInvocation {
    /// Build a marker for one dispatch. Validates that `tool_use_id`
    /// and `tool_name` are well-formed request identifiers (no
    /// control characters, no whitespace-only strings) so a malformed
    /// dispatch never poisons the phase channel.
    pub fn new(tool_use_id: impl Into<String>, tool_name: impl Into<String>) -> Result<Self> {
        let tool_use_id =
            validated_identifier("CurrentToolInvocation::new", "tool_use_id", tool_use_id)?;
        let tool_name = validated_identifier("CurrentToolInvocation::new", "tool_name", tool_name)?;
        Ok(Self {
            tool_use_id,
            tool_name,
            started_at: Instant::now(),
        })
    }

    /// Stable tool-use id for this dispatch.
    #[must_use]
    pub fn tool_use_id(&self) -> &str {
        &self.tool_use_id
    }

    /// Tool name for this dispatch.
    #[must_use]
    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }

    /// Wall-clock elapsed since dispatch start, saturated to
    /// `u64::MAX` milliseconds. Sinks use this as a per-dispatch
    /// timeline reference.
    #[must_use]
    pub fn dispatch_elapsed_ms(&self) -> u64 {
        u64::try_from(self.started_at.elapsed().as_millis()).unwrap_or(u64::MAX)
    }
}

fn validated_identifier(surface: &str, field: &str, value: impl Into<String>) -> Result<String> {
    let value = value.into();
    validate_request_identifier(&format!("{surface}: {field}"), &value)?;
    Ok(value)
}

/// Consumer of tool-phase transitions.
///
/// Operators wire one impl (UI dashboard, OTel event stream, log
/// channel) into [`crate::ExecutionContext`] via
/// [`ToolProgressSinkHandle`] and tools call [`record_phase`] inside
/// their bodies. Distinct from [`crate::AuditSink`] (lifecycle audit
/// trail, fire-and-forget by invariant 18) and from
/// [`crate::events::EventBus`] (token-stream deltas) — phases sit at
/// the granularity of *named steps inside one tool's work*.
///
/// [`record_phase`]: crate::ExecutionContext::record_phase
#[async_trait]
pub trait ToolProgressSink: Send + Sync + 'static {
    /// Record one phase transition. Returning `Err` propagates to the
    /// emit site so tools that must observe the channel (streaming UI
    /// heartbeat) can react; best-effort callers `let _ = ...` it.
    async fn record_progress(&self, progress: ToolProgress) -> Result<()>;
}

/// Refcounted handle to a [`ToolProgressSink`]. Operators attach one
/// per request via [`crate::ExecutionContext::with_tool_progress_sink`]
/// and downstream layers / tools resolve it through the typed
/// extension lookup.
#[derive(Clone)]
pub struct ToolProgressSinkHandle {
    sink: Arc<dyn ToolProgressSink>,
}

impl ToolProgressSinkHandle {
    /// Wrap a concrete sink.
    #[must_use]
    pub fn new<S>(sink: S) -> Self
    where
        S: ToolProgressSink,
    {
        Self {
            sink: Arc::new(sink),
        }
    }

    /// Wrap an existing trait-object sink.
    #[must_use]
    pub fn from_arc(sink: Arc<dyn ToolProgressSink>) -> Self {
        Self { sink }
    }

    /// Borrow the underlying sink for direct dispatch — used by the
    /// `record_phase` emit path on [`crate::ExecutionContext`].
    #[must_use]
    pub fn inner(&self) -> &Arc<dyn ToolProgressSink> {
        &self.sink
    }
}

impl std::fmt::Debug for ToolProgressSinkHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolProgressSinkHandle")
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_tool_invocation_rejects_invalid_identity() {
        let Err(err) = CurrentToolInvocation::new(" ", "echo") else {
            panic!("expected invalid tool_use_id to fail");
        };
        assert!(format!("{err}").contains("tool_use_id must not be empty"));

        let Err(err) = CurrentToolInvocation::new("tu-1", "echo\nnext") else {
            panic!("expected invalid tool_name to fail");
        };
        assert!(format!("{err}").contains("tool_name must not contain control characters"));
    }

    #[test]
    fn current_tool_invocation_accepts_valid_identity() -> Result<()> {
        let current = CurrentToolInvocation::new("tu-1", "echo")?;
        assert_eq!(current.tool_use_id(), "tu-1");
        assert_eq!(current.tool_name(), "echo");
        Ok(())
    }
}
