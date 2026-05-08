//! Tool-progress reporting primitives.
//!
//! Progress is an explicit tool-authored signal, not a runtime
//! heuristic. Long-running tools call [`AgentContext::progress`] or
//! [`ExecutionContext::progress`] at meaningful boundaries; runtimes
//! that care attach a [`ToolProgressSinkHandle`] to the
//! [`ExecutionContext`]. When no sink is attached the call is a
//! no-op, so library tools can report progress without depending on
//! a particular agent runtime.

use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::Result;
use crate::identity::validate_request_identifier;

/// Status of a tool-progress update.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ToolProgressStatus {
    /// A named step has begun.
    Started,
    /// A named step is still running.
    Running,
    /// A named step completed successfully.
    Completed,
    /// A named step failed. The tool may still return its own error
    /// through the normal `Tool::execute` path.
    Failed,
}

/// One explicit progress update emitted by a tool.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolProgress {
    /// Per-run correlation id.
    pub run_id: String,
    /// Stable tool-use id matching the originating model tool call.
    pub tool_use_id: String,
    /// Tool name being dispatched.
    pub tool_name: String,
    /// Human-readable step name.
    pub step: String,
    /// Step status.
    pub status: ToolProgressStatus,
    /// Elapsed wall-clock time since this tool dispatch began.
    pub duration_ms: u64,
    /// Optional structured metadata for UIs and telemetry sinks.
    pub metadata: Value,
}

/// Current tool-dispatch identity stored in request extensions while
/// a tool is executing.
#[derive(Clone, Debug)]
pub struct CurrentToolInvocation {
    tool_use_id: String,
    tool_name: String,
    started_at: Instant,
}

impl CurrentToolInvocation {
    /// Build a current-tool marker for one dispatch.
    pub fn new(tool_use_id: impl Into<String>, tool_name: impl Into<String>) -> Result<Self> {
        let tool_use_id = validated_progress_identifier(
            "CurrentToolInvocation::new",
            "tool_use_id",
            tool_use_id,
        )?;
        let tool_name =
            validated_progress_identifier("CurrentToolInvocation::new", "tool_name", tool_name)?;
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

    /// Elapsed wall-clock time since dispatch start, saturated to
    /// `u64::MAX` milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        u64::try_from(self.started_at.elapsed().as_millis()).unwrap_or(u64::MAX)
    }
}

fn validated_progress_identifier(
    surface: &str,
    field: &str,
    value: impl Into<String>,
) -> Result<String> {
    let value = value.into();
    validate_request_identifier(&format!("{surface}: {field}"), &value)?;
    Ok(value)
}

/// Consumer of explicit tool-progress updates.
///
/// Operators wire one `ToolProgressSink` impl into the
/// `ExecutionContext` (a UI dashboard, an OTel event stream, a
/// log channel) and tools running long enough to warrant inflight
/// status report through it. Distinct from `AgentEvent::ToolInvoked`
/// — that fires once per dispatch lifecycle; progress fires
/// repeatedly *during* one dispatch.
#[async_trait]
pub trait ToolProgressSink: Send + Sync + 'static {
    /// Record one progress update.
    ///
    /// Tools call this indirectly via the `ToolProgressSinkHandle`
    /// they pull off `ExecutionContext::extension`; sinks observe
    /// the `(tool_name, tool_use_id)` identity carried on the
    /// emitted [`ToolProgress`] when the dispatch path attaches it.
    async fn record_progress(&self, progress: ToolProgress) -> Result<()>;
}

/// Refcounted handle stored in [`crate::ExecutionContext`]
/// extensions so tools can report progress without depending on the
/// agent crate.
#[derive(Clone)]
pub struct ToolProgressSinkHandle {
    sink: Arc<dyn ToolProgressSink>,
}

impl ToolProgressSinkHandle {
    /// Wrap a concrete progress sink.
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

    /// Borrow the underlying sink.
    #[must_use]
    pub fn inner(&self) -> &Arc<dyn ToolProgressSink> {
        &self.sink
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
