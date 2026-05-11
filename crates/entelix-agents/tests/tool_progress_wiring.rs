//! Regression — `ToolEventLayer` attaches `CurrentToolInvocation` on
//! dispatch entry, and leaf-tool `ctx.record_phase(...)` calls fan
//! into the operator-attached `ToolProgressSink`.
//!
//! Four cases pin the contract:
//! 1. Sink + layer wired → every phase transition reaches the sink.
//! 2. Sink wired, dispatch happens outside the layer → no-op (no
//!    `CurrentToolInvocation` to correlate against).
//! 3. Layer wired, no sink → no-op (sink-absent silent fail closed).
//! 4. Dispatch outside any tool (raw `ctx.record_phase`) → no-op.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use serde_json::{Value, json};

use entelix_agents::{AgentEventSink, DroppingSink, ToolEventLayer};
use entelix_core::tools::{Tool, ToolEffect, ToolMetadata, ToolRegistry};
use entelix_core::{
    AgentContext, CurrentToolInvocation, ExecutionContext, Result, ToolProgress, ToolProgressSink,
    ToolProgressSinkHandle, ToolProgressStatus,
};

#[derive(Default)]
struct RecordingSink {
    captured: Mutex<Vec<ToolProgress>>,
}

#[async_trait]
impl ToolProgressSink for RecordingSink {
    async fn record_progress(&self, progress: ToolProgress) -> Result<()> {
        self.captured.lock().push(progress);
        Ok(())
    }
}

impl RecordingSink {
    fn snapshot(&self) -> Vec<ToolProgress> {
        self.captured.lock().clone()
    }
}

#[derive(Debug)]
struct PhasedTool {
    metadata: ToolMetadata,
}

impl PhasedTool {
    fn new() -> Self {
        Self {
            metadata: ToolMetadata::function(
                "phased",
                "Tool emitting three phases per dispatch.",
                json!({"type": "object"}),
            )
            .with_effect(ToolEffect::ReadOnly),
        }
    }
}

#[async_trait]
impl Tool<()> for PhasedTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    async fn execute(&self, _input: Value, ctx: &AgentContext) -> Result<Value> {
        ctx.record_phase("schema_lookup", ToolProgressStatus::Started)
            .await?;
        ctx.record_phase("schema_lookup", ToolProgressStatus::Completed)
            .await?;
        ctx.record_phase_with(
            "validation",
            ToolProgressStatus::Completed,
            json!({"rows_checked": 42}),
        )
        .await?;
        Ok(json!({"ok": true}))
    }
}

async fn dispatch_through_layer(ctx: ExecutionContext) -> Result<Value> {
    // DroppingSink swallows AgentEvents — we care about
    // `ToolProgressSink`, not `AgentEvent` wiring here.
    let event_sink: Arc<dyn AgentEventSink<()>> = Arc::new(DroppingSink);
    let registry = ToolRegistry::<()>::new()
        .layer(ToolEventLayer::<()>::new(event_sink))
        .register(Arc::new(PhasedTool::new()))?;
    registry.dispatch("tu-1", "phased", json!({}), &ctx).await
}

#[tokio::test]
async fn sink_and_layer_wired_yields_every_phase() -> Result<()> {
    let sink = Arc::new(RecordingSink::default());
    let handle = ToolProgressSinkHandle::from_arc(sink.clone());
    let ctx = ExecutionContext::new()
        .with_run_id("run-1")
        .with_tool_progress_sink(handle);

    dispatch_through_layer(ctx).await?;

    let captured = sink.snapshot();
    assert_eq!(captured.len(), 3, "three phase transitions expected");

    assert_eq!(captured[0].phase, "schema_lookup");
    assert_eq!(captured[0].status, ToolProgressStatus::Started);
    assert_eq!(captured[0].tool_name, "phased");
    assert_eq!(captured[0].tool_use_id, "tu-1");
    assert_eq!(captured[0].run_id, "run-1");
    assert_eq!(captured[0].metadata, Value::Null);

    assert_eq!(captured[1].phase, "schema_lookup");
    assert_eq!(captured[1].status, ToolProgressStatus::Completed);

    assert_eq!(captured[2].phase, "validation");
    assert_eq!(captured[2].metadata, json!({"rows_checked": 42}));

    // `dispatch_elapsed_ms` is monotonic non-decreasing across the
    // three transitions in the same dispatch.
    assert!(captured[0].dispatch_elapsed_ms <= captured[1].dispatch_elapsed_ms);
    assert!(captured[1].dispatch_elapsed_ms <= captured[2].dispatch_elapsed_ms);

    Ok(())
}

#[tokio::test]
async fn sink_without_layer_attached_marker_is_no_op() -> Result<()> {
    // Sink attached but no `CurrentToolInvocation` marker — calling
    // `ctx.record_phase(...)` from outside a dispatch silently no-ops.
    let sink = Arc::new(RecordingSink::default());
    let handle = ToolProgressSinkHandle::from_arc(sink.clone());
    let ctx = ExecutionContext::new()
        .with_run_id("run-2")
        .with_tool_progress_sink(handle);

    ctx.record_phase("orphan", ToolProgressStatus::Started)
        .await?;
    ctx.record_phase("orphan", ToolProgressStatus::Completed)
        .await?;

    assert!(
        sink.snapshot().is_empty(),
        "phase emit outside any dispatch must silently no-op"
    );
    Ok(())
}

#[tokio::test]
async fn layer_attaches_marker_so_dispatch_can_emit_without_explicit_setup() -> Result<()> {
    // Same as case 1 minus the assertions on payload — the focus is
    // that the layer attaches a marker the inner tool can resolve
    // without any plumbing in the tool body.
    let sink = Arc::new(RecordingSink::default());
    let handle = ToolProgressSinkHandle::from_arc(sink.clone());
    let ctx = ExecutionContext::new().with_tool_progress_sink(handle);
    dispatch_through_layer(ctx).await?;
    assert!(!sink.snapshot().is_empty());
    Ok(())
}

#[tokio::test]
async fn dispatch_without_sink_does_not_panic_or_error() -> Result<()> {
    // Marker attached (by the layer) but no sink — emit silently
    // no-ops. The tool body still runs to completion.
    let ctx = ExecutionContext::new().with_run_id("run-3");
    let value = dispatch_through_layer(ctx).await?;
    assert_eq!(value, json!({"ok": true}));
    Ok(())
}

#[tokio::test]
async fn current_tool_invocation_marker_resolves_inside_dispatch() -> Result<()> {
    let event_sink: Arc<dyn AgentEventSink<()>> = Arc::new(DroppingSink);
    let registry = ToolRegistry::<()>::new()
        .layer(ToolEventLayer::<()>::new(event_sink))
        .register(Arc::new(MarkerInspectingTool::new()))?;
    let captured: Arc<Mutex<Option<(String, String)>>> = Arc::new(Mutex::new(None));
    let ctx = ExecutionContext::new().add_extension(captured.clone());
    registry
        .dispatch("tu-marker", "marker_inspecting", json!({}), &ctx)
        .await?;
    let got = captured.lock().clone().expect("marker observed");
    assert_eq!(
        got,
        ("tu-marker".to_owned(), "marker_inspecting".to_owned())
    );
    Ok(())
}

#[derive(Debug)]
struct MarkerInspectingTool {
    metadata: ToolMetadata,
}

impl MarkerInspectingTool {
    fn new() -> Self {
        Self {
            metadata: ToolMetadata::function(
                "marker_inspecting",
                "Captures the CurrentToolInvocation marker for assertion.",
                json!({"type": "object"}),
            ),
        }
    }
}

#[async_trait]
impl Tool<()> for MarkerInspectingTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    async fn execute(&self, _input: Value, ctx: &AgentContext) -> Result<Value> {
        let captured = ctx
            .extension::<Arc<Mutex<Option<(String, String)>>>>()
            .expect("captured slot present");
        let marker = ctx
            .extension::<CurrentToolInvocation>()
            .expect("marker attached by ToolEventLayer");
        *captured.lock() = Some((
            marker.tool_use_id().to_owned(),
            marker.tool_name().to_owned(),
        ));
        Ok(Value::Null)
    }
}
