//! Lifecycle-event coverage:
//!
//! - `Failed{run_id, error}` emitted on the sink when the inner
//!   runnable returns an error (mirrors `Complete{run_id, state}`).
//! - `run_id` is shared by `Started` and the matching terminal
//!   event for the same call.
//! - `ToolEventLayer` emits `ToolStart` / `ToolComplete` /
//!   `ToolError` for tool dispatches that flow through the layer
//!   while a run is in progress.

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use std::sync::Arc;

use async_trait::async_trait;
use entelix_agents::{Agent, AgentEvent, CaptureSink, ToolEventLayer};
use entelix_core::context::ExecutionContext;
use entelix_core::AgentContext;
use entelix_core::error::{Error, Result};
use entelix_core::tools::{Tool, ToolMetadata, ToolRegistry};
use entelix_runnable::{Runnable, RunnableLambda};
use serde_json::{Value, json};
use tower::ServiceExt;

fn echo_runnable() -> impl Runnable<i32, i32> {
    RunnableLambda::new(|n: i32, _ctx| async move { Ok::<_, _>(n + 1) })
}

fn failing_runnable() -> impl Runnable<i32, i32> {
    RunnableLambda::new(|_n: i32, _ctx| async move {
        Err::<i32, _>(Error::provider_http(503, "transient backend"))
    })
}

#[tokio::test]
async fn started_and_complete_share_the_same_run_id() {
    let sink = CaptureSink::<i32>::new();
    let agent = Agent::<i32>::builder()
        .with_name("scope")
        .with_runnable(echo_runnable())
        .with_sink(sink.clone())
        .build()
        .unwrap();
    agent.execute(0, &ExecutionContext::new()).await.unwrap();
    let events = sink.events();
    assert_eq!(events.len(), 2);
    let started_run_id = match &events[0] {
        AgentEvent::Started { run_id, .. } => run_id.clone(),
        other => panic!("expected Started, got {other:?}"),
    };
    let complete_run_id = match &events[1] {
        AgentEvent::Complete { run_id, .. } => run_id.clone(),
        other => panic!("expected Complete, got {other:?}"),
    };
    assert_eq!(
        started_run_id, complete_run_id,
        "Started and Complete must share the same run_id"
    );
    assert!(!started_run_id.is_empty(), "run_id must be assigned");
}

#[tokio::test]
async fn caller_supplied_run_id_is_inherited() {
    let sink = CaptureSink::<i32>::new();
    let agent = Agent::<i32>::builder()
        .with_name("inherit")
        .with_runnable(echo_runnable())
        .with_sink(sink.clone())
        .build()
        .unwrap();
    let ctx = ExecutionContext::new().with_run_id("caller-supplied-7");
    agent.execute(0, &ctx).await.unwrap();
    let events = sink.events();
    for event in &events {
        let id = match event {
            AgentEvent::Started { run_id, .. } | AgentEvent::Complete { run_id, .. } => run_id,
            other => panic!("unexpected event: {other:?}"),
        };
        assert_eq!(id, "caller-supplied-7");
    }
}

#[tokio::test]
async fn failed_event_is_emitted_when_inner_runnable_errors() {
    let sink = CaptureSink::<i32>::new();
    let agent = Agent::<i32>::builder()
        .with_name("fails")
        .with_runnable(failing_runnable())
        .with_sink(sink.clone())
        .build()
        .unwrap();
    let err = agent
        .execute(0, &ExecutionContext::new())
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        Error::Provider {
            kind: entelix_core::ProviderErrorKind::Http(503),
            ..
        }
    ));
    let events = sink.events();
    assert_eq!(events.len(), 2);
    assert!(matches!(events[0], AgentEvent::Started { .. }));
    let (run_id, error) = match &events[1] {
        AgentEvent::Failed { run_id, error } => (run_id.clone(), error.clone()),
        other => panic!("expected Failed, got {other:?}"),
    };
    let started_run_id = match &events[0] {
        AgentEvent::Started { run_id, .. } => run_id.clone(),
        _ => unreachable!(),
    };
    assert_eq!(run_id, started_run_id);
    assert!(error.contains("transient backend"));
}

#[tokio::test]
async fn execute_stream_emits_failed_then_typed_err_on_caller_side() {
    use futures::StreamExt;

    let sink = CaptureSink::<i32>::new();
    let agent = Agent::<i32>::builder()
        .with_name("stream-fails")
        .with_runnable(failing_runnable())
        .with_sink(sink.clone())
        .build()
        .unwrap();
    let ctx = ExecutionContext::new();
    let mut stream = agent.execute_stream(0, &ctx);
    let mut yielded: Vec<Result<AgentEvent<i32>>> = Vec::new();
    while let Some(item) = stream.next().await {
        yielded.push(item);
    }
    // Caller stream: Started → Failed → Err.
    assert_eq!(yielded.len(), 3);
    assert!(matches!(yielded[0], Ok(AgentEvent::Started { .. })));
    assert!(matches!(yielded[1], Ok(AgentEvent::Failed { .. })));
    assert!(matches!(
        yielded[2],
        Err(Error::Provider {
            kind: entelix_core::ProviderErrorKind::Http(503),
            ..
        })
    ));
    // Sink-side terminal: Started → Failed.
    let sink_events = sink.events();
    assert_eq!(sink_events.len(), 2);
    assert!(matches!(sink_events[1], AgentEvent::Failed { .. }));
}

// ── Tool event layer ───────────────────────────────────────────────────────

#[derive(Debug)]
struct AddOneTool {
    metadata: ToolMetadata,
}

impl AddOneTool {
    fn new() -> Self {
        Self {
            metadata: ToolMetadata::function(
                "add_one",
                "increment input by 1",
                json!({"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}),
            ),
        }
    }
}

#[async_trait]
impl Tool for AddOneTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }
    async fn execute(&self, input: Value, _ctx: &AgentContext<()>) -> Result<Value> {
        let n = input.get("n").and_then(Value::as_i64).unwrap_or(0);
        Ok(json!({"result": n + 1}))
    }
}

#[derive(Debug)]
struct AlwaysFailTool {
    metadata: ToolMetadata,
}

impl AlwaysFailTool {
    fn new() -> Self {
        Self {
            metadata: ToolMetadata::function(
                "always_fail",
                "always returns an error",
                json!({"type": "object", "properties": {}}),
            ),
        }
    }
}

#[async_trait]
impl Tool for AlwaysFailTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }
    async fn execute(&self, _input: Value, _ctx: &AgentContext<()>) -> Result<Value> {
        Err(Error::config("intentional failure"))
    }
}

/// Tool that advertises a stable version string so the agent-event
/// pipeline can be exercised with a non-`None` `tool_version`.
#[derive(Debug)]
struct VersionedTool {
    fail: bool,
    metadata: ToolMetadata,
}

impl VersionedTool {
    fn new(fail: bool) -> Self {
        Self {
            fail,
            metadata: ToolMetadata::function(
                "versioned",
                "tool with a non-empty version",
                json!({"type": "object", "properties": {}}),
            )
            .with_version("1.4.2"),
        }
    }
}

#[async_trait]
impl Tool for VersionedTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }
    async fn execute(&self, _input: Value, _ctx: &AgentContext<()>) -> Result<Value> {
        if self.fail {
            Err(Error::config("versioned tool intentional failure"))
        } else {
            Ok(json!({"ok": true}))
        }
    }
}

#[tokio::test]
async fn tool_event_layer_emits_start_and_complete() {
    let sink = Arc::new(CaptureSink::<i32>::new());
    let registry = ToolRegistry::new()
        .layer(ToolEventLayer::new(sink.clone() as Arc<_>))
        .register(Arc::new(AddOneTool::new()))
        .unwrap();
    let ctx = ExecutionContext::new().with_run_id("run-abc");
    let out = registry
        .dispatch("tool_use_42", "add_one", json!({"n": 5}), &ctx)
        .await
        .unwrap();
    assert_eq!(out["result"], 6);
    let events = sink.events();
    assert_eq!(events.len(), 2);
    match &events[0] {
        AgentEvent::ToolStart {
            run_id,
            tool_use_id,
            tool,
            ..
        } => {
            assert_eq!(run_id, "run-abc");
            assert_eq!(tool_use_id, "tool_use_42");
            assert_eq!(tool, "add_one");
        }
        other => panic!("expected ToolStart, got {other:?}"),
    }
    match &events[1] {
        AgentEvent::ToolComplete {
            run_id,
            tool_use_id,
            tool,
            ..
        } => {
            assert_eq!(run_id, "run-abc");
            assert_eq!(tool_use_id, "tool_use_42");
            assert_eq!(tool, "add_one");
        }
        other => panic!("expected ToolComplete, got {other:?}"),
    }
}

#[tokio::test]
async fn tool_event_layer_emits_error_on_tool_failure() {
    let sink = Arc::new(CaptureSink::<i32>::new());
    let registry = ToolRegistry::new()
        .layer(ToolEventLayer::new(sink.clone() as Arc<_>))
        .register(Arc::new(AlwaysFailTool::new()))
        .unwrap();
    let ctx = ExecutionContext::new().with_run_id("run-xyz");
    let err = registry
        .dispatch("tool_use_9", "always_fail", json!({}), &ctx)
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Config(_)));
    let events = sink.events();
    assert_eq!(events.len(), 2);
    assert!(matches!(events[0], AgentEvent::ToolStart { .. }));
    match &events[1] {
        AgentEvent::ToolError {
            run_id,
            tool,
            error,
            ..
        } => {
            assert_eq!(run_id, "run-xyz");
            assert_eq!(tool, "always_fail");
            assert!(error.contains("intentional failure"));
        }
        other => panic!("expected ToolError, got {other:?}"),
    }
}

#[tokio::test]
async fn tool_event_layer_propagates_tool_version_to_start_and_complete() {
    let sink = Arc::new(CaptureSink::<i32>::new());
    let registry = ToolRegistry::new()
        .layer(ToolEventLayer::new(sink.clone() as Arc<_>))
        .register(Arc::new(VersionedTool::new(false)))
        .unwrap();
    let ctx = ExecutionContext::new().with_run_id("run-ver-ok");
    registry
        .dispatch("tu_v1", "versioned", json!({}), &ctx)
        .await
        .unwrap();
    let events = sink.events();
    assert_eq!(events.len(), 2);
    match &events[0] {
        AgentEvent::ToolStart { tool_version, .. } => {
            assert_eq!(tool_version.as_deref(), Some("1.4.2"));
        }
        other => panic!("expected ToolStart, got {other:?}"),
    }
    match &events[1] {
        AgentEvent::ToolComplete { tool_version, .. } => {
            assert_eq!(tool_version.as_deref(), Some("1.4.2"));
        }
        other => panic!("expected ToolComplete, got {other:?}"),
    }
}

#[tokio::test]
async fn tool_event_layer_propagates_tool_version_to_error() {
    let sink = Arc::new(CaptureSink::<i32>::new());
    let registry = ToolRegistry::new()
        .layer(ToolEventLayer::new(sink.clone() as Arc<_>))
        .register(Arc::new(VersionedTool::new(true)))
        .unwrap();
    let ctx = ExecutionContext::new().with_run_id("run-ver-err");
    let _ = registry
        .dispatch("tu_v2", "versioned", json!({}), &ctx)
        .await
        .unwrap_err();
    let events = sink.events();
    assert_eq!(events.len(), 2);
    match &events[1] {
        AgentEvent::ToolError { tool_version, .. } => {
            assert_eq!(tool_version.as_deref(), Some("1.4.2"));
        }
        other => panic!("expected ToolError, got {other:?}"),
    }
}

#[tokio::test]
async fn tool_event_layer_is_quiet_without_run_id() {
    // Outside an agent run (no run_id stamped on ctx), the layer
    // is a no-op — passes the call through, emits nothing.
    let sink = Arc::new(CaptureSink::<i32>::new());
    let registry = ToolRegistry::new()
        .layer(ToolEventLayer::new(sink.clone() as Arc<_>))
        .register(Arc::new(AddOneTool::new()))
        .unwrap();
    let ctx = ExecutionContext::new();
    let _ = registry
        .dispatch("tu1", "add_one", json!({"n": 1}), &ctx)
        .await
        .unwrap();
    assert!(
        sink.events().is_empty(),
        "layer must not emit when run_id is absent"
    );
}

#[tokio::test]
async fn tower_layer_compiles_with_oneshot_path() {
    // Compile-time confirmation that ToolEventLayer composes through
    // the tower pipeline — useful when refactoring layer signatures.
    let sink = Arc::new(CaptureSink::<i32>::new());
    let registry = ToolRegistry::new()
        .layer(ToolEventLayer::new(sink as Arc<_>))
        .register(Arc::new(AddOneTool::new()))
        .unwrap();
    let svc = registry.service("add_one").unwrap();
    let ctx = ExecutionContext::new().with_run_id("compile-only");
    let inv = entelix_core::ToolInvocation::new(
        "tu".into(),
        Arc::new(ToolMetadata::function(
            "add_one",
            "increment input by 1",
            json!({"type": "object"}),
        )),
        json!({"n": 0}),
        ctx,
    );
    let _ = svc.oneshot(inv).await.unwrap();
}
