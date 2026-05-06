//! `ToolToRunnableAdapter` integration test — proves a `Tool` composes into
//! a `.pipe()` chain (closing the ADR-0011 promise).

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::unnecessary_literal_bound
)]

use async_trait::async_trait;
use entelix_core::tools::{Tool, ToolMetadata};
use entelix_core::{AgentContext, ExecutionContext, Result};
use entelix_runnable::{Runnable, RunnableExt, RunnableLambda, ToolToRunnableAdapter};

/// Tool that doubles a numeric `value` field.
struct DoubleTool {
    metadata: ToolMetadata,
}

impl DoubleTool {
    fn new() -> Self {
        Self {
            metadata: ToolMetadata::function(
                "double",
                "Doubles the integer in the `value` field.",
                serde_json::json!({
                    "type": "object",
                    "properties": { "value": { "type": "integer" } },
                    "required": ["value"],
                }),
            ),
        }
    }
}

#[async_trait]
impl Tool for DoubleTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        _ctx: &AgentContext<()>,
    ) -> Result<serde_json::Value> {
        let n = input
            .get("value")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(0);
        Ok(serde_json::json!({ "value": n * 2 }))
    }
}

#[tokio::test]
async fn adapter_invokes_tool_via_runnable_trait() -> Result<()> {
    let ctx = ExecutionContext::new();
    let adapter = ToolToRunnableAdapter::new(DoubleTool::new());
    let out = adapter
        .invoke(serde_json::json!({ "value": 21 }), &ctx)
        .await?;
    assert_eq!(out, serde_json::json!({ "value": 42 }));
    Ok(())
}

#[tokio::test]
async fn adapter_pipes_into_lambda() -> Result<()> {
    let ctx = ExecutionContext::new();

    // Tool reads `value: int`, returns `value: 2*int`.
    let tool = ToolToRunnableAdapter::new(DoubleTool::new());

    // Then a lambda renames the field.
    let rename = RunnableLambda::new(|v: serde_json::Value, _ctx| async move {
        let n = v
            .get("value")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(0);
        Ok::<_, _>(serde_json::json!({ "doubled": n }))
    });

    let chain = tool.pipe(rename);
    let out = chain
        .invoke(serde_json::json!({ "value": 5 }), &ctx)
        .await?;
    assert_eq!(out, serde_json::json!({ "doubled": 10 }));
    Ok(())
}

#[tokio::test]
async fn adapter_exposes_tool_metadata() {
    let adapter = ToolToRunnableAdapter::new(DoubleTool::new());
    assert_eq!(adapter.inner().metadata().name, "double");
    assert_eq!(adapter.name(), "double");
}

#[tokio::test]
async fn adapter_clones_share_inner_arc() {
    let adapter = ToolToRunnableAdapter::new(DoubleTool::new());
    let cloned = adapter.clone();
    // Both should point at the same underlying tool.
    assert!(std::sync::Arc::ptr_eq(adapter.inner(), cloned.inner()));
}
