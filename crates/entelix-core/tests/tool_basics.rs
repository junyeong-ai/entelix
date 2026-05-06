//! Smoke tests for the `Tool` trait.
//!
//! Defines a tiny in-test `EchoTool` and exercises metadata access /
//! execute. Real tool implementations land in `entelix-tools`.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::unnecessary_literal_bound
)]

use async_trait::async_trait;
use entelix_core::tools::{Tool, ToolMetadata};
use entelix_core::{AgentContext, Result};

struct EchoTool {
    metadata: ToolMetadata,
}

impl EchoTool {
    fn new() -> Self {
        Self {
            metadata: ToolMetadata::function(
                "echo",
                "Returns its input verbatim.",
                serde_json::json!({
                    "type": "object",
                    "properties": { "message": { "type": "string" } },
                    "required": ["message"],
                }),
            ),
        }
    }
}

#[async_trait]
impl Tool for EchoTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        _ctx: &AgentContext<()>,
    ) -> Result<serde_json::Value> {
        Ok(input)
    }
}

#[tokio::test]
async fn tool_metadata_is_accessible() {
    let t = EchoTool::new();
    let m = t.metadata();
    assert_eq!(m.name, "echo");
    assert_eq!(m.description, "Returns its input verbatim.");
    assert_eq!(m.input_schema["type"], "object");
}

#[tokio::test]
async fn tool_execute_returns_input() -> Result<()> {
    let t = EchoTool::new();
    let ctx = AgentContext::default();
    let input = serde_json::json!({ "message": "hello" });
    let out = t.execute(input.clone(), &ctx).await?;
    assert_eq!(out, input);
    Ok(())
}

#[tokio::test]
async fn tool_object_safe_via_arc_dyn() {
    use std::sync::Arc;
    let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(EchoTool::new())];
    assert_eq!(tools[0].metadata().name, "echo");
}
