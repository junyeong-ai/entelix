//! `ToolToRunnableAdapter` — bridges the `Tool` trait into the `Runnable`
//! composition contract per ADR-0011.
//!
//! `Tool` lives in `entelix-core` (DAG root) and does not extend `Runnable`.
//! When a user wants to drop a tool into a `.pipe()` chain, they wrap it
//! here:
//!
//! ```ignore
//! use entelix_runnable::{RunnableExt, ToolToRunnableAdapter};
//! let chain = prompt.pipe(model).pipe(ToolToRunnableAdapter::new(my_tool));
//! ```
//!
//! The adapter forwards `invoke` to `Tool::execute`; metadata
//! (`name`, `description`, `input_schema`) stays accessible via
//! `inner()` for callers that need it.

use std::sync::Arc;

use entelix_core::tools::Tool;
use entelix_core::{AgentContext, ExecutionContext, Result};

use crate::runnable::Runnable;

/// Wraps any `Tool` as `Runnable<serde_json::Value, serde_json::Value>`.
///
/// Cheap to clone (internal `Arc`).
pub struct ToolToRunnableAdapter {
    inner: Arc<dyn Tool>,
}

impl ToolToRunnableAdapter {
    /// Wrap a concrete `Tool`.
    pub fn new<T: Tool>(tool: T) -> Self {
        Self {
            inner: Arc::new(tool),
        }
    }

    /// Wrap an already-shared `Arc<dyn Tool>` (e.g. one stored in a
    /// `ToolRegistry`). Avoids a needless `Arc::new`.
    pub fn from_arc(tool: Arc<dyn Tool>) -> Self {
        Self { inner: tool }
    }

    /// Borrow the wrapped tool — useful for inspecting metadata.
    pub fn inner(&self) -> &Arc<dyn Tool> {
        &self.inner
    }
}

impl Clone for ToolToRunnableAdapter {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

#[async_trait::async_trait]
impl Runnable<serde_json::Value, serde_json::Value> for ToolToRunnableAdapter {
    async fn invoke(
        &self,
        input: serde_json::Value,
        ctx: &ExecutionContext,
    ) -> Result<serde_json::Value> {
        // Bridge `Runnable` (D-free, ExecutionContext) → `Tool`
        // (typed `D`). Adapter-wrapped tools are always `Tool<()>`
        // because composition does not carry typed deps; operators
        // threading deps reach for the typed registry path
        // (slice 103).
        let agent_ctx = AgentContext::<()>::from(ctx.clone());
        self.inner.execute(input, &agent_ctx).await
    }

    fn name(&self) -> std::borrow::Cow<'_, str> {
        std::borrow::Cow::Borrowed(&self.inner.metadata().name)
    }
}
