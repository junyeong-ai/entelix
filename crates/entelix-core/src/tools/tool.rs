//! The `Tool` trait — the Hand contract (invariant 4).
//!
//! `Tool` is intentionally minimal: a single [`ToolMetadata`] accessor
//! and a single `execute(input, ctx) → output` method. No `prepare`,
//! no `cleanup`, no back-channels. Anything else lives in
//! [`crate::context::ExecutionContext`] or in adapter types.
//!
//! Per ADR-0011, `Tool` does NOT extend `Runnable`. The
//! `entelix-runnable::adapters::ToolToRunnableAdapter` provides the
//! bridge when composition (`.pipe()`) is needed.

use async_trait::async_trait;

use crate::context::ExecutionContext;
use crate::error::Result;
use crate::tools::metadata::ToolMetadata;

/// A capability the agent can dispatch.
///
/// Implementors hold a [`ToolMetadata`] (typically constructed once
/// in `new()`) and return a borrow from [`Tool::metadata`]. The
/// runtime treats that struct as authoritative — codecs render it
/// into the on-the-wire `ToolSpec`, OTel layers stamp
/// `gen_ai.tool.*` attributes, and `Approver` defaults route off
/// `metadata.effect`.
///
/// # Implementing a tool
///
/// ```ignore
/// use async_trait::async_trait;
/// use entelix_core::tools::{Tool, ToolMetadata};
/// use entelix_core::{ExecutionContext, Result};
///
/// pub struct EchoTool {
///     metadata: ToolMetadata,
/// }
///
/// impl EchoTool {
///     pub fn new() -> Self {
///         Self {
///             metadata: ToolMetadata::function(
///                 "echo",
///                 "Echoes its input verbatim.",
///                 serde_json::json!({ "type": "object" }),
///             ),
///         }
///     }
/// }
///
/// #[async_trait]
/// impl Tool for EchoTool {
///     fn metadata(&self) -> &ToolMetadata {
///         &self.metadata
///     }
///
///     async fn execute(
///         &self,
///         input: serde_json::Value,
///         _ctx: &ExecutionContext,
///     ) -> Result<serde_json::Value> {
///         Ok(input)
///     }
/// }
/// ```
#[async_trait]
pub trait Tool: Send + Sync + 'static {
    /// Borrow this tool's descriptor. Cheap — implementors return a
    /// reference to a field they constructed once.
    fn metadata(&self) -> &ToolMetadata;

    /// Run the tool against `input`. The `ctx` argument carries
    /// cancellation token, deadline, and tenant scope — but
    /// **never** credentials (invariant 10).
    async fn execute(
        &self,
        input: serde_json::Value,
        ctx: &ExecutionContext,
    ) -> Result<serde_json::Value>;
}
