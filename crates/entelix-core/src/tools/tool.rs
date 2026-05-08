//! The `Tool` trait — the Hand contract (invariant 4).
//!
//! `Tool<D>` is intentionally minimal: a single [`ToolMetadata`]
//! accessor and a single `execute(input, ctx) → output` method. No
//! `prepare`, no `cleanup`, no back-channels. Anything else lives
//! in [`crate::AgentContext`] (request-scope carrier with infra
//! context + typed operator-side deps `D`) or in adapter types.
//!
//! `D` defaults to `()` so deps-less tools stay `Tool` without an
//! annotation. Tools that need typed operator-side handles (DB
//! pool, HTTP client, tenant config) declare `Tool<MyDeps>` and
//! reach for them via `ctx.deps()`. .
//!
//! Per, `Tool` does NOT extend `Runnable`. The
//! `entelix-runnable::adapter::ToolToRunnableAdapter` provides the
//! bridge when composition (`.pipe()`) is needed.

use async_trait::async_trait;

use crate::agent_context::AgentContext;
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
/// use entelix_core::{AgentContext, Result};
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
///         _ctx: &AgentContext,
///     ) -> Result<serde_json::Value> {
///         Ok(input)
///     }
/// }
/// ```
#[async_trait]
pub trait Tool<D = ()>: Send + Sync + 'static
where
    D: Send + Sync + 'static,
{
    /// Borrow this tool's descriptor. Cheap — implementors return a
    /// reference to a field they constructed once.
    fn metadata(&self) -> &ToolMetadata;

    /// Run the tool against `input`. `ctx` carries the infra
    /// context (cancellation, deadline, tenant scope — reachable
    /// via `ctx.core()` or the forwarder accessors) and the typed
    /// operator-side deps `D` (reachable via `ctx.deps()`).
    /// Credentials never appear in either slot (invariant 10).
    async fn execute(
        &self,
        input: serde_json::Value,
        ctx: &AgentContext<D>,
    ) -> Result<serde_json::Value>;
}
