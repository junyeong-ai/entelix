//! `McpToolAdapter` — bridges an MCP-published tool descriptor onto
//! the [`entelix_core::tools::Tool`] trait so the agent can dispatch
//! it like any first-party tool.
//!
//! One adapter instance binds to one `(server, tool_name)` pair on a
//! shared [`crate::McpManager`]. The adapter holds no connection — it
//! defers to the manager, which routes to the correct
//! `(tenant, server)` `McpClient` based on `AgentContext::tenant_id`.
//!
//! ## Namespacing — `mcp:{server}:{tool}` by default
//!
//! Two MCP servers can each publish a tool named `read`. To prevent
//! silent collision when those adapters are registered into the same
//! [`entelix_core::ToolRegistry`], the adapter exposes its name as
//! `mcp:{server}:{tool}` (the *qualified* name). The unqualified
//! definition name is preserved internally so `McpManager::call_tool`
//! still uses the wire-correct identifier.
//!
//! Single-server deployments that have already trained models on the
//! unqualified name can opt out via [`McpToolAdapter::with_unqualified_name`]
//! — the trade-off is the operator's, not the SDK's. The default is
//! the safe path.

use async_trait::async_trait;
use serde_json::Value;

use entelix_core::AgentContext;
use entelix_core::error::Result;
use entelix_core::tools::{Tool, ToolMetadata};

use crate::manager::McpManager;
use crate::tool_definition::McpToolDefinition;

/// Prefix applied to MCP-sourced tool names so collisions across
/// servers are impossible.
const MCP_NAME_PREFIX: &str = "mcp";

/// Adapter that exposes one MCP tool through the entelix `Tool` trait.
pub struct McpToolAdapter {
    manager: McpManager,
    server: String,
    definition: McpToolDefinition,
    metadata: ToolMetadata,
}

impl McpToolAdapter {
    /// Build an adapter for `definition` published by `server`. The
    /// resulting tool name is `mcp:{server}:{tool}` so two servers
    /// exposing identically-named tools can coexist in the same
    /// [`entelix_core::ToolRegistry`].
    ///
    /// ```ignore
    /// let tools = manager.list_tools(&ctx, "filesystem").await?;
    /// let adapters = tools
    ///     .into_iter()
    ///     .map(|d| McpToolAdapter::new(manager.clone(), "filesystem", d))
    ///     .collect::<Vec<_>>();
    /// // adapters[i].metadata().name == "mcp:filesystem:<tool>"
    /// ```
    pub fn new(
        manager: McpManager,
        server: impl Into<String>,
        definition: McpToolDefinition,
    ) -> Self {
        let server = server.into();
        let metadata = ToolMetadata::function(
            qualified_name(&server, &definition.name),
            definition.description.clone(),
            definition.input_schema.clone(),
        );
        Self {
            manager,
            server,
            definition,
            metadata,
        }
    }

    /// Opt out of namespacing — expose the adapter under the bare
    /// MCP tool name instead of `mcp:{server}:{tool}`. Use only when
    /// the deployment has exactly one MCP server and the model is
    /// already prompt-trained on the unqualified name. With multiple
    /// servers this re-introduces the collision risk the default
    /// guards against.
    #[must_use]
    pub fn with_unqualified_name(mut self) -> Self {
        self.metadata.name = self.definition.name.clone();
        self
    }

    /// Borrow the underlying MCP definition (diagnostics / tests).
    pub const fn definition(&self) -> &McpToolDefinition {
        &self.definition
    }

    /// Borrow the upstream MCP server name.
    pub fn server(&self) -> &str {
        &self.server
    }

    /// Borrow the unqualified MCP tool name (the value the upstream
    /// server published). Distinct from `metadata().name` which
    /// returns the namespace-qualified name unless
    /// [`Self::with_unqualified_name`] was called.
    pub fn mcp_tool_name(&self) -> &str {
        &self.definition.name
    }
}

/// Build the canonical `mcp:{server}:{tool}` namespaced name.
///
/// Public so operators that pre-build a `ToolRegistry` and want to
/// look up adapters by name can compute the same key without owning
/// an adapter instance.
#[must_use]
pub fn qualified_name(server: &str, tool: &str) -> String {
    format!("{MCP_NAME_PREFIX}:{server}:{tool}")
}

#[async_trait]
impl Tool for McpToolAdapter {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    async fn execute(&self, input: Value, ctx: &AgentContext<()>) -> Result<Value> {
        // Always dispatch using the unqualified MCP name — the
        // `mcp:` prefix is an entelix-side namespacing decision and
        // would be rejected by the upstream server.
        let result = self
            .manager
            .call_tool(ctx.core(), &self.server, &self.definition.name, input)
            .await?;
        Ok(result)
    }
}
