//! `ToolSpec` and `ToolChoice` — the tool surface advertised to the model.
//!
//! These are IR-level descriptions. The dispatch-side `Tool` trait lives
//! in [`crate::tools`].

use serde::{Deserialize, Serialize};

use crate::ir::cache::CacheControl;

/// One tool advertised to the model in a `ModelRequest`.
///
/// The `kind` discriminates between operator-supplied function tools
/// and vendor built-ins (web search, computer use). The struct stays
/// flat for `name` / `description` because every `kind` advertises
/// both — even built-ins surface a description so recipes can reason
/// about them uniformly.
///
/// `cache_control` marks the tool declaration itself as
/// cacheable on vendors that support per-block caching of tool
/// definitions (Anthropic, Bedrock-on-Anthropic). Stable tool
/// catalogues amortise their declaration tokens across calls.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    /// Tool name. Must be unique within a request.
    pub name: String,
    /// Human-readable description shown to the model.
    pub description: String,
    /// What kind of tool this is — function, web search, computer use, etc.
    pub kind: ToolKind,
    /// Cache directive for the tool declaration itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl ToolSpec {
    /// Convenience constructor for a function tool with the given JSON Schema.
    #[must_use]
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            kind: ToolKind::Function { input_schema },
            cache_control: None,
        }
    }

    /// Attach a cache directive to this tool declaration. Codecs
    /// that support per-block tool caching honour it; others emit
    /// `LossyEncode`.
    #[must_use]
    pub fn with_cache_control(mut self, cache: CacheControl) -> Self {
        self.cache_control = Some(cache);
        self
    }
}

/// Discriminator for `ToolSpec` — function vs vendor built-in.
///
/// `#[non_exhaustive]` so additional vendor built-ins land cleanly.
/// Codecs that lack a native equivalent for a given variant emit
/// [`crate::ir::ModelWarning::LossyEncode`] (invariant 6) so the
/// operator sees that the tool was advertised but the vendor cannot
/// honour it.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolKind {
    /// Operator-supplied callable. The vendor invokes it via
    /// [`ContentPart::ToolUse`](crate::ir::ContentPart::ToolUse) and the
    /// harness dispatches into [`crate::tools::Tool`].
    Function {
        /// JSON Schema for the tool's input.
        input_schema: serde_json::Value,
    },
    /// Vendor built-in web search.
    ///
    /// Anthropic `web_search_*`, OpenAI Responses `web_search`, Gemini
    /// `google_search`.
    WebSearch {
        /// Maximum number of search invocations the model may make per
        /// turn. `None` = unlimited / vendor default.
        max_uses: Option<u32>,
        /// Allowlist for domain restrictions. Empty = unrestricted.
        allowed_domains: Vec<String>,
    },
    /// Vendor built-in computer-use tool (screenshot + actions).
    ///
    /// Anthropic `computer_*`.
    Computer {
        /// Display width in pixels.
        display_width: u32,
        /// Display height in pixels.
        display_height: u32,
    },
    /// Anthropic `text_editor_*` — string-editing primitives the
    /// model uses to view / create / edit / undo files inside the
    /// tool sandbox the operator supplies.
    TextEditor,
    /// Anthropic `bash_*` — shell-execution primitive paired with a
    /// sandbox the operator supplies. The harness still routes the
    /// dispatch through its `Sandbox` impl (invariant 9).
    Bash,
    /// Anthropic `code_execution_*` — server-side Python sandbox
    /// (vendor-managed, no operator wiring).
    CodeExecution,
    /// OpenAI Responses `file_search` — vector-store-backed retrieval
    /// against operator-uploaded files.
    FileSearch {
        /// Vector-store identifiers the model may search. Required by
        /// the wire shape; an empty list is rejected at encode time.
        vector_store_ids: Vec<String>,
    },
    /// OpenAI Responses `code_interpreter` — vendor-managed Python
    /// sandbox.
    CodeInterpreter,
    /// OpenAI Responses `image_generation` — vendor-managed image
    /// synthesis tool. The model emits an
    /// `image_generation_call` output item the codec surfaces as
    /// [`ContentPart::ImageOutput`](crate::ir::ContentPart::ImageOutput).
    ImageGeneration,
    /// Anthropic `mcp` connector — the vendor (not the operator)
    /// brokers an MCP server connection on the request's behalf.
    /// The harness's own MCP client (entelix-mcp) is unaffected;
    /// this surface is for vendor-side hosting.
    McpConnector {
        /// Connector descriptor name (vendor-side identifier).
        name: String,
        /// Vendor-reachable MCP server URL.
        server_url: String,
        /// Optional bearer token the vendor presents to the MCP
        /// server. Operator-provided; the codec passes it on the
        /// wire verbatim. Anthropic stores this server-side per
        /// connection.
        authorization_token: Option<String>,
    },
    /// Anthropic `memory_*` — vendor-managed persistent memory.
    /// Distinct from the `entelix-memory` long-term memory crate
    /// (which the operator owns); this surface lets Anthropic store
    /// model-side memory across requests.
    Memory,
}

/// Constraints on which tool the model is allowed to call next.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ToolChoice {
    /// Model picks freely (or no tool at all).
    #[default]
    Auto,
    /// Model must call exactly one tool, but may pick which.
    Required,
    /// Model must call this specific tool.
    Specific {
        /// Required tool name.
        name: String,
    },
    /// Tools are advertised but the model is forbidden from calling any.
    None,
}
