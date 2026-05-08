//! Sandbox-internal filesystem tools — `read_file`, `write_file`,
//! `list_dir`. Each is a thin `Tool` adapter over the matching
//! [`Sandbox`] method.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use entelix_core::AgentContext;
use entelix_core::error::Result;
use entelix_core::sandbox::Sandbox;
use entelix_core::tools::{Tool, ToolEffect, ToolMetadata};

use crate::error::ToolError;

/// Read a file from the sandbox-internal filesystem.
pub struct SandboxedReadFileTool {
    sandbox: Arc<dyn Sandbox>,
    metadata: ToolMetadata,
}

impl SandboxedReadFileTool {
    /// Build with the given sandbox handle.
    #[must_use]
    pub fn new(sandbox: Arc<dyn Sandbox>) -> Self {
        Self {
            sandbox,
            metadata: read_file_metadata(),
        }
    }
}

fn read_file_metadata() -> ToolMetadata {
    ToolMetadata::function(
        "read_file",
        "Read a file from the sandbox-internal filesystem. The path is interpreted by the \
         sandbox backend.",
        json!({
            "type": "object",
            "required": ["path"],
            "properties": {"path": {"type": "string"}},
            "additionalProperties": false
        }),
    )
    .with_effect(ToolEffect::ReadOnly)
    .with_idempotent(true)
}

impl std::fmt::Debug for SandboxedReadFileTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SandboxedReadFileTool")
            .field("backend", &self.sandbox.backend())
            .field("metadata", &self.metadata.name)
            .finish()
    }
}

#[derive(Debug, Deserialize)]
struct PathInput {
    path: String,
}

#[async_trait]
impl Tool for SandboxedReadFileTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    async fn execute(&self, input: Value, ctx: &AgentContext<()>) -> Result<Value> {
        let parsed: PathInput = serde_json::from_value(input).map_err(ToolError::from)?;
        let bytes = self.sandbox.read_file(&parsed.path, ctx.core()).await?;
        // Lean LLM-facing payload: text on UTF-8, base64 on binary.
        // Operators that always know binary handle it explicitly.
        let content = String::from_utf8(bytes.clone()).unwrap_or_else(|e| {
            // Non-UTF-8: the model still benefits from a textual
            // summary (size + first-prefix) so it can reason about
            // the file shape without seeing raw bytes.
            let prefix_len = bytes.len().min(256);
            let prefix = bytes.get(..prefix_len).unwrap_or(&[]);
            format!(
                "<non-utf8 file ({} bytes); first valid prefix: {}>",
                e.as_bytes().len(),
                String::from_utf8_lossy(prefix),
            )
        });
        // Lean LLM-facing payload: `content` only.
        // `bytes_read` would be observability noise on the model's
        // next turn — sinks observe size via the tool's OTel span.
        Ok(json!({"content": content}))
    }
}

/// Write a file to the sandbox-internal filesystem.
pub struct SandboxedWriteFileTool {
    sandbox: Arc<dyn Sandbox>,
    metadata: ToolMetadata,
}

impl SandboxedWriteFileTool {
    /// Build with the given sandbox handle.
    #[must_use]
    pub fn new(sandbox: Arc<dyn Sandbox>) -> Self {
        Self {
            sandbox,
            metadata: write_file_metadata(),
        }
    }
}

fn write_file_metadata() -> ToolMetadata {
    ToolMetadata::function(
        "write_file",
        "Write a file to the sandbox-internal filesystem. UTF-8 content only — binary writes \
         use a sandbox-specific tool.",
        json!({
            "type": "object",
            "required": ["path", "content"],
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"}
            },
            "additionalProperties": false
        }),
    )
    .with_effect(ToolEffect::Mutating)
}

impl std::fmt::Debug for SandboxedWriteFileTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SandboxedWriteFileTool")
            .field("backend", &self.sandbox.backend())
            .field("metadata", &self.metadata.name)
            .finish()
    }
}

#[derive(Debug, Deserialize)]
struct WriteInput {
    path: String,
    content: String,
}

#[async_trait]
impl Tool for SandboxedWriteFileTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    async fn execute(&self, input: Value, ctx: &AgentContext<()>) -> Result<Value> {
        let parsed: WriteInput = serde_json::from_value(input).map_err(ToolError::from)?;
        let bytes = parsed.content.as_bytes();
        self.sandbox
            .write_file(&parsed.path, bytes, ctx.core())
            .await?;
        // Lean LLM-facing payload — bare confirmation. The model
        // already knows the path it wrote to and the content it
        // sent; echoing those is token waste.
        Ok(json!({"ok": true}))
    }
}

/// List a sandbox-internal directory.
pub struct SandboxedListDirTool {
    sandbox: Arc<dyn Sandbox>,
    metadata: ToolMetadata,
}

impl SandboxedListDirTool {
    /// Build with the given sandbox handle.
    #[must_use]
    pub fn new(sandbox: Arc<dyn Sandbox>) -> Self {
        Self {
            sandbox,
            metadata: list_dir_metadata(),
        }
    }
}

fn list_dir_metadata() -> ToolMetadata {
    ToolMetadata::function(
        "list_dir",
        "List entries in a sandbox-internal directory. Returns [{name, is_dir, size_bytes?}, \
         ...].",
        json!({
            "type": "object",
            "required": ["path"],
            "properties": {"path": {"type": "string"}},
            "additionalProperties": false
        }),
    )
    .with_effect(ToolEffect::ReadOnly)
    .with_idempotent(true)
}

impl std::fmt::Debug for SandboxedListDirTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SandboxedListDirTool")
            .field("backend", &self.sandbox.backend())
            .field("metadata", &self.metadata.name)
            .finish()
    }
}

#[async_trait]
impl Tool for SandboxedListDirTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    async fn execute(&self, input: Value, ctx: &AgentContext<()>) -> Result<Value> {
        let parsed: PathInput = serde_json::from_value(input).map_err(ToolError::from)?;
        let entries = self.sandbox.list_dir(&parsed.path, ctx.core()).await?;
        // Lean LLM-facing payload — just the entries; the model
        // already knows the path it queried.
        Ok(json!({"entries": entries}))
    }
}
