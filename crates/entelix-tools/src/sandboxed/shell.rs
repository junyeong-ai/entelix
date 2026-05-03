//! `SandboxedShellTool` — `Tool` impl that runs commands through
//! a [`Sandbox`] backend with a [`ShellPolicy`] allowlist.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use entelix_core::context::ExecutionContext;
use entelix_core::error::Result;
use entelix_core::sandbox::{CommandSpec, Sandbox};
use entelix_core::tools::{Tool, ToolEffect, ToolMetadata};

use crate::error::ToolError;
use crate::sandboxed::policy::ShellPolicy;

/// Shell tool that delegates execution to a [`Sandbox`] backend
/// after a [`ShellPolicy`] allowlist check.
///
/// The tool name surfaced to the model is `"shell"`. Operators that
/// need a different identifier wrap or rename via
/// [`ToolRegistry::register`](entelix_core::ToolRegistry::register)
/// after construction.
pub struct SandboxedShellTool {
    sandbox: Arc<dyn Sandbox>,
    policy: ShellPolicy,
    metadata: ToolMetadata,
}

impl SandboxedShellTool {
    /// Build with the given sandbox and policy.
    #[must_use]
    pub fn new(sandbox: Arc<dyn Sandbox>, policy: ShellPolicy) -> Self {
        Self {
            sandbox,
            policy,
            metadata: shell_tool_metadata(),
        }
    }

    /// Convenience constructor with the read-only baseline policy.
    #[must_use]
    pub fn read_only(sandbox: Arc<dyn Sandbox>) -> Self {
        Self::new(sandbox, ShellPolicy::read_only_baseline())
    }
}

fn shell_tool_metadata() -> ToolMetadata {
    ToolMetadata::function(
        "shell",
        "Execute a shell command in an isolated sandbox. Argv is allowlisted by an \
         operator-set ShellPolicy. Output shape: success → {\"stdout\": \"...\"}; failure → \
         {\"exit_code\": N, \"stdout\": \"...\", \"stderr\": \"...\"}.",
        json!({
            "type": "object",
            "required": ["argv"],
            "properties": {
                "argv": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "Command argv. argv[0] is the program; allowlist applies to it."
                },
                "working_dir": {
                    "type": ["string", "null"],
                    "description": "Sandbox-internal working directory (optional)."
                }
            },
            "additionalProperties": false
        }),
    )
    .with_effect(ToolEffect::Destructive)
}

impl std::fmt::Debug for SandboxedShellTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SandboxedShellTool")
            .field("backend", &self.sandbox.backend())
            .field("allowed_commands", &self.policy.allowed().len())
            .field("max_duration", &self.policy.max_duration())
            .field("metadata", &self.metadata.name)
            .finish()
    }
}

#[derive(Debug, Deserialize)]
struct ShellToolInput {
    /// Argv vector — `["ls", "-la"]` style.
    argv: Vec<String>,
    /// Optional working directory inside the sandbox.
    #[serde(default)]
    working_dir: Option<String>,
}

#[async_trait]
impl Tool for SandboxedShellTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    async fn execute(&self, input: Value, ctx: &ExecutionContext) -> Result<Value> {
        let parsed: ShellToolInput = serde_json::from_value(input).map_err(ToolError::from)?;

        // Policy check — first defense layer.
        self.policy.check(&parsed.argv).map_err(ToolError::config)?;

        // Sandbox dispatch — second (and stronger) defense layer.
        let spec = CommandSpec {
            argv: parsed.argv,
            working_dir: parsed.working_dir,
            env: std::collections::BTreeMap::new(),
            stdin: None,
            timeout: Some(self.policy.max_duration()),
        };
        let output = self.sandbox.run_command(spec, ctx).await?;

        // Lean LLM-facing payload (ADR-0024 §7):
        // - Success: stdout text only — the model reasons over output.
        // - Failure: exit_code + stderr + stdout — the model needs the
        //   error class to recover.
        // Observability metadata (`duration_ms`, raw stderr bytes,
        // backend identifier) is surfaced to the `OtelLayer` /
        // `AgentEventSink` separately, never via the tool return.
        if output.succeeded() {
            Ok(json!({"stdout": output.stdout_lossy()}))
        } else {
            Ok(json!({
                "exit_code": output.exit_code,
                "stdout": output.stdout_lossy(),
                "stderr": output.stderr_lossy(),
            }))
        }
    }
}
