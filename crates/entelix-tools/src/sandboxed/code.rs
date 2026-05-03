//! `SandboxedCodeTool` — execute source code in a chosen language
//! through a [`Sandbox`] backend.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use entelix_core::context::ExecutionContext;
use entelix_core::error::Result;
use entelix_core::sandbox::{CodeSpec, Sandbox, SandboxLanguage};
use entelix_core::tools::{Tool, ToolEffect, ToolMetadata};

use crate::error::ToolError;

/// Allowlist + timeout cap for [`SandboxedCodeTool`].
///
/// Restricts which [`SandboxLanguage`] values the agent may
/// request. Default permits Python only — the most common safe
/// choice for analytic agents. Operators that need shell-as-code
/// add `Bash` explicitly.
#[derive(Clone, Debug)]
pub struct CodePolicy {
    allowed_languages: HashSet<SandboxLanguage>,
    max_duration: Duration,
}

impl CodePolicy {
    /// Build with an explicit language allowlist.
    #[must_use]
    pub fn new<I>(allowed: I) -> Self
    where
        I: IntoIterator<Item = SandboxLanguage>,
    {
        Self {
            allowed_languages: allowed.into_iter().collect(),
            max_duration: Duration::from_secs(30),
        }
    }

    /// Override the duration cap.
    #[must_use]
    pub const fn with_max_duration(mut self, duration: Duration) -> Self {
        self.max_duration = duration;
        self
    }

    /// Whether the supplied language is on the allowlist.
    #[must_use]
    pub fn admits(&self, language: SandboxLanguage) -> bool {
        self.allowed_languages.contains(&language)
    }

    /// Borrow the duration cap.
    #[must_use]
    pub const fn max_duration(&self) -> Duration {
        self.max_duration
    }
}

impl Default for CodePolicy {
    /// Python-only default.
    fn default() -> Self {
        Self::new([SandboxLanguage::Python])
    }
}

/// Code-execution tool that delegates to a [`Sandbox`] after
/// language-allowlist enforcement.
pub struct SandboxedCodeTool {
    sandbox: Arc<dyn Sandbox>,
    policy: CodePolicy,
    metadata: ToolMetadata,
}

impl SandboxedCodeTool {
    /// Build with the given sandbox and policy.
    #[must_use]
    pub fn new(sandbox: Arc<dyn Sandbox>, policy: CodePolicy) -> Self {
        Self {
            sandbox,
            policy,
            metadata: code_tool_metadata(),
        }
    }
}

fn code_tool_metadata() -> ToolMetadata {
    ToolMetadata::function(
        "code",
        "Execute source code in an isolated sandbox. Language is allowlisted by an \
         operator-set CodePolicy. Output shape: success → {\"stdout\": \"...\"}; failure → \
         {\"exit_code\": N, \"stdout\": \"...\", \"stderr\": \"...\"}.",
        json!({
            "type": "object",
            "required": ["language", "source"],
            "properties": {
                "language": {
                    "type": "string",
                    "enum": ["bash", "python", "type_script", "java_script"],
                    "description": "Source language."
                },
                "source": {
                    "type": "string",
                    "description": "Source code to execute."
                }
            },
            "additionalProperties": false
        }),
    )
    .with_effect(ToolEffect::Destructive)
}

impl std::fmt::Debug for SandboxedCodeTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SandboxedCodeTool")
            .field("backend", &self.sandbox.backend())
            .field("max_duration", &self.policy.max_duration())
            .field("metadata", &self.metadata.name)
            .finish()
    }
}

#[derive(Debug, Deserialize)]
struct CodeToolInput {
    language: SandboxLanguage,
    source: String,
}

#[async_trait]
impl Tool for SandboxedCodeTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    async fn execute(&self, input: Value, ctx: &ExecutionContext) -> Result<Value> {
        let parsed: CodeToolInput = serde_json::from_value(input).map_err(ToolError::from)?;
        if !self.policy.admits(parsed.language) {
            return Err(ToolError::config_msg(format!(
                "language {:?} is not on the CodePolicy allowlist",
                parsed.language
            ))
            .into());
        }
        let spec = CodeSpec {
            language: parsed.language,
            source: parsed.source,
            timeout: Some(self.policy.max_duration()),
        };
        let output = self.sandbox.run_code(spec, ctx).await?;
        // Lean LLM-facing payload — see SandboxedShellTool for the
        // success/failure split rationale (ADR-0024 §7).
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
