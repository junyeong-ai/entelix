//! `Sandbox` — sandbox-agnostic isolated execution environment.
//!
//! Per (invariant 9 reinterpretation): first-party
//! crates touch zero `std::fs` / `std::process`; **all** filesystem
//! and shell intent flows through this trait. The trait is
//! `entelix-core` 1st-class so [`crate::tools::Tool`] adapters can
//! depend on it without re-importing across crate boundaries.
//!
//! Concrete implementations ship as 1.x companion crates
//! (`entelix-sandbox-e2b`, `entelix-sandbox-modal`,
//! `entelix-sandbox-fly`, `entelix-sandbox-lambda`,
//! `entelix-sandbox-k8s-job`). For tests, `MockSandbox` lives in
//! `entelix-tools::sandboxed`.
//!
//! ## Threat model
//!
//! Every `run_command`, `run_code`, `read_file`, `write_file`,
//! `list_dir` call is **untrusted host-process-isolated**:
//!
//! - Process isolation enforced by the backend (E2B microVM,
//!   Modal container, Fly.io firecracker, Lambda firecracker, …).
//! - No host filesystem reach — every fs / shell op routes through
//!   the sandbox backend (invariant 9).
//! - Network egress, CPU, memory caps are the backend's concern;
//!   `Sandbox` exposes only the orchestration surface.
//!
//! Tool-level allowlists (e.g. `ShellPolicy` in
//! `entelix-tools::sandboxed`) are a **second** defense layer on
//! top of sandbox isolation — defense in depth, not redundancy.

use std::collections::BTreeMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::context::ExecutionContext;
use crate::error::Result;

/// Sandbox-agnostic isolated execution environment.
///
/// Implementations should be cheap to clone — the canonical handle
/// is `Arc<dyn Sandbox>` so multiple tools can share one connection
/// pool / authentication context. Methods are async because most
/// sandbox backends speak HTTP / gRPC; in-process adapters are
/// possible but rare.
///
/// All methods take an [`ExecutionContext`] so cancellation,
/// deadlines, and tenant scope propagate uniformly into the
/// backend. Backends that ignore `ctx` violate Invariant 3.
#[async_trait]
pub trait Sandbox: Send + Sync {
    /// Backend identifier surfaced in OTel attributes
    /// (`entelix.sandbox.backend`). Examples: `"e2b"`, `"modal"`,
    /// `"fly-machines"`, `"k8s-job"`. Stable across releases.
    fn backend(&self) -> &str;

    /// Execute a shell command in the isolated environment. The
    /// caller (typically a [`crate::tools::Tool`] adapter)
    /// validates `spec.argv` against an allowlist before reaching
    /// this method — `Sandbox` enforces process isolation, not
    /// argument shape.
    async fn run_command(&self, spec: CommandSpec, ctx: &ExecutionContext)
    -> Result<CommandOutput>;

    /// Execute source code in the isolated environment. Language
    /// support is backend-dependent; the trait surface enumerates
    /// the canonical set via [`SandboxLanguage`].
    async fn run_code(&self, spec: CodeSpec, ctx: &ExecutionContext) -> Result<CommandOutput>;

    /// Read a file from the sandbox-internal filesystem.
    async fn read_file(&self, path: &str, ctx: &ExecutionContext) -> Result<Vec<u8>>;

    /// Write a file to the sandbox-internal filesystem. The path
    /// is interpreted by the backend; entelix imposes no
    /// host-style normalization.
    async fn write_file(&self, path: &str, bytes: &[u8], ctx: &ExecutionContext) -> Result<()>;

    /// List a sandbox-internal directory.
    async fn list_dir(&self, path: &str, ctx: &ExecutionContext) -> Result<Vec<DirEntry>>;
}

/// Command-execution spec — what to run and how.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandSpec {
    /// Argv vector. `argv[0]` is the program name; the backend
    /// resolves it on its own `PATH`.
    pub argv: Vec<String>,
    /// Working directory inside the sandbox (`None` = backend
    /// default — typically `/workspace` or `/tmp`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub working_dir: Option<String>,
    /// Environment variables passed to the spawned process.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
    /// Optional stdin payload. Backends that do not support stdin
    /// emit a `LossyEncode` warning at run time (per Invariant 6).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stdin: Option<Vec<u8>>,
    /// Per-call timeout (caller-imposed, on top of any backend
    /// limit). `None` defers to the backend default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout: Option<std::time::Duration>,
}

impl CommandSpec {
    /// Convenience: a command spec from an `argv` vector with all
    /// other fields defaulted.
    #[must_use]
    pub fn new(argv: Vec<String>) -> Self {
        Self {
            argv,
            ..Self::default()
        }
    }
}

/// Code-execution spec.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodeSpec {
    /// Language the source is interpreted as.
    pub language: SandboxLanguage,
    /// Source text. The backend may impose size / line caps.
    pub source: String,
    /// Per-call timeout. `None` defers to the backend default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout: Option<std::time::Duration>,
}

impl CodeSpec {
    /// Convenience: a code spec with the canonical fields and
    /// `timeout: None`.
    #[must_use]
    pub fn new(language: SandboxLanguage, source: impl Into<String>) -> Self {
        Self {
            language,
            source: source.into(),
            timeout: None,
        }
    }
}

/// Language identifier the backend uses to pick an interpreter.
///
/// `#[non_exhaustive]` so new languages can be added without
/// breaking match arms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SandboxLanguage {
    /// Bash / sh shell scripting.
    Bash,
    /// Python (3.x — backend chooses minor version).
    Python,
    /// `TypeScript` (typically transpiled with `tsx` / `ts-node`).
    TypeScript,
    /// Plain `JavaScript` (Node.js).
    JavaScript,
}

/// Result of a `run_command` or `run_code` call.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandOutput {
    /// POSIX-style exit code. `0` = success.
    pub exit_code: i32,
    /// Captured stdout — raw bytes (callers decide on UTF-8
    /// interpretation; binary outputs survive the trait).
    pub stdout: Vec<u8>,
    /// Captured stderr — raw bytes.
    pub stderr: Vec<u8>,
    /// Wall-clock execution duration measured by the backend.
    pub duration_ms: u64,
}

impl CommandOutput {
    /// Whether the command exited with a zero status.
    #[must_use]
    pub const fn succeeded(&self) -> bool {
        self.exit_code == 0
    }

    /// Decode stdout as UTF-8 with `\u{FFFD}` substitution for
    /// invalid sequences. Convenience for `Tool` adapters that
    /// only forward textual output.
    #[must_use]
    pub fn stdout_lossy(&self) -> String {
        String::from_utf8_lossy(&self.stdout).into_owned()
    }

    /// Decode stderr as UTF-8 with `\u{FFFD}` substitution for
    /// invalid sequences.
    #[must_use]
    pub fn stderr_lossy(&self) -> String {
        String::from_utf8_lossy(&self.stderr).into_owned()
    }
}

/// One entry returned by [`Sandbox::list_dir`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirEntry {
    /// Entry name (basename, not full path).
    pub name: String,
    /// `true` for directories, `false` for files. Symlinks are
    /// reported per the backend's resolution policy — most
    /// backends follow.
    pub is_dir: bool,
    /// Size in bytes for files (`None` for directories or when
    /// the backend cannot cheaply stat).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn command_output_helpers_decode_utf8_lossily() {
        let output = CommandOutput {
            exit_code: 0,
            stdout: b"hello".to_vec(),
            stderr: vec![0xff, 0x66, 0x6f],
            duration_ms: 12,
        };
        assert!(output.succeeded());
        assert_eq!(output.stdout_lossy(), "hello");
        // Invalid UTF-8 byte (0xff) → replacement char.
        assert!(output.stderr_lossy().contains('\u{FFFD}'));
    }

    #[test]
    fn command_spec_default_has_empty_argv() {
        let spec = CommandSpec::default();
        assert!(spec.argv.is_empty());
        assert!(spec.env.is_empty());
        assert!(spec.timeout.is_none());
    }

    #[test]
    fn sandbox_language_round_trips_via_serde() {
        let langs = [
            SandboxLanguage::Bash,
            SandboxLanguage::Python,
            SandboxLanguage::TypeScript,
            SandboxLanguage::JavaScript,
        ];
        for lang in langs {
            let json = serde_json::to_string(&lang).expect("serialize");
            let back: SandboxLanguage = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(lang, back);
        }
    }

    #[test]
    fn command_spec_new_seeds_argv_and_defaults() {
        let spec = CommandSpec::new(vec!["ls".into(), "-la".into()]);
        assert_eq!(spec.argv, vec!["ls".to_owned(), "-la".to_owned()]);
        assert!(spec.working_dir.is_none());
    }
}
