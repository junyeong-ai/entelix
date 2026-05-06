//! `MockSandbox` — in-memory `Sandbox` implementation for tests.
//!
//! Offers programmable canned responses keyed by argv prefix, an
//! in-memory filesystem (path → bytes), and call counters for
//! assertions. Production sandboxes (E2B, Modal, Fly, …) ship in
//! 1.x companion crates; this mock exists to drive
//! `SandboxedShellTool` / `SandboxedReadFileTool` integration tests
//! deterministically.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;

use entelix_core::context::ExecutionContext;
use entelix_core::error::Result;
use entelix_core::sandbox::{CodeSpec, CommandOutput, CommandSpec, DirEntry, Sandbox};

/// Programmable in-memory sandbox.
///
/// Construct via [`Self::new`] and seed canned responses with
/// [`Self::add_command_response`]. Files written via `write_file` round-trip
/// through `read_file`. Public so 1.x companion crates
/// (`entelix-sandbox-e2b`, `entelix-sandbox-modal`, …) can use it
/// as a regression baseline when validating their own [`Sandbox`]
/// implementations.
pub struct MockSandbox {
    inner: Arc<Mutex<MockState>>,
}

struct MockState {
    /// argv[0] → canned output. Lookup is exact on `argv[0]`.
    canned: HashMap<String, CommandOutput>,
    /// Default response when no canned entry matches.
    default: CommandOutput,
    /// In-memory filesystem.
    fs: HashMap<String, Vec<u8>>,
    /// Total commands seen.
    command_calls: u32,
    /// Total code-runs seen.
    code_calls: u32,
}

impl MockSandbox {
    /// Empty mock — no canned responses, empty filesystem.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(MockState {
                canned: HashMap::new(),
                default: CommandOutput {
                    exit_code: 0,
                    stdout: b"<mock default>\n".to_vec(),
                    stderr: Vec::new(),
                    duration_ms: 1,
                },
                fs: HashMap::new(),
                command_calls: 0,
                code_calls: 0,
            })),
        }
    }

    /// Register a canned response keyed by `argv[0]`. Subsequent
    /// calls to `Sandbox::run_command` whose `argv[0]` matches
    /// return the stored output; unmatched calls return the
    /// default mock output.
    #[must_use]
    pub fn add_command_response(self, argv0: &str, output: CommandOutput) -> Self {
        self.inner.lock().canned.insert(argv0.to_owned(), output);
        self
    }

    /// Total `Sandbox::run_command` calls observed since
    /// construction.
    #[must_use]
    pub fn command_calls(&self) -> u32 {
        self.inner.lock().command_calls
    }

    /// Total `Sandbox::run_code` calls observed since
    /// construction.
    #[must_use]
    pub fn code_calls(&self) -> u32 {
        self.inner.lock().code_calls
    }

    /// Synchronous helper for test setup — seed a file at the given
    /// sandbox-internal path. Use [`Sandbox::write_file`] from inside
    /// an async context; reach for this when assembling fixtures
    /// outside `async`.
    pub fn seed_file(&self, path: impl Into<String>, bytes: impl Into<Vec<u8>>) {
        self.inner.lock().fs.insert(path.into(), bytes.into());
    }
}

impl Default for MockSandbox {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Sandbox for MockSandbox {
    fn backend(&self) -> &str {
        "mock"
    }

    async fn run_command(
        &self,
        spec: CommandSpec,
        _ctx: &ExecutionContext,
    ) -> Result<CommandOutput> {
        let mut state = self.inner.lock();
        state.command_calls += 1;
        let head = spec.argv.first().cloned().unwrap_or_default();
        Ok(state
            .canned
            .get(&head)
            .cloned()
            .unwrap_or_else(|| state.default.clone()))
    }

    async fn run_code(&self, _spec: CodeSpec, _ctx: &ExecutionContext) -> Result<CommandOutput> {
        let mut state = self.inner.lock();
        state.code_calls += 1;
        Ok(state.default.clone())
    }

    async fn read_file(&self, path: &str, _ctx: &ExecutionContext) -> Result<Vec<u8>> {
        let state = self.inner.lock();
        Ok(state.fs.get(path).cloned().unwrap_or_default())
    }

    async fn write_file(&self, path: &str, bytes: &[u8], _ctx: &ExecutionContext) -> Result<()> {
        self.inner.lock().fs.insert(path.to_owned(), bytes.to_vec());
        Ok(())
    }

    async fn list_dir(&self, path: &str, _ctx: &ExecutionContext) -> Result<Vec<DirEntry>> {
        let state = self.inner.lock();
        // Normalise the prefix so `<path>` and `<path>/` both work.
        let prefix = if path.is_empty() || path == "/" {
            String::new()
        } else {
            let trimmed = path.trim_end_matches('/');
            format!("{trimmed}/")
        };

        // Walk every stored absolute path. A path that begins with
        // the prefix and has a single remaining segment is a child
        // *file*; a path with more segments contributes its first
        // remaining segment as a child *directory*.
        let mut files: Vec<(String, u64)> = Vec::new();
        let mut dirs: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for (full, bytes) in &state.fs {
            let rest = if prefix.is_empty() {
                full.trim_start_matches('/')
            } else if let Some(rest) = full.strip_prefix(&prefix) {
                rest
            } else {
                continue;
            };
            match rest.split_once('/') {
                None => files.push((rest.to_owned(), bytes.len() as u64)),
                Some((head, _)) if !head.is_empty() => {
                    dirs.insert(head.to_owned());
                }
                Some(_) => {}
            }
        }
        let mut out: Vec<DirEntry> = Vec::with_capacity(dirs.len() + files.len());
        for name in dirs {
            out.push(DirEntry {
                name,
                is_dir: true,
                size_bytes: None,
            });
        }
        for (name, size) in files {
            out.push(DirEntry {
                name,
                is_dir: false,
                size_bytes: Some(size),
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use serde_json::json;

    use entelix_core::AgentContext;
    use entelix_core::tools::Tool;

    use super::*;
    use crate::sandboxed::{
        CodePolicy, SandboxedCodeTool, SandboxedListDirTool, SandboxedReadFileTool,
        SandboxedShellTool, SandboxedWriteFileTool, ShellPolicy,
    };

    #[tokio::test]
    async fn shell_tool_round_trips_command_output() {
        let sandbox = Arc::new(MockSandbox::new().add_command_response(
            "ls",
            CommandOutput {
                exit_code: 0,
                stdout: b"file_a\nfile_b\n".to_vec(),
                stderr: Vec::new(),
                duration_ms: 5,
            },
        ));
        let tool = SandboxedShellTool::read_only(sandbox.clone());
        let out = tool
            .execute(json!({"argv": ["ls", "-la"]}), &AgentContext::default())
            .await
            .unwrap();
        // Lean success payload: stdout only, no exit_code or
        // duration metadata (ADR-0024 §7).
        assert!(out["stdout"].as_str().unwrap().contains("file_a"));
        assert!(out.get("duration_ms").is_none());
        assert!(out.get("exit_code").is_none());
        assert_eq!(sandbox.command_calls(), 1);
    }

    #[tokio::test]
    async fn shell_tool_blocks_command_outside_allowlist() {
        let sandbox: Arc<dyn Sandbox> = Arc::new(MockSandbox::new());
        let tool = SandboxedShellTool::new(sandbox, ShellPolicy::read_only_baseline());
        let err = tool
            .execute(
                json!({"argv": ["rm", "-rf", "/"]}),
                &AgentContext::default(),
            )
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("not on the allowlist"));
    }

    #[tokio::test]
    async fn code_tool_blocks_language_outside_policy() {
        let sandbox: Arc<dyn Sandbox> = Arc::new(MockSandbox::new());
        // Default policy permits Python only — Bash is rejected.
        let tool = SandboxedCodeTool::new(sandbox, CodePolicy::default());
        let err = tool
            .execute(
                json!({"language": "bash", "source": "echo hi"}),
                &AgentContext::default(),
            )
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("not on the CodePolicy allowlist"));
    }

    #[tokio::test]
    async fn code_tool_dispatches_admitted_language() {
        let sandbox = Arc::new(MockSandbox::new());
        let tool = SandboxedCodeTool::new(sandbox.clone(), CodePolicy::default());
        let out = tool
            .execute(
                json!({"language": "python", "source": "print('hi')"}),
                &AgentContext::default(),
            )
            .await
            .unwrap();
        // Lean success payload: stdout only.
        assert!(out["stdout"].as_str().is_some());
        assert!(out.get("exit_code").is_none());
        assert_eq!(sandbox.code_calls(), 1);
    }

    #[tokio::test]
    async fn fs_round_trip_via_write_then_read() {
        let sandbox: Arc<dyn Sandbox> = Arc::new(MockSandbox::new());
        let writer = SandboxedWriteFileTool::new(Arc::clone(&sandbox));
        let reader = SandboxedReadFileTool::new(Arc::clone(&sandbox));
        let lister = SandboxedListDirTool::new(Arc::clone(&sandbox));

        let _ = writer
            .execute(
                json!({"path": "/tmp/note.txt", "content": "hello"}),
                &AgentContext::default(),
            )
            .await
            .unwrap();
        let read = reader
            .execute(json!({"path": "/tmp/note.txt"}), &AgentContext::default())
            .await
            .unwrap();
        assert_eq!(read["content"], "hello");

        let listing = lister
            .execute(json!({"path": "/tmp"}), &AgentContext::default())
            .await
            .unwrap();
        let entries = listing["entries"].as_array().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["name"], "note.txt");
        // Lean payload: no echo of `path`.
        assert!(listing.get("path").is_none());
    }
}
