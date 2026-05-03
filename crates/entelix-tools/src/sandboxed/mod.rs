//! Sandbox-backed `Tool` adapters per ADR-0024 §3.
//!
//! All tool execution that needs filesystem or shell access flows
//! through [`entelix_core::sandbox::Sandbox`] — none of this module
//! imports `std::fs` or `std::process` (Invariant 9 is enforced by
//! `scripts/check-no-fs.sh`).
//!
//! Tools shipped here:
//!
//! - [`SandboxedShellTool`] — execute commands with an allowlist
//!   policy.
//! - [`SandboxedCodeTool`] — execute source code in a chosen
//!   language.
//! - [`SandboxedReadFileTool`] / [`SandboxedWriteFileTool`] /
//!   [`SandboxedListDirTool`] — sandbox-internal filesystem ops.
//!
//! All of them hold an `Arc<dyn Sandbox>` and forward execution.
//! Concrete sandbox backends (E2B, Modal, Fly, Lambda, K8s Job)
//! ship as 1.x companion crates.

mod code;
mod fs;
mod policy;
mod shell;

pub mod mock;

pub use code::{CodePolicy, SandboxedCodeTool};
pub use fs::{SandboxedListDirTool, SandboxedReadFileTool, SandboxedWriteFileTool};
pub use mock::MockSandbox;
pub use policy::{ShellPolicy, ShellPolicyError};
pub use shell::SandboxedShellTool;
