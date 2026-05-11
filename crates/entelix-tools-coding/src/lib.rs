//! # entelix-tools-coding
//!
//! Coding-agent vertical tools: shell, code execution, filesystem,
//! and the Anthropic Claude Skills `SKILL.md` layout. Every tool
//! delegates IO through [`entelix_core::sandbox::Sandbox`] — invariant
//! 9 preserved, no `std::fs` / `std::process` import lands here.
//!
//! This crate is a **companion** to [`entelix-tools`](https://docs.rs/entelix-tools).
//! The horizontal `entelix-tools` surface ships defensive primitives
//! (HTTP fetch with SSRF allowlist, calculator, search adapter,
//! memory tools, in-memory skills) that any agent benefits from.
//! The vertical `entelix-tools-coding` surface ships the
//! coding-agent territory — separated so a research / customer-support
//! / RAG / agentic agent does not pull shell-and-filesystem opinions
//! it never uses.
//!
//! ## Surface
//!
//! - [`SandboxedShellTool`] — execute commands with a [`ShellPolicy`]
//!   allowlist on top of `Sandbox`-routed dispatch.
//! - [`SandboxedCodeTool`] — execute source code in a chosen
//!   [`SandboxLanguage`](entelix_core::sandbox::SandboxLanguage)
//!   under a [`CodePolicy`] allowlist.
//! - [`SandboxedReadFileTool`] / [`SandboxedWriteFileTool`] /
//!   [`SandboxedListDirTool`] — sandbox-internal filesystem ops.
//! - [`SandboxSkill`] — `Skill` implementation backed by a
//!   sandbox-internal directory tree mirroring the Anthropic Claude
//!   Skills layout (`SKILL.md` + sibling files).
//! - [`MockSandbox`] — programmable in-memory `Sandbox` for tests
//!   and as a regression baseline for downstream sandbox companions
//!   (`entelix-sandbox-e2b`, `entelix-sandbox-modal`, …).
//!
//! ## Wiring
//!
//! ```ignore
//! use std::sync::Arc;
//! use entelix_core::ToolRegistry;
//! use entelix_tools_coding::{SandboxedShellTool, ShellPolicy, MockSandbox};
//!
//! let sandbox: Arc<dyn entelix_core::sandbox::Sandbox> = Arc::new(MockSandbox::new());
//! let policy = ShellPolicy::READ_ONLY_BASELINE.clone();
//! let registry = ToolRegistry::new()
//!     .register(Arc::new(SandboxedShellTool::new(sandbox, policy)))?;
//! # Ok::<(), entelix_core::Error>(())
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_root_url = "https://docs.rs/entelix-tools-coding/0.4.2")]
#![deny(missing_docs)]
#![allow(
    clippy::doc_markdown,
    clippy::elidable_lifetime_names,
    clippy::missing_const_for_fn,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::needless_lifetimes,
    clippy::needless_pass_by_value,
    clippy::option_if_let_else,
    clippy::significant_drop_tightening,
    clippy::unnecessary_literal_bound
)]

mod code;
mod fs;
mod manifest;
pub mod mock;
mod policy;
mod shell;
mod skill;

pub use code::{CodePolicy, SandboxedCodeTool};
pub use fs::{SandboxedListDirTool, SandboxedReadFileTool, SandboxedWriteFileTool};
pub use manifest::{ManifestError, SkillManifest, parse_skill_md};
pub use mock::MockSandbox;
pub use policy::{ShellPolicy, ShellPolicyError};
pub use shell::SandboxedShellTool;
pub use skill::{SandboxResource, SandboxSkill};
