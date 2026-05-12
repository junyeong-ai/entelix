//! # entelix-tools
//!
//! Built-in [`entelix_core::tools::Tool`] impls. First-party code
//! touches zero `std::fs` / `std::process` (Invariant 9); shell- and
//! filesystem-class tools delegate execution through the
//! [`entelix_core::sandbox::Sandbox`] trait, whose concrete
//! implementations ship as 1.x companion crates.
//!
//! ## Surface
//!
//! - [`HttpFetchTool`] — HTTP fetch with **mandatory host allowlist**
//!   (SSRF defense), redirect cap, response-size cap, method
//!   allowlist, cancellation-aware. The allowlist is *opt-in by
//!   construction*: an `HttpFetchTool::builder()` without any allowed
//!   hosts produces a tool that refuses every URL.
//! - [`HostAllowlist`] — domain / wildcard / literal-IP rules with a
//!   fail-closed default policy.
//! - [`Calculator`] — arithmetic over `f64` (`+ - * / parens`).
//!   Recursive-descent parser; no `eval` or shell-out. Generated
//!   from a free async fn via the [`tool`] macro.
//! - [`SearchProvider`] (trait) + [`SearchTool`] — adapter for
//!   external search APIs (Brave / Tavily / Perplexity / …).
//!   Concrete providers are deferred to 1.1 (same trait-only policy
//!   as for `Embedder`).
//!
//! Coding-agent vertical tools (`SandboxedShellTool`,
//! `SandboxedCodeTool`, `Sandboxed{Read,Write,ListDir}FileTool`,
//! `ShellPolicy`, `CodePolicy`, `SandboxSkill`) live in the
//! `entelix-tools-coding` companion crate so this horizontal surface
//! stays free of coding-shape opinions.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_root_url = "https://docs.rs/entelix-tools/0.5.2")]
#![deny(missing_docs)]
// Doc-prose lints fire on legitimate proper nouns (HTTP, URL, SSRF,
// API names) and on the `RFC1918` shorthand; the redundant_pub_crate
// lint disagrees with the workspace `unreachable_pub` rule for items
// inside private modules. `Tool` trait methods return `&str` whose
// lifetime is bound by the trait — clippy keeps suggesting elision
// that the trait signature forbids.
#![allow(
    clippy::doc_markdown,
    clippy::elidable_lifetime_names,
    clippy::equatable_if_let,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::missing_const_for_fn,
    clippy::needless_lifetimes,
    clippy::needless_pass_by_value,
    clippy::option_if_let_else,
    clippy::redundant_pub_crate,
    clippy::significant_drop_tightening,
    clippy::unnecessary_literal_bound
)]

// Make the crate self-referential so the `#[tool]` proc-macro's
// generated `::entelix_tools::SchemaTool` paths resolve when the
// macro is invoked from inside this crate. Standard Rust hygiene
// pattern for proc-macro consumers that re-export their own macro.
extern crate self as entelix_tools;

pub mod calculator;
mod dns;
mod error;
mod http_fetch;
pub mod memory;
mod schema_tool;
mod search;
pub mod skills;

pub use calculator::{Calculator, CalculatorInput, CalculatorOutput};
pub use dns::{SsrfSafeDnsResolver, is_ssrf_blocked};
/// `#[tool]` attribute macro — generates a [`SchemaTool`] impl
/// from an `async fn` signature. Doc-comment first paragraph
/// becomes the tool description; the function name (snake_case)
/// becomes the tool struct name (PascalCase). See the
/// `entelix-tool-derive` crate docs for the full contract.
pub use entelix_tool_derive::tool;
pub use error::{ToolError, ToolResult};
pub use http_fetch::{
    DEFAULT_FETCH_TIMEOUT, DEFAULT_MAX_REDIRECTS, DEFAULT_MAX_RESPONSE_BYTES, HostAllowlist,
    HostRule, HttpFetchTool, HttpFetchToolBuilder,
};
pub use schema_tool::{SchemaTool, SchemaToolAdapter, SchemaToolExt};
pub use search::{DEFAULT_MAX_RESULTS, SearchProvider, SearchResult, SearchTool};
pub use skills::{
    ActivateSkillTool, InMemorySkill, InMemorySkillBuilder, ListSkillsTool, ReadSkillResourceTool,
    StaticResource,
};
