//! Shared visitor utilities for invariant gates. Centralises file walking,
//! source loading, and the `Violation` reporting shape so each per-invariant
//! module reads as a focused predicate.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use walkdir::WalkDir;

/// One detected violation. Always carries `path:line:col` so editors can jump
/// straight to it; `message` is the actionable fragment a reviewer sees.
pub(crate) struct Violation {
    pub(crate) path: PathBuf,
    pub(crate) line: usize,
    pub(crate) col: usize,
    pub(crate) message: String,
}

impl Violation {
    pub(crate) fn new(
        path: impl Into<PathBuf>,
        line: usize,
        col: usize,
        message: impl Into<String>,
    ) -> Self {
        Self {
            path: path.into(),
            line,
            col,
            message: message.into(),
        }
    }
    /// File-level violation (no source line). Used by manifest checks.
    pub(crate) fn file(path: impl Into<PathBuf>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            line: 0,
            col: 0,
            message: message.into(),
        }
    }
}

/// Print every violation in the canonical `path:line:col: msg` form, prefix the
/// reviewer-facing remediation block, and return `Err` so the binary exits 1.
pub(crate) fn report(invariant: &str, violations: Vec<Violation>, remediation: &str) -> Result<()> {
    if violations.is_empty() {
        return Ok(());
    }
    let mut out = String::new();
    writeln!(out, "{invariant} — {} violation(s):", violations.len()).ok();
    for v in &violations {
        if v.line == 0 {
            writeln!(out, "  {}: {}", v.path.display(), v.message).ok();
        } else {
            writeln!(
                out,
                "  {}:{}:{}: {}",
                v.path.display(),
                v.line,
                v.col,
                v.message
            )
            .ok();
        }
    }
    writeln!(out).ok();
    writeln!(out, "{remediation}").ok();
    anyhow::bail!("{out}");
}

/// All `.rs` files under `crates/`. Skips `target/` and `tests/` directories
/// ( — invariant gates apply to first-party production code).
pub(crate) fn rust_source_files(repo_root: &Path) -> Vec<PathBuf> {
    let crates = repo_root.join("crates");
    let mut out = Vec::new();
    for entry in WalkDir::new(&crates).into_iter().filter_map(Result::ok) {
        let path = entry.path();
        if !path.is_file() || path.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        if path.components().any(|c| {
            let s = c.as_os_str().to_string_lossy();
            s == "target" || s == "tests"
        }) {
            continue;
        }
        out.push(path.to_path_buf());
    }
    out.sort();
    out
}

/// All `.rs` files under one specific path (file or directory). Same skip rules.
pub(crate) fn rust_source_files_in(path: &Path) -> Vec<PathBuf> {
    if path.is_file() {
        return if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            vec![path.to_path_buf()]
        } else {
            vec![]
        };
    }
    let mut out = Vec::new();
    for entry in WalkDir::new(path).into_iter().filter_map(Result::ok) {
        let p = entry.path();
        if !p.is_file() || p.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        if p.components().any(|c| {
            let s = c.as_os_str().to_string_lossy();
            s == "target" || s == "tests"
        }) {
            continue;
        }
        out.push(p.to_path_buf());
    }
    out.sort();
    out
}

/// Read a source file from disk.
pub(crate) fn read(path: &Path) -> Result<String> {
    fs::read_to_string(path).with_context(|| format!("read {}", path.display()))
}

/// Parse a Rust source file into a `syn::File`. Caller drops the source string
/// once parsed; visitors operate on the AST, not text.
pub(crate) fn parse(path: &Path) -> Result<(String, syn::File)> {
    let src = read(path)?;
    let ast = syn::parse_file(&src).with_context(|| format!("parse {}", path.display()))?;
    Ok((src, ast))
}

/// Resolve `(line, col)` from a byte offset into `source`. `proc_macro2::Span`
/// is the more idiomatic carrier but file-relative byte offsets are what
/// `proc_macro2::LineColumn` reports through the `extra-traits` feature; this
/// helper exists for the few sites where we resolve our own offsets (comments,
/// macro fragments).
pub(crate) fn line_col(source: &str, offset: usize) -> (usize, usize) {
    let mut line = 1;
    let mut col = 1;
    for (i, ch) in source.char_indices() {
        if i >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}

/// Repo root — every subcommand assumes the binary is invoked via the
/// `cargo xtask` alias which sets `CARGO` and starts at workspace root.
pub(crate) fn repo_root() -> Result<PathBuf> {
    // `CARGO_MANIFEST_DIR` for xtask is `<repo>/xtask`, so take its parent.
    let manifest = std::env::var("CARGO_MANIFEST_DIR").context("CARGO_MANIFEST_DIR unset")?;
    let path = PathBuf::from(manifest)
        .parent()
        .context("xtask manifest has no parent")?
        .to_path_buf();
    Ok(path)
}

/// Span → file-relative `(line, col)`. `proc_macro2::Span::start` returns
/// `LineColumn { line, column }` where line is 1-indexed and column is
/// 0-indexed; we normalise column to 1-indexed for editor parity.
pub(crate) fn span_loc(span: proc_macro2::Span) -> (usize, usize) {
    let lc = span.start();
    (lc.line, lc.column + 1)
}
