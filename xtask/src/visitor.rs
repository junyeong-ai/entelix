//! Shared visitor utilities for invariant gates. Centralises file walking,
//! source loading, and the `Violation` reporting shape so each per-invariant
//! module reads as a focused predicate.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rayon::prelude::*;
use walkdir::WalkDir;

/// One detected violation. Always carries `path:line:col` so editors can jump
/// straight to it; `message` is the actionable fragment a reviewer sees.
#[derive(Clone)]
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

/// One sequenced gate — name shown in the banner, closure that runs
/// the underlying check. Failure short-circuits the cadence. Shared
/// between [`crate::gates`] (bundled cadence orchestration) and the
/// per-invariant CLI entry points.
pub(crate) type Gate = (&'static str, fn() -> Result<()>);

/// Single-file invariant visitor — share-parse contract. The
/// orchestrator walks every workspace `.rs` once, parses one
/// `syn::File` per file, and dispatches to every [`FileGate`] whose
/// [`Self::applies_to`] is true. Each gate sees the same AST without
/// re-parsing.
///
/// Compared to the legacy "each invariant walks the workspace itself"
/// shape, this collapses N×file_count parses into file_count parses,
/// and rayon-parallelises across cores. Adding a new AST invariant is
/// O(visitor logic), not O(another full workspace walk).
pub(crate) trait FileGate: Send + Sync {
    /// Identifier reported on violation. Matches the corresponding
    /// `cargo xtask <name>` subcommand for editor / CI jump.
    fn name(&self) -> &'static str;

    /// Filter — return `false` to skip the gate for this path
    /// (zone-scoped invariants narrow to e.g. `crates/entelix-core/src`).
    /// Default: applies to every Rust source file the workspace walker
    /// surfaces.
    fn applies_to(&self, path: &Path) -> bool {
        let _ = path;
        true
    }

    /// Inspect one parsed file, push every detected violation into
    /// `violations`. The orchestrator merges and reports in
    /// canonical gate order at the end.
    fn visit(&self, path: &Path, src: &str, ast: &syn::File, violations: &mut Vec<Violation>);

    /// Reviewer-facing remediation block printed beneath the
    /// per-violation list. Independent of `name` so the gate's
    /// remediation can survive a rename without docstring drift.
    fn remediation(&self) -> &'static str;
}

/// Run every gate in `gates` against every `.rs` file under
/// `crates/`. Parses each file exactly once, dispatches to applicable
/// gates, collects violations, then reports in the input slice order
/// (fail-fast at the first gate with violations).
///
/// Parallelised across cores via rayon — a workspace of ~300 files
/// fans out to N visitors per file across the available CPUs. File
/// I/O + parse cost is amortised across every visitor that applies.
pub(crate) fn run_file_gates(gates: &[Box<dyn FileGate>]) -> Result<()> {
    if gates.is_empty() {
        return Ok(());
    }
    let root = repo_root()?;
    let files = rust_source_files(&root);

    let per_gate: HashMap<&'static str, Vec<Violation>> = files
        .par_iter()
        .map(|path| -> HashMap<&'static str, Vec<Violation>> {
            // Build the applicable-gate slice before parsing so files
            // outside every gate's scope skip the parse entirely.
            let applicable: Vec<&dyn FileGate> = gates
                .iter()
                .filter(|g| g.applies_to(path))
                .map(|g| g.as_ref())
                .collect();
            if applicable.is_empty() {
                return HashMap::new();
            }
            let Ok((src, ast)) = parse(path) else {
                return HashMap::new();
            };
            let mut local: HashMap<&'static str, Vec<Violation>> = HashMap::new();
            for gate in applicable {
                let mut vs = Vec::new();
                gate.visit(path, &src, &ast, &mut vs);
                if !vs.is_empty() {
                    local.entry(gate.name()).or_default().extend(vs);
                }
            }
            local
        })
        .reduce(HashMap::new, |mut a, b| {
            for (k, v) in b {
                a.entry(k).or_default().extend(v);
            }
            a
        });

    // Report in input order — deterministic fail-fast that respects
    // the canonical CI sequence callers pass.
    for gate in gates {
        if let Some(violations) = per_gate.get(gate.name())
            && !violations.is_empty()
        {
            return report(gate.name(), violations.clone(), gate.remediation());
        }
    }
    Ok(())
}
