//! Shared visitor utilities for invariant gates. Centralises file walking,
//! source loading, and the `Violation` reporting shape so each per-invariant
//! module reads as a focused predicate.
//!
//! One pipeline drives every AST invariant: [`run_invariants`] parses
//! the workspace once into a shared [`WorkspaceParse`] cache, runs
//! every [`FileGate`] file-by-file against that cache, then runs every
//! [`WorkspaceGate`] over the same cache for cross-file checks. Each
//! workspace `.rs` is read + parsed exactly once per invocation
//! regardless of how many gates inspect it — no double-parse, no
//! "walk + reparse per trait name" regression.
//!
//! The pipeline is single-threaded by necessity. `syn::File` is
//! `!Send` because proc-macro2's `Span` enum wraps a native
//! `proc_macro::Span` variant, so ASTs cannot cross thread boundaries
//! and the cache cannot fan out across rayon workers. Bounded cost
//! (~300 files × ~µs per parse) keeps the wall-time profile within
//! the cadence budget; correctness + share-parse hygiene beats the
//! ~50ms parallelism win the prior shape extracted.
//!
//! [`report`] sorts violations by `(path, line, col, message)` before
//! printing so two runs against the same code produce byte-identical
//! output regardless of the visit order.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use walkdir::WalkDir;

/// One detected violation. Always carries `path:line:col` so editors can jump
/// straight to it; `message` is the actionable fragment a reviewer sees.
/// `path` is workspace-relative so the report renders the same form a
/// reviewer would type — `crates/entelix-core/src/lib.rs:12:3`.
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

/// Print every violation in the canonical `path:line:col: msg` form,
/// prefix the reviewer-facing remediation block, and return `Err` so
/// the binary exits 1. Sorts violations by `(path, line, col,
/// message)` so the report is deterministic — same input, same
/// output, every run.
pub(crate) fn report(invariant: &str, violations: Vec<Violation>, remediation: &str) -> Result<()> {
    if violations.is_empty() {
        return Ok(());
    }
    let mut violations = violations;
    violations.sort_by(|a, b| {
        a.path
            .cmp(&b.path)
            .then(a.line.cmp(&b.line))
            .then(a.col.cmp(&b.col))
            .then_with(|| a.message.cmp(&b.message))
    });
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

/// Parse a Rust source file into a `syn::File`. Caller decides whether
/// to keep the source string (for raw-line scans) or drop it. Kept
/// alongside [`read`] because non-pipeline gates (e.g.
/// `facade_completeness`'s targeted re-parse of a single facade
/// `lib.rs`) need an ad-hoc parse path outside the orchestrator.
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

/// One sequenced cadence step — name shown in the banner, closure that
/// runs the underlying check. Failure short-circuits the cadence.
/// Distinct from [`FileGate`] / [`WorkspaceGate`] — those are the AST
/// dispatch shapes; this typedef is the outer cadence sequencer.
pub(crate) type Gate = (&'static str, fn() -> Result<()>);

/// One parsed workspace file in the shared cache.
pub(crate) struct ParsedFile {
    /// Path relative to the workspace root (e.g.
    /// `crates/entelix-core/src/lib.rs`). Stable across machines.
    pub(crate) rel: PathBuf,
    /// File source text — retained so file gates that scan raw lines
    /// (`no-shims` comment markers, `silent-fallback` marker
    /// detection) operate on the same string the parser consumed
    /// without a second `read_to_string`.
    pub(crate) src: String,
    /// Parsed Rust AST. Every gate that inspects this file borrows
    /// the same instance — no double-parse, no thread-shipping
    /// (`syn::File` is `!Send`).
    pub(crate) ast: syn::File,
}

/// Single-threaded parse cache shared across every gate in one
/// pipeline invocation. Built once by [`run_invariants`] and iterated
/// by every [`FileGate`] and [`WorkspaceGate`] in turn.
pub(crate) struct WorkspaceParse {
    files: Vec<ParsedFile>,
}

impl WorkspaceParse {
    pub(crate) fn iter(&self) -> impl Iterator<Item = &ParsedFile> {
        self.files.iter()
    }
}

/// Single-file invariant visitor. The orchestrator dispatches the
/// shared AST to every gate whose [`Self::applies_to`] matches —
/// each file is parsed once across the whole pipeline.
///
/// `applies_to` operates on the **workspace-relative** path
/// (`crates/<crate>/src/<file>.rs`) so component-aware
/// [`Path::starts_with`] gives correct zone scoping. Substring
/// matching on absolute paths is reviewer-rejected because an
/// unrelated path component could coincidentally embed the zone
/// string.
pub(crate) trait FileGate: Send + Sync {
    /// Identifier reported on violation. Matches the corresponding
    /// `cargo xtask <name>` subcommand for editor / CI jump.
    fn name(&self) -> &'static str;

    /// Filter — return `false` to skip the gate for this path
    /// (zone-scoped invariants narrow to e.g. `crates/entelix-core/src`).
    /// Default: applies to every Rust source file the workspace walker
    /// surfaces.
    fn applies_to(&self, rel_path: &Path) -> bool {
        let _ = rel_path;
        true
    }

    /// Inspect one parsed file, push every detected violation into
    /// `violations`. `rel_path` is the workspace-relative form —
    /// visitors stash it verbatim in [`Violation::new`] so reports
    /// render `crates/.../file.rs:L:C`.
    fn visit(&self, rel_path: &Path, src: &str, ast: &syn::File, violations: &mut Vec<Violation>);

    /// Reviewer-facing remediation block printed beneath the
    /// per-violation list. Independent of `name` so the gate's
    /// remediation can survive a rename without docstring drift.
    fn remediation(&self) -> &'static str;
}

/// Cross-file invariant gate — runs after every [`FileGate`] against
/// the same shared [`WorkspaceParse`] cache. Use this shape for
/// checks that need a whole-workspace view (which file declares
/// trait X, what methods does it carry, …).
pub(crate) trait WorkspaceGate: Send + Sync {
    fn name(&self) -> &'static str;

    /// Inspect every parsed file (or a relevant subset) and push every
    /// detected violation into `violations`.
    fn check(&self, parse: &WorkspaceParse, violations: &mut Vec<Violation>);

    fn remediation(&self) -> &'static str;
}

/// Run the full invariant pipeline against the workspace.
///
/// Stages, all sharing one `WorkspaceParse` cache:
///
/// 1. **Parse** — every `.rs` file under `crates/` is read + parsed
///    once. A parse error surfaces immediately as `Err`; a malformed
///    Rust file is a genuine signal, not a "gate had nothing to
///    report" outcome.
/// 2. **File pass** — for each cached file, every [`FileGate`] whose
///    `applies_to` matches receives the shared AST. Violations
///    accumulate per gate.
/// 3. **Workspace pass** — every [`WorkspaceGate`] receives the full
///    cache. Violations accumulate per gate.
///
/// Fails fast at the first gate (file or workspace) that emitted
/// violations, in input order. Single-threaded — see module docs for
/// the `syn::File: !Send` constraint that forces this.
pub(crate) fn run_invariants(
    file_gates: &[Box<dyn FileGate>],
    workspace_gates: &[Box<dyn WorkspaceGate>],
) -> Result<()> {
    if file_gates.is_empty() && workspace_gates.is_empty() {
        return Ok(());
    }
    let root = repo_root()?;
    let files = rust_source_files(&root);

    // ── Stage 1: parse the workspace once ──
    let mut cache: Vec<ParsedFile> = Vec::with_capacity(files.len());
    for abs in &files {
        let rel = abs
            .strip_prefix(&root)
            .with_context(|| format!("strip repo root from {}", abs.display()))?
            .to_path_buf();
        let (src, ast) = parse(abs)?;
        cache.push(ParsedFile { rel, src, ast });
    }
    let parse_cache = WorkspaceParse { files: cache };

    // ── Stage 2: file pass over the shared cache ──
    if !file_gates.is_empty() {
        let mut per_gate: HashMap<&'static str, Vec<Violation>> = HashMap::new();
        for file in parse_cache.iter() {
            for gate in file_gates {
                if !gate.applies_to(&file.rel) {
                    continue;
                }
                let mut vs = Vec::new();
                gate.visit(&file.rel, &file.src, &file.ast, &mut vs);
                if !vs.is_empty() {
                    per_gate.entry(gate.name()).or_default().extend(vs);
                }
            }
        }
        for gate in file_gates {
            if let Some(violations) = per_gate.get(gate.name())
                && !violations.is_empty()
            {
                return report(gate.name(), violations.clone(), gate.remediation());
            }
        }
    }

    // ── Stage 3: workspace pass over the same cache ──
    for gate in workspace_gates {
        let mut violations = Vec::new();
        gate.check(&parse_cache, &mut violations);
        if !violations.is_empty() {
            return report(gate.name(), violations, gate.remediation());
        }
    }

    Ok(())
}
