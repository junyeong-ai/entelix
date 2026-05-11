//! Bundled gate orchestration — encodes the two cadences a single
//! source of truth so every contributor runs the same sequence at the
//! same boundary.
//!
//! The local cadence ([`run_local`]) is meant for the commit-time
//! discipline: cheap-enough to run between slices, comprehensive
//! enough to catch the bugs CI would otherwise reject. It runs
//! `cargo fmt --check`, `cargo clippy` over `--lib --bins --tests`
//! (catches test-file lint regressions without the example /
//! benchmark blow-up of `--all-targets`), the test sweep — through
//! `cargo nextest run` when nextest is installed, otherwise `cargo
//! test` — and the AST-walking invariant set. It deliberately omits
//! `--all-features`, public-API drift, feature-matrix isolation, and
//! supply-chain auditing — those reach across feature combinations CI
//! is paid to compile while the iteration loop should not be.
//!
//! The CI cadence ([`run_ci`]) is meant for push-time enforcement:
//! every gate the local cadence runs, plus `--all-features
//! --all-targets` clippy / test sweeps, doc-link verification under
//! `RUSTDOCFLAGS="-D warnings"`, public-API drift baselines, feature
//! isolation (each facade feature compiles alone), and supply-chain
//! auditing.
//!
//! Both cadences fail fast at the first red gate so output stays
//! focused — the contributor sees the failing step, not a wall of
//! later cascades.

use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::invariants;
use crate::visitor::{
    FileGate, Gate, WorkspaceGate, repo_root, run_file_gates, run_workspace_gates,
};

/// Run the local fast cadence — pre-commit discipline.
///
/// Sequence (each step fails fast):
/// 1. `cargo fmt --all -- --check`
/// 2. `cargo clippy --workspace --lib --bins --tests -- -D warnings`
///    (default features; tests covered without examples / benches)
/// 3. Tests — `cargo nextest run --workspace` if nextest is on PATH,
///    otherwise `cargo test --workspace` (default features)
/// 4. AST-walking invariants ([`run_all_ast`])
pub(crate) fn run_local() -> Result<()> {
    let steps: &[Gate] = &[
        ("fmt", fmt_check),
        (
            "clippy (default features, lib + bins + tests)",
            clippy_local,
        ),
        ("test (default features)", test_local),
        ("invariants (AST)", run_all_ast),
    ];
    run_sequence("gates", steps)
}

/// Run the full CI cadence — push-time / merge-gate enforcement.
///
/// Sequence (each step fails fast):
/// 1. `cargo fmt --all -- --check`
/// 2. `cargo clippy --workspace --all-features --all-targets -- -D warnings`
/// 3. `cargo test --workspace --all-features`
/// 4. `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps`
/// 5. AST-walking invariants
/// 6. Public-API drift baselines (`cargo xtask public-api`)
/// 7. Feature-matrix isolation (`cargo xtask feature-matrix`)
/// 8. Supply-chain audit (`cargo xtask supply-chain`)
pub(crate) fn run_ci() -> Result<()> {
    let steps: &[Gate] = &[
        ("fmt", fmt_check),
        ("clippy (all features, all targets)", clippy_full),
        ("test (all features)", test_full),
        ("doc (all features)", doc_full),
        ("invariants (AST)", run_all_ast),
        ("public-api drift", invariants::public_api::run),
        ("feature-matrix isolation", invariants::feature_matrix::run),
        ("supply-chain audit", invariants::supply_chain::run),
    ];
    run_sequence("gates-ci", steps)
}

fn run_sequence(banner: &str, steps: &[Gate]) -> Result<()> {
    println!("══ {banner}");
    for (name, step) in steps {
        println!("── {name}");
        step().with_context(|| name.to_string())?;
    }
    println!("\n✓ {banner} — all clean.");
    Ok(())
}

// ── cargo wrappers ───────────────────────────────────────────────────

fn fmt_check() -> Result<()> {
    run_cargo(&["fmt", "--all", "--", "--check"], None)
}

fn clippy_local() -> Result<()> {
    run_cargo(
        &[
            "clippy",
            "--workspace",
            "--lib",
            "--bins",
            "--tests",
            "--",
            "-D",
            "warnings",
        ],
        None,
    )
}

fn clippy_full() -> Result<()> {
    run_cargo(
        &[
            "clippy",
            "--workspace",
            "--all-features",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ],
        None,
    )
}

fn test_local() -> Result<()> {
    if has_nextest() {
        // nextest does not run doctests — `--doc` is the documented
        // workaround. Two-step keeps local coverage equivalent to a
        // plain `cargo test --workspace` regardless of which runner
        // the developer has installed.
        run_cargo(&["nextest", "run", "--workspace"], None)?;
        run_cargo(&["test", "--workspace", "--doc"], None)
    } else {
        run_cargo(&["test", "--workspace"], None)
    }
}

fn test_full() -> Result<()> {
    if has_nextest() {
        run_cargo(&["nextest", "run", "--workspace", "--all-features"], None)?;
        run_cargo(&["test", "--workspace", "--all-features", "--doc"], None)
    } else {
        run_cargo(&["test", "--workspace", "--all-features"], None)
    }
}

/// True when `cargo-nextest` is available on PATH — auto-detect so the
/// cadence picks up the faster test runner without configuration when
/// the developer has it installed. Falls back transparently to plain
/// `cargo test` on machines without it. Detection runs once per
/// cadence invocation; the result is cached for the process lifetime.
fn has_nextest() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned())) // silent-fallback-ok: PATH-resolved `cargo` is the universal fallback when CARGO is unset.
            .args(["nextest", "--version"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    })
}

fn doc_full() -> Result<()> {
    run_cargo(
        &["doc", "--workspace", "--all-features", "--no-deps"],
        Some(("RUSTDOCFLAGS", "-D warnings")),
    )
}

fn run_cargo(args: &[&str], env: Option<(&str, &str)>) -> Result<()> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned()); // silent-fallback-ok: PATH-resolved `cargo` is the universal fallback when CARGO is unset (e.g. running xtask binary directly).
    let mut cmd = Command::new(&cargo);
    cmd.current_dir(repo_root()?).args(args);
    if let Some((key, value)) = env {
        cmd.env(key, value);
    }
    let status = cmd
        .status()
        .with_context(|| format!("spawn {cargo} {}", args.join(" ")))?;
    if !status.success() {
        bail!("{cargo} {} exited with {status}", args.join(" "));
    }
    Ok(())
}

/// Canonical AST-walking invariant set, share-parse-aware. Three
/// pipeline stages:
///
///   1. **File pass** — every [`FileGate`] from the AST invariant
///      modules runs against every workspace `.rs` once. The
///      orchestrator parses each file exactly once and dispatches the
///      shared `syn::File` to every gate whose `applies_to` matches.
///   2. **Workspace pass** — every [`WorkspaceGate`] runs against the
///      shared parse cache the file pass produced. Cross-file checks
///      (per-trait declaration lookup, memory-pattern files) reuse
///      the ASTs without re-reading or re-parsing.
///   3. **Sequential tail** — manifest / markdown / cross-file gates
///      that aren't Rust-AST-shaped (lock-ordering, dead-deps, …)
///      run in canonical order.
///
/// Single source of truth for "which gates are static analysis" and
/// "in what order they fire". Adding a new file-level AST gate: add
/// its `file_gates()` collection here. Adding a workspace-level gate:
/// add its `workspace_gates()` collection here.
pub(crate) fn run_all_ast() -> Result<()> {
    // ── Stage 1: file pass — N invariants, 1 parse per file ──
    let mut file_gates: Vec<Box<dyn FileGate>> = Vec::new();
    file_gates.extend(invariants::no_fs::gates());
    file_gates.extend(invariants::managed_shape::gates());
    file_gates.extend(invariants::naming::file_gates());
    file_gates.extend(invariants::surface_hygiene::gates());
    file_gates.extend(invariants::silent_fallback::gates());
    file_gates.extend(invariants::magic_constants::gates());
    file_gates.extend(invariants::no_shims::gates());
    run_file_gates(&file_gates).with_context(|| "ast file-gates".to_string())?;

    // ── Stage 2: workspace pass — cross-file checks parse the
    // workspace once into a single-threaded cache that every
    // workspace gate borrows in succession. ──
    let mut workspace_gates: Vec<Box<dyn WorkspaceGate>> = Vec::new();
    workspace_gates.extend(invariants::naming::workspace_gates());
    run_workspace_gates(&workspace_gates).with_context(|| "ast workspace-gates".to_string())?;

    // ── Stage 3: manifest / cross-file / markdown gates — sequential tail ──
    let tail: &[Gate] = &[
        ("lock-ordering", invariants::lock_ordering::run),
        ("dead-deps", invariants::dead_deps::run),
        ("facade-completeness", invariants::facade_completeness::run),
        ("doc-canonical-paths", invariants::doc_canonical_paths::run),
    ];
    for (name, gate) in tail {
        gate().with_context(|| (*name).to_string())?;
    }
    Ok(())
}
