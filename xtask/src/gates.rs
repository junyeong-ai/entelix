//! Bundled gate orchestration — encodes the two cadences a single
//! source of truth so every contributor runs the same sequence at the
//! same boundary.
//!
//! The local cadence ([`run_local`]) is meant for the commit-time
//! discipline: cheap-enough to run between slices, comprehensive
//! enough to catch the bugs CI would otherwise reject. It runs
//! `cargo fmt --check`, `cargo clippy` against the workspace's default
//! features (the lints fire on the same code paths the typical
//! consumer compiles), `cargo test` against the same scope, and the
//! AST-walking invariant set. It deliberately omits `--all-features`,
//! `--all-targets`, public-API drift, feature-matrix isolation, and
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
use crate::visitor::repo_root;

/// One sequenced gate — name shown in the banner, closure that runs
/// the underlying check. Failure short-circuits the cadence.
pub(crate) type Step = (&'static str, fn() -> Result<()>);

/// Run the local fast cadence — pre-commit discipline.
///
/// Sequence (each step fails fast):
/// 1. `cargo fmt --all -- --check`
/// 2. `cargo clippy --workspace -- -D warnings` (default features)
/// 3. `cargo test --workspace` (default features)
/// 4. AST-walking invariants ([`run_all_ast`])
pub(crate) fn run_local() -> Result<()> {
    let steps: &[Step] = &[
        ("fmt", fmt_check),
        ("clippy (default features)", clippy_local),
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
    let steps: &[Step] = &[
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

fn run_sequence(banner: &str, steps: &[Step]) -> Result<()> {
    println!("══ {banner}");
    for (name, step) in steps {
        println!("── {name}");
        step().with_context(|| format!("{name}"))?;
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
        &["clippy", "--workspace", "--", "-D", "warnings"],
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
    run_cargo(&["test", "--workspace"], None)
}

fn test_full() -> Result<()> {
    run_cargo(&["test", "--workspace", "--all-features"], None)
}

fn doc_full() -> Result<()> {
    run_cargo(
        &[
            "doc",
            "--workspace",
            "--all-features",
            "--no-deps",
        ],
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

/// Canonical AST-walking invariant set in CI order. Both the bundled
/// cadences and the standalone `cargo xtask invariants` consume this
/// list — single source of truth for "which gates are static
/// analysis" and "in what order they fire".
pub(crate) fn ast_gates() -> &'static [Step] {
    &[
        ("no-fs", invariants::no_fs::run),
        ("managed-shape", invariants::managed_shape::run),
        ("naming", invariants::naming::run),
        ("surface-hygiene", invariants::surface_hygiene::run),
        ("silent-fallback", invariants::silent_fallback::run),
        ("magic-constants", invariants::magic_constants::run),
        ("no-shims", invariants::no_shims::run),
        ("lock-ordering", invariants::lock_ordering::run),
        ("dead-deps", invariants::dead_deps::run),
        ("facade-completeness", invariants::facade_completeness::run),
        ("doc-canonical-paths", invariants::doc_canonical_paths::run),
    ]
}

/// Run every AST-walking invariant in canonical order, silently —
/// the surrounding bundled cadence prints one banner per logical
/// step, so the per-gate detail stays quiet unless one fires. Stops
/// at the first failure.
pub(crate) fn run_all_ast() -> Result<()> {
    for (name, gate) in ast_gates() {
        gate().with_context(|| (*name).to_string())?;
    }
    Ok(())
}
