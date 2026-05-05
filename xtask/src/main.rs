//! `cargo xtask <invariant>` — workspace invariant enforcement.
//!
//! Single binary, one subcommand per CLAUDE.md invariant. Visitors operate on
//! typed `syn` / `toml_edit` ASTs (no regex over source) so the patterns
//! `pub async fn`, trait-default methods, fully-qualified `std::fs::read(...)`
//! calls, and `include_str!` macros are all caught — failure modes the prior
//! shell scripts could not see. Per-visitor scope and rationale: ADR-0073.
//!
//! Each subcommand exits `0` on clean and `1` on at least one violation,
//! prints `path:line:col` plus actionable remediation, and is invoked
//! identically from local shell and from `.github/workflows/ci.yml`.

#![forbid(unsafe_code)]

use std::process::ExitCode;

use clap::{Parser, Subcommand};

mod invariants;
mod visitor;

#[derive(Parser)]
#[command(
    name = "xtask",
    about = "entelix invariant enforcement (CLAUDE.md §Architecture invariants)",
    disable_help_subcommand = true
)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run every invariant check in the canonical order. Equivalent to
    /// the preflight + supply-chain + feature-matrix + public-api CI jobs.
    Invariants,

    /// Invariant 9 — no `std::fs` / `std::process` / `tokio::fs` /
    /// sandbox crates, no FQP calls, no `include_str!` / `include_bytes!`.
    NoFs,

    /// Invariant 14 — no `#[deprecated]`, no `// deprecated`, no
    /// `pub use OldName as NewName`, no `// formerly` / `// removed for backcompat`.
    NoShims,

    /// Invariant 15 — no unaudited `unwrap_or*` in codecs / transports / cost
    /// meter. Every site requires a `// silent-fallback-ok: <reason>` marker.
    SilentFallback,

    /// Invariant 17 — no probability-shaped literals (`0.X`) in heuristic
    /// hot paths. Every site requires a `// magic-ok: <reason>` marker.
    MagicConstants,

    /// Invariants 1, 2, 4, 10 + ADR-0035 — managed-agent shape.
    ManagedShape,

    /// ADR-0010 naming taxonomy — forbidden suffixes, `get_*`, `with_*(&self)`,
    /// builder verb-prefix, ctx parameter ordering. Catches `pub async fn`
    /// and trait-default methods the prior regex missed.
    Naming,

    /// `#[non_exhaustive]` on every public enum + Tier-1 config struct;
    /// `#[source]` / `#[from]` on error variants carrying inner errors.
    SurfaceHygiene,

    /// CLAUDE.md §"Lock ordering" — `await_holding_lock` +
    /// `await_holding_refcell_ref` clippy lints pinned at deny level.
    LockOrdering,

    /// `[workspace.dependencies]` hygiene — every entry inherited by ≥1 crate.
    DeadDeps,

    /// Live operator-facing docs reference facade paths (`entelix::Foo`),
    /// not underlying crate paths (`entelix_core::Foo`).
    DocCanonicalPaths,

    /// Every sub-crate `pub use` item is reachable through `entelix::*`.
    FacadeCompleteness,

    /// `cargo audit` (RustSec CVE) + `cargo deny check`
    /// (license + bans + transitive).
    SupplyChain,

    /// Every facade feature compiles in isolation.
    FeatureMatrix,

    /// Per-crate public-API drift against `docs/public-api/<crate>.txt`.
    PublicApi,

    /// Refreeze public-API baselines into `docs/public-api/<crate>.txt`.
    /// Run only after a deliberate, ADR-documented API change.
    FreezePublicApi {
        /// Specific crates to refreeze. Empty = all baselines.
        crates: Vec<String>,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let outcome = match cli.command {
        Cmd::Invariants => run_all(),
        Cmd::NoFs => invariants::no_fs::run(),
        Cmd::NoShims => invariants::no_shims::run(),
        Cmd::SilentFallback => invariants::silent_fallback::run(),
        Cmd::MagicConstants => invariants::magic_constants::run(),
        Cmd::ManagedShape => invariants::managed_shape::run(),
        Cmd::Naming => invariants::naming::run(),
        Cmd::SurfaceHygiene => invariants::surface_hygiene::run(),
        Cmd::LockOrdering => invariants::lock_ordering::run(),
        Cmd::DeadDeps => invariants::dead_deps::run(),
        Cmd::DocCanonicalPaths => invariants::doc_canonical_paths::run(),
        Cmd::FacadeCompleteness => invariants::facade_completeness::run(),
        Cmd::SupplyChain => invariants::supply_chain::run(),
        Cmd::FeatureMatrix => invariants::feature_matrix::run(),
        Cmd::PublicApi => invariants::public_api::run(),
        Cmd::FreezePublicApi { crates } => invariants::public_api::freeze(&crates),
    };
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("\n✗ {e:#}");
            ExitCode::FAILURE
        }
    }
}

/// One static-analysable gate in the canonical CI order.
type Gate = (&'static str, fn() -> anyhow::Result<()>);

/// Runs every invariant check in the canonical CI order. Stops at the first
/// failing gate so output stays focused — matches the preflight job pattern.
fn run_all() -> anyhow::Result<()> {
    let gates: &[Gate] = &[
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
    ];
    for (name, gate) in gates {
        println!("── {name}");
        gate().map_err(|e| anyhow::anyhow!("{name}: {e:#}"))?;
    }
    println!("\n✓ {} invariants — all clean.", gates.len());
    Ok(())
}
