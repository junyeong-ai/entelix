//! Per-crate public-API drift gate. Compares the live `cargo public-api`
//! output against `docs/public-api/<crate>.txt`. The companion
//! `cargo xtask freeze-public-api [<crate>...]` regenerates baselines after
//! a deliberate API change.

use std::process::Command;

use anyhow::Result;

use crate::visitor::{Violation, repo_root, report};

const CRATES: &[&str] = &[
    "entelix-core",
    "entelix-runnable",
    "entelix-prompt",
    "entelix-graph",
    "entelix-graph-derive",
    "entelix-tool-derive",
    "entelix-session",
    "entelix-memory",
    "entelix-memory-openai",
    "entelix-memory-pgvector",
    "entelix-memory-qdrant",
    "entelix-graphmemory-pg",
    "entelix-rag",
    "entelix-persistence",
    "entelix-tools",
    "entelix-tools-coding",
    "entelix-mcp",
    "entelix-cloud",
    "entelix-policy",
    "entelix-otel",
    "entelix-server",
    "entelix-agents",
];

pub(crate) fn run() -> Result<()> {
    if !command_exists("cargo-public-api") {
        eprintln!(
            "public-api: cargo-public-api not installed — skipping (install with `cargo install cargo-public-api --locked`)"
        );
        return Ok(());
    }
    let root = repo_root()?;
    let baseline_dir = root.join("docs/public-api");
    let mut violations = Vec::new();

    for c in CRATES {
        let baseline = baseline_dir.join(format!("{c}.txt"));
        if !baseline.exists() {
            violations.push(Violation::file(
                baseline.clone(),
                format!("baseline missing — run `cargo xtask freeze-public-api {c}`"),
            ));
            continue;
        }
        let live = run_public_api(&root, c)?;
        let baseline_text = std::fs::read_to_string(&baseline)?;
        if normalise(&live) != normalise(&baseline_text) {
            violations.push(Violation::file(
                baseline,
                format!(
                    "public-API drift in {c} — refreeze with `cargo xtask freeze-public-api {c}` if intentional"
                ),
            ));
        }
    }

    report(
        "public-api",
        violations,
        "Refreeze deliberately:\n  cargo xtask freeze-public-api <crate>\n\
         \n\
         Or refreeze every crate:\n  cargo xtask freeze-public-api",
    )
}

pub(crate) fn freeze(crates: &[String]) -> Result<()> {
    if !command_exists("cargo-public-api") {
        anyhow::bail!(
            "cargo-public-api not installed; run: cargo install cargo-public-api --locked"
        );
    }
    let root = repo_root()?;
    let baseline_dir = root.join("docs/public-api");
    std::fs::create_dir_all(&baseline_dir)?;
    let targets: Vec<&str> = if crates.is_empty() {
        CRATES.to_vec()
    } else {
        crates.iter().map(String::as_str).collect()
    };
    for c in targets {
        let live = run_public_api(&root, c)?;
        let path = baseline_dir.join(format!("{c}.txt"));
        std::fs::write(&path, &live)?;
        let lines = live.lines().count();
        println!("  → {} ({lines} lines)", path.display());
    }
    println!("\n✓ baselines refreshed.");
    Ok(())
}

fn run_public_api(root: &std::path::Path, crate_name: &str) -> Result<String> {
    let out = Command::new("cargo")
        .args(["public-api", "-p", crate_name, "--simplified"])
        .current_dir(root)
        .output()?;
    if !out.status.success() {
        anyhow::bail!(
            "cargo public-api failed for {crate_name}:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(String::from_utf8(out.stdout)?)
}

fn normalise(s: &str) -> String {
    s.trim().to_string()
}

fn command_exists(name: &str) -> bool {
    Command::new(name)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}
