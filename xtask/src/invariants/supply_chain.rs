//! Supply-chain hardening. Two layers:
//!
//!   1. `cargo audit` — RustSec advisory database (live fetch). Fails on any
//!      unfixed CVE not allow-listed in `.cargo/audit.toml` / `deny.toml`.
//!   2. `cargo deny check` — three-layer policy: advisories + licenses
//!      (OSI-permissive allow-list) + bans (no `landlock` / `seatbelt` /
//!      `tree-sitter` even transitively — invariant 9 dep-graph layer).

use std::process::Command;

use anyhow::Result;

use crate::visitor::{Violation, repo_root, report};

pub(crate) fn run() -> Result<()> {
    let root = repo_root()?;
    let mut violations = Vec::new();

    if !command_exists("cargo-audit") {
        anyhow::bail!("cargo-audit not installed; run: cargo install cargo-audit --locked");
    }
    if !command_exists("cargo-deny") {
        anyhow::bail!("cargo-deny not installed; run: cargo install cargo-deny --locked");
    }

    // cargo audit — exits non-zero on unfixed CVE. Captured + replayed
    // on failure so CI logs carry the actual advisory body without a
    // separate `cargo audit` re-run.
    let audit_output = Command::new("cargo")
        .args(["audit", "--deny", "warnings"])
        .current_dir(&root)
        .output()?;
    if !audit_output.status.success() {
        eprintln!("\n── cargo audit output ───────────────────────────────");
        eprint!("{}", String::from_utf8_lossy(&audit_output.stderr));
        eprint!("{}", String::from_utf8_lossy(&audit_output.stdout));
        eprintln!("─────────────────────────────────────────────────────\n");
        violations.push(Violation::file(
            root.join("Cargo.lock"),
            "cargo audit reported an unfixed CVE — output above",
        ));
    }

    // cargo deny — must produce the four-line all-ok summary. On
    // failure, replay both streams so CI logs surface the offending
    // advisory / license / ban / source.
    let deny_output = Command::new("cargo")
        .args(["deny", "check"])
        .current_dir(&root)
        .output()?;
    let combined = String::from_utf8_lossy(&deny_output.stderr).to_string()
        + &String::from_utf8_lossy(&deny_output.stdout);
    if !combined.contains("advisories ok, bans ok, licenses ok, sources ok") {
        eprintln!("\n── cargo deny check output ──────────────────────────");
        eprint!("{}", String::from_utf8_lossy(&deny_output.stderr));
        eprint!("{}", String::from_utf8_lossy(&deny_output.stdout));
        eprintln!("─────────────────────────────────────────────────────\n");
        violations.push(Violation::file(
            root.join("deny.toml"),
            "cargo deny check did not produce all-ok summary — output above",
        ));
    }

    report(
        "supply-chain",
        violations,
        "Each unfixed CVE must either (a) be patched by upgrading the offending\n\
         crate, or (b) be added to deny.toml [advisories.ignore] with a comment\n\
         explaining why the vulnerability is structurally absent in entelix's\n\
         usage. Disallowed licenses must be resolved by pinning to a different\n\
         version or adding the license to deny.toml [licenses.allow] after\n\
         legal review.",
    )
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
