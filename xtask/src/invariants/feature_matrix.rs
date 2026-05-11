//! Facade feature-isolation gate. Every facade feature must compile alone —
//! catches `feature = ["dep:foo"]` regressions where the pass-through to
//! `foo`'s own internal feature is missing. cargo's default `--all-features`
//! union masks this.
//!
//! The feature list is **discovered** from `crates/entelix/Cargo.toml`'s
//! `[features]` table at runtime, not hand-maintained here. CLAUDE.md
//! declares the facade `Cargo.toml` the single source of truth for
//! features; this gate honours that by parsing it.

use std::process::Command;

use anyhow::{Context, Result};

use crate::visitor::{Violation, read, repo_root, report};

pub(crate) fn run() -> Result<()> {
    let root = repo_root()?;
    let features = discover_facade_features(&root)?;
    let mut violations = Vec::new();

    for f in &features {
        if !cargo_check(&root, &["--no-default-features", "--features", f])? {
            violations.push(Violation::file(
                root.join("crates/entelix/Cargo.toml"),
                format!("feature `{f}` does not compile in isolation"),
            ));
        }
    }
    if !cargo_check(&root, &["--no-default-features"])? {
        violations.push(Violation::file(
            root.join("crates/entelix/Cargo.toml"),
            "no-default-features build fails",
        ));
    }

    report(
        "feature-matrix",
        violations,
        "Most common cause: `feature = [\"dep:foo\"]` enables the dep but\n\
         omits the pass-through to foo's internal feature, e.g.\n\
         \n  postgres = [\"dep:entelix-persistence\"]                              # broken\n  postgres = [\"dep:entelix-persistence\", \"entelix-persistence/postgres\"]  # ok\n\
         \n\
         Reproduce locally with:\n  cargo check -p entelix --no-default-features --features <FEATURE>",
    )
}

/// Parse `crates/entelix/Cargo.toml` and return every entry in its
/// `[features]` table except `default` (the empty starting point) —
/// `full` and every individual feature must each compile alone.
fn discover_facade_features(root: &std::path::Path) -> Result<Vec<String>> {
    let path = root.join("crates/entelix/Cargo.toml");
    let text = read(&path)?;
    let doc: toml_edit::DocumentMut = text
        .parse()
        .with_context(|| format!("parse {}", path.display()))?;
    let features = doc
        .get("features")
        .and_then(|v| v.as_table())
        .with_context(|| format!("`[features]` table missing in {}", path.display()))?;
    let mut out: Vec<String> = features
        .iter()
        .map(|(name, _)| name.to_owned())
        .filter(|n| n != "default")
        .collect();
    out.sort();
    Ok(out)
}

fn cargo_check(root: &std::path::Path, extra: &[&str]) -> Result<bool> {
    let mut args = vec!["check", "-p", "entelix", "--quiet"];
    args.extend(extra.iter().copied());
    let status = Command::new("cargo")
        .args(&args)
        .current_dir(root)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()?;
    Ok(status.success())
}
