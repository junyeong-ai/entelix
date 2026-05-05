//! Facade feature-isolation gate. Every facade feature must compile alone —
//! catches `feature = ["dep:foo"]` regressions where the pass-through to
//! `foo`'s own internal feature is missing. cargo's default `--all-features`
//! union masks this.

use std::process::Command;

use anyhow::Result;

use crate::visitor::{Violation, repo_root, report};

const FEATURES: &[&str] = &[
    "mcp",
    "mcp-chatmodel",
    "postgres",
    "redis",
    "otel",
    "aws",
    "gcp",
    "azure",
    "policy",
    "server",
    "embedders-openai",
    "vectorstores-qdrant",
    "vectorstores-pgvector",
    "graphmemory-pg",
];

pub(crate) fn run() -> Result<()> {
    let root = repo_root()?;
    let mut violations = Vec::new();

    for f in FEATURES {
        if !cargo_check(&root, &["--no-default-features", "--features", f])? {
            violations.push(Violation::file(
                root.join("crates/entelix/Cargo.toml"),
                format!("feature `{f}` does not compile in isolation"),
            ));
        }
    }
    if !cargo_check(&root, &["--no-default-features", "--features", "full"])? {
        violations.push(Violation::file(
            root.join("crates/entelix/Cargo.toml"),
            "feature `full` does not compile",
        ));
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
