//! CLAUDE.md §"Lock ordering" — verifies the workspace clippy
//! `await_holding_lock` + `await_holding_refcell_ref` lints are pinned at
//! deny level. Together they statically forbid any mutex / RefCell guard
//! crossing an `.await` point. The cross-mutex *order* rule (`tenant >
//! session > checkpoint > memory > tool_registry > orchestrator`) is
//! reviewer-only — no parser can decide it.

use anyhow::Result;
use toml_edit::DocumentMut;

use crate::visitor::{Violation, repo_root, report};

const REQUIRED: &[&str] = &["await_holding_lock", "await_holding_refcell_ref"];

pub(crate) fn run() -> Result<()> {
    let root = repo_root()?;
    let manifest_path = root.join("Cargo.toml");
    let src = std::fs::read_to_string(&manifest_path)?;
    let doc: DocumentMut = src.parse()?;

    let clippy = doc
        .get("workspace")
        .and_then(|w| w.get("lints"))
        .and_then(|l| l.get("clippy"));

    let mut violations = Vec::new();
    for lint in REQUIRED {
        let pinned_deny = clippy
            .and_then(|c| c.get(lint))
            .and_then(|v| v.as_str())
            .map(|s| s == "deny")
            .unwrap_or(false);
        if !pinned_deny {
            violations.push(Violation::file(
                manifest_path.clone(),
                format!("[workspace.lints.clippy] {lint} = \"deny\" is missing"),
            ));
        }
    }
    report(
        "lock-ordering",
        violations,
        "CLAUDE.md §\"Lock ordering\" requires both lints at deny level.\n\
         Add under [workspace.lints.clippy] in workspace Cargo.toml:\n\
         \n  await_holding_lock = \"deny\"\n  await_holding_refcell_ref = \"deny\"",
    )
}
