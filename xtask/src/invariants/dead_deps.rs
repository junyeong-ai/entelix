//! `[workspace.dependencies]` hygiene — every entry must be inherited by ≥1
//! sub-crate via `<dep> = { workspace = true }`. Internal `entelix-*` refs
//! are exempt because they pin the lockstep workspace version.

use anyhow::Result;
use toml_edit::{DocumentMut, Item, Value};
use walkdir::WalkDir;

use crate::visitor::{Violation, repo_root, report};

pub(crate) fn run() -> Result<()> {
    let root = repo_root()?;
    let manifest = std::fs::read_to_string(root.join("Cargo.toml"))?;
    let doc: DocumentMut = manifest.parse()?;

    let workspace_deps = doc
        .get("workspace")
        .and_then(|w| w.get("dependencies"))
        .and_then(|d| d.as_table())
        .map(|t| t.iter().map(|(k, _)| k.to_string()).collect::<Vec<_>>())
        .unwrap_or_default();

    // Collect every sub-crate Cargo.toml dep / dev-dep / build-dep that
    // sets `workspace = true`.
    let mut inherited: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for entry in WalkDir::new(root.join("crates"))
        .max_depth(2)
        .into_iter()
        .filter_map(std::result::Result::ok)
    {
        if entry.file_name() != "Cargo.toml" {
            continue;
        }
        let src = std::fs::read_to_string(entry.path())?;
        let crate_doc: DocumentMut = src.parse()?;
        for section in ["dependencies", "dev-dependencies", "build-dependencies"] {
            if let Some(tbl) = crate_doc.get(section).and_then(Item::as_table) {
                for (name, item) in tbl.iter() {
                    if uses_workspace(item) {
                        inherited.insert(name.to_string());
                    }
                }
            }
        }
    }

    // xtask itself is allowed to introduce its own (non-workspace-inherited)
    // deps — it's the enforcement binary, not a published crate.
    let xtask_manifest = root.join("xtask/Cargo.toml");
    if xtask_manifest.exists() {
        let src = std::fs::read_to_string(&xtask_manifest)?;
        let xtask_doc: DocumentMut = src.parse()?;
        if let Some(tbl) = xtask_doc.get("dependencies").and_then(Item::as_table) {
            for (name, item) in tbl.iter() {
                if uses_workspace(item) {
                    inherited.insert(name.to_string());
                }
            }
        }
    }

    let mut violations = Vec::new();
    for key in &workspace_deps {
        if key.starts_with("entelix-") {
            // Internal refs pin lockstep version; exempt.
            continue;
        }
        if !inherited.contains(key) {
            violations.push(Violation::file(
                root.join("Cargo.toml"),
                format!("[workspace.dependencies] {key} — no sub-crate inherits via {{ workspace = true }}"),
            ));
        }
    }

    report(
        "dead-deps",
        violations,
        "Either remove the entry from [workspace.dependencies] or wire it into\n\
         the consuming crate via `<dep> = { workspace = true }` under\n\
         [dependencies] or [dev-dependencies]. Internal entelix-* refs are\n\
         exempt — they pin lockstep workspace version.",
    )
}

fn uses_workspace(item: &Item) -> bool {
    match item {
        Item::Value(Value::InlineTable(t)) => t
            .get("workspace")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        Item::Table(t) => t.get("workspace").and_then(Item::as_bool).unwrap_or(false),
        _ => false,
    }
}
