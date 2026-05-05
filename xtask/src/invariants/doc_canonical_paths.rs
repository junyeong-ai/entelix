//! Live operator-facing docs (README, PLAN, docs/architecture, docs/migrations)
//! reference the facade path (`entelix::Type`) rather than the underlying
//! crate path (`entelix_core::Type`). Historical retrospectives in
//! `docs/adr/`, `CHANGELOG.md`, and per-crate `CLAUDE.md` are intentionally
//! exempt — they pin past type names or document the underlying crate's own
//! surface.

use anyhow::Result;
use walkdir::WalkDir;

use crate::visitor::{Violation, line_col, read, repo_root, report};

const SUBCRATES: &[&str] = &[
    "entelix_core",
    "entelix_runnable",
    "entelix_prompt",
    "entelix_graph",
    "entelix_graph_derive",
    "entelix_session",
    "entelix_memory",
    "entelix_memory_openai",
    "entelix_memory_qdrant",
    "entelix_memory_pgvector",
    "entelix_graphmemory_pg",
    "entelix_persistence",
    "entelix_tools",
    "entelix_mcp",
    "entelix_mcp_chatmodel",
    "entelix_cloud",
    "entelix_policy",
    "entelix_otel",
    "entelix_server",
    "entelix_agents",
];

pub(crate) fn run() -> Result<()> {
    let root = repo_root()?;
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    for top in ["README.md", "PLAN.md"] {
        let p = root.join(top);
        if p.exists() {
            files.push(p);
        }
    }
    for sub in ["docs/architecture", "docs/migrations"] {
        for entry in WalkDir::new(root.join(sub))
            .into_iter()
            .filter_map(std::result::Result::ok)
        {
            if entry.path().extension().and_then(|s| s.to_str()) == Some("md") {
                files.push(entry.path().to_path_buf());
            }
        }
    }

    let mut violations = Vec::new();
    for file in &files {
        let src = read(file)?;
        for sub in SUBCRATES {
            // `entelix_core::` etc. — only path-uses, not bare crate-name mentions.
            let needle = format!("{sub}::");
            let mut start = 0;
            while let Some(pos) = src[start..].find(&needle) {
                let abs = start + pos;
                let (line, col) = line_col(&src, abs);
                violations.push(Violation::new(
                    file.clone(),
                    line,
                    col,
                    format!("underlying crate path `{sub}::` — replace with `entelix::`"),
                ));
                start = abs + needle.len();
            }
        }
    }

    report(
        "doc-canonical-paths",
        violations,
        "README, docs/architecture/, docs/migrations/ are user-facing canonical\n\
         references — facade paths only. Replace each `entelix_<crate>::Type`\n\
         with `entelix::Type`. If the type is genuinely missing from the\n\
         facade, add the re-export to `crates/entelix/src/lib.rs` first and\n\
         let `cargo xtask facade-completeness` confirm. ADR-0064.",
    )
}
