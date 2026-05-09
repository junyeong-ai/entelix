//! Facade re-export coverage gate. Every `pub use` item in every sub-crate's
//! `lib.rs` must be reachable through the `entelix` facade — either via a
//! flat re-export, a sub-module re-export, or a parent-module re-export.
//!
//! Items intentionally excluded from the facade are listed inline in
//! [`EXCLUDES`] with the reason — keeping the list in source (rather than a
//! separate `.txt` data file) means the rationale travels with the gate.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::Result;
use syn::visit::Visit;

use crate::visitor::{Violation, parse, repo_root, report};

/// Items advertised by a sub-crate that we deliberately keep behind the
/// underlying crate path. Each line is `entelix-<crate>::<Item>` plus a
/// reason. Adding to this list requires the same review discipline as a
/// `// silent-fallback-ok:` marker.
const EXCLUDES: &[(&str, &str)] = &[
    // Low-level parallel-fanout dispatch primitives — `add_send_edges` is
    // the operator-facing surface; these are the internals it stitches.
    (
        "entelix-graph::Dispatch",
        "low-level parallel-fanout primitive",
    ),
    (
        "entelix-graph::scatter",
        "low-level parallel-fanout primitive",
    ),
    (
        "entelix-graph::FinalizingStream",
        "low-level parallel-fanout primitive",
    ),
    // ApprovalDecision lives in entelix-core and is re-exported from
    // entelix-agents for ergonomics; the canonical reach is through the
    // entelix-core path the facade already exposes.
    (
        "entelix-agents::ApprovalDecision",
        "duplicated re-export — canonical path is entelix-core",
    ),
];

const SUBCRATES: &[&str] = &[
    "entelix-core",
    "entelix-runnable",
    "entelix-prompt",
    "entelix-graph",
    "entelix-graph-derive",
    "entelix-tool-derive",
    "entelix-session",
    "entelix-memory",
    "entelix-memory-openai",
    "entelix-memory-qdrant",
    "entelix-memory-pgvector",
    "entelix-graphmemory-pg",
    "entelix-persistence",
    "entelix-tools",
    "entelix-mcp",
    "entelix-cloud",
    "entelix-policy",
    "entelix-otel",
    "entelix-server",
    "entelix-agents",
];

pub(crate) fn run() -> Result<()> {
    let root = repo_root()?;
    let facade = parse_pub_uses(&root.join("crates/entelix/src/lib.rs"))?;
    let excludes: BTreeSet<String> = EXCLUDES.iter().map(|(s, _)| s.to_string()).collect();

    let mut violations = Vec::new();
    for sub in SUBCRATES {
        let lib = root.join(format!("crates/{sub}/src/lib.rs"));
        if !lib.exists() {
            continue;
        }
        let crate_uses = parse_pub_uses(&lib)?;
        let crate_underscore = sub.replace('-', "_");
        for (parent_mod, items) in &crate_uses.items_per_module {
            for item in items {
                let key = format!("{sub}::{item}");
                if excludes.contains(&key) {
                    continue;
                }
                if facade
                    .flat_per_crate
                    .get(crate_underscore.as_str())
                    .map(|s| s.contains(item))
                    .unwrap_or(false)
                {
                    continue;
                }
                if let Some(modset) = facade.modules_per_crate.get(crate_underscore.as_str()) {
                    if modset.contains(parent_mod) {
                        continue;
                    }
                }
                violations.push(Violation::file(
                    lib.clone(),
                    format!("`{key}` is missing from the entelix facade"),
                ));
            }
        }
    }

    report(
        "facade-completeness",
        violations,
        "Two ways to resolve:\n\
         (a) Add the item to `crates/entelix/src/lib.rs` under the matching\n\
             `pub use entelix_X::{...}` block (cargo fmt enforces order).\n\
         (b) If the item is intentionally advanced/internal, add it to\n\
             EXCLUDES in xtask/src/invariants/facade_completeness.rs with\n\
             a reason. Each entry is reviewed at the same bar as a\n\
             `// silent-fallback-ok:` marker.",
    )
}

#[derive(Default)]
struct PubUses {
    /// `pub use entelix_X::{a, b};` → `flat["entelix_X"]` contains `{a, b}`.
    flat_per_crate: BTreeMap<String, BTreeSet<String>>,
    /// `pub use entelix_X::submod;` → `modules["entelix_X"]` contains `{submod}`.
    modules_per_crate: BTreeMap<String, BTreeSet<String>>,
    /// For sub-crate parsing — `pub use submod::Item;` → `items_per_module["submod"]`
    /// contains `Item`.
    items_per_module: BTreeMap<String, BTreeSet<String>>,
}

fn parse_pub_uses(file: &std::path::Path) -> Result<PubUses> {
    let (_, ast) = parse(file)?;
    let mut out = PubUses::default();
    let mut v = PubUseVisitor { out: &mut out };
    v.visit_file(&ast);
    Ok(out)
}

struct PubUseVisitor<'v> {
    out: &'v mut PubUses,
}

impl<'ast, 'v> Visit<'ast> for PubUseVisitor<'v> {
    fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
        if !matches!(item.vis, syn::Visibility::Public(_)) {
            return;
        }
        gather(&item.tree, &mut Vec::new(), self.out);
    }
}

fn gather(tree: &syn::UseTree, prefix: &mut Vec<String>, out: &mut PubUses) {
    match tree {
        syn::UseTree::Path(p) => {
            prefix.push(p.ident.to_string());
            gather(&p.tree, prefix, out);
            prefix.pop();
        }
        syn::UseTree::Name(n) => {
            let item = n.ident.to_string();
            register(prefix, &item, out);
        }
        syn::UseTree::Rename(r) => {
            // Two contexts share this code path:
            // - facade parsing (`pub use entelix_X::Item as MyAlias`) —
            //   the facade is reaching the *source* item, so coverage
            //   asks "is `Item` reachable?" → answer yes via the alias.
            //   Record the source ident.
            // - sub-crate parsing (`pub use submod::Item as Other`) —
            //   the sub-crate exposes `Other` to its consumers; the
            //   facade has to reach `Other`, not `Item`. Record alias.
            let is_facade = prefix
                .first()
                .map(|s| s.starts_with("entelix_"))
                .unwrap_or(false);
            let item = if is_facade {
                r.ident.to_string()
            } else {
                r.rename.to_string()
            };
            register(prefix, &item, out);
        }
        syn::UseTree::Glob(_) => {
            // Treat as a single "module" entry — the parent module path.
            // Glob in facade means "everything in that module", which the
            // sub-crate parsing maps to module-level coverage.
            if let Some(top) = prefix.first() {
                if top.starts_with("entelix_") {
                    if let Some(sub) = prefix.get(1) {
                        out.modules_per_crate
                            .entry(top.clone())
                            .or_default()
                            .insert(sub.clone());
                    }
                }
            }
        }
        syn::UseTree::Group(g) => {
            for it in &g.items {
                gather(it, prefix, out);
            }
        }
    }
}

fn register(prefix: &[String], item: &str, out: &mut PubUses) {
    if let Some(top) = prefix.first() {
        if top.starts_with("entelix_") {
            // Item is always reachable by name through the facade. Lowercase
            // singletons (`erase`, `tools`) are syntactically ambiguous —
            // could be a function or a module re-export — so we register
            // them in *both* maps. The downstream check accepts a sub-crate
            // item if either form covers it; this avoids the lossy heuristic
            // where `erase` (function) was misclassified as a module and
            // dropped from the flat map.
            out.flat_per_crate
                .entry(top.clone())
                .or_default()
                .insert(item.to_string());
            if looks_like_module(item) {
                out.modules_per_crate
                    .entry(top.clone())
                    .or_default()
                    .insert(item.to_string());
            }
            return;
        }
    }
    // Sub-crate `lib.rs` parsing — `pub use submod::Item`.
    if let Some(submod) = prefix.first() {
        out.items_per_module
            .entry(submod.clone())
            .or_default()
            .insert(item.to_string());
    }
}

/// Lowercase single word — *might* be a module. Real modules and bare
/// lowercase functions / single-word constants share this shape; we
/// register both views to stay correct without a sub-crate-scoped lookup.
fn looks_like_module(item: &str) -> bool {
    let first = item.chars().next();
    first.map(|c| c.is_ascii_lowercase()).unwrap_or(false) && !item.contains('_')
}
