//! Invariant 9 — no `std::fs` / `std::process` / `tokio::fs` / sandbox crates.
//!
//! Three classes of access are flagged via `syn` AST visit:
//!   1. `use std::fs;` / `use tokio::fs::*;` style imports.
//!   2. Fully-qualified-path calls — `std::fs::read("...")`,
//!      `std::process::Command::new(...)`, `tokio::fs::write(...)`.
//!   3. Compile-time fs macros — `include_str!`, `include_bytes!`.
//!
//! Companion `cargo deny` `bans` section guards the dep graph
//! against `landlock` / `seatbelt` / `tree-sitter` even transitively.

use anyhow::Result;
use syn::spanned::Spanned;
use syn::visit::Visit;

use crate::visitor::{Violation, parse, repo_root, report, rust_source_files, span_loc};

/// Path-prefix denylist. The matcher checks `prefix == path[..prefix.len()]`,
/// so `["std", "fs"]` catches both `std::fs` and `std::fs::read`.
const FORBIDDEN_PREFIXES: &[&[&str]] = &[
    &["std", "fs"],
    &["std", "process"],
    &["std", "os", "unix", "process"],
    &["tokio", "fs"],
    &["tokio", "process"],
    &["landlock"],
    &["seatbelt"],
    &["tree_sitter"],
    &["nix"],
];

const FORBIDDEN_MACROS: &[&str] = &["include_str", "include_bytes"];

/// Crate paths exempted from invariant 9 because the crate's
/// raison d'être is operator-credential persistence — a documented
/// exception in CLAUDE.md. The exemption is per-crate, not
/// per-file, so a regression that adds shell / process behaviour
/// to the exempted crate still has to land via reviewer approval
/// rather than being caught by the gate.
const CREDENTIAL_STORAGE_EXEMPTIONS: &[&str] = &["crates/entelix-auth-claude-code/"];

pub(crate) fn run() -> Result<()> {
    let root = repo_root()?;
    let files = rust_source_files(&root);

    let mut violations = Vec::new();
    for file in &files {
        let rel = file.strip_prefix(&root).unwrap_or(file);
        let rel_str = rel.to_string_lossy();
        if CREDENTIAL_STORAGE_EXEMPTIONS
            .iter()
            .any(|prefix| rel_str.starts_with(prefix))
        {
            continue;
        }
        let (_, ast) = parse(file)?;
        let mut v = NoFsVisitor {
            file: file.clone(),
            violations: &mut violations,
        };
        v.visit_file(&ast);
    }

    report(
        "no-fs (invariant 9)",
        violations,
        "Invariant 9 — entelix is web-service-native. No filesystem,\n\
         no shell, no local sandbox. Replace `std::fs` reads with HTTP / Store /\n\
         user-supplied bytes; replace `std::process` shell-outs with first-class\n\
         tools or MCP servers; replace `include_str!` / `include_bytes!` with\n\
         runtime data. Credential-storage backends (e.g. \
         `entelix-auth-claude-code`) are the documented exception — fs access \
         there is bounded to the OAuth credential file the upstream Claude \
         Code CLI shares. Any other crate proposing fs access must amend \
         CLAUDE.md and this gate's exemption list together.",
    )
}

struct NoFsVisitor<'v> {
    file: std::path::PathBuf,
    violations: &'v mut Vec<Violation>,
}

impl<'v> NoFsVisitor<'v> {
    fn flag(&mut self, span: proc_macro2::Span, what: String) {
        let (line, col) = span_loc(span);
        self.violations
            .push(Violation::new(self.file.clone(), line, col, what));
    }
}

impl<'ast, 'v> Visit<'ast> for NoFsVisitor<'v> {
    fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
        let mut paths = Vec::new();
        gather_use_paths(&item.tree, &mut Vec::new(), &mut paths);
        for path in paths {
            for forbid in FORBIDDEN_PREFIXES {
                if has_prefix(&path, forbid) {
                    self.flag(
                        item.use_token.span,
                        format!("forbidden import — `use {}`", path.join("::")),
                    );
                    break;
                }
            }
        }
        syn::visit::visit_item_use(self, item);
    }

    fn visit_expr_path(&mut self, expr: &'ast syn::ExprPath) {
        let segments: Vec<String> = expr
            .path
            .segments
            .iter()
            .map(|s| s.ident.to_string())
            .collect();
        for forbid in FORBIDDEN_PREFIXES {
            if has_prefix(&segments, forbid) {
                self.flag(
                    expr.path.span(),
                    format!("forbidden FQP call — `{}`", segments.join("::")),
                );
                break;
            }
        }
        syn::visit::visit_expr_path(self, expr);
    }

    fn visit_macro(&mut self, m: &'ast syn::Macro) {
        if let Some(last) = m.path.segments.last() {
            let name = last.ident.to_string();
            if FORBIDDEN_MACROS.contains(&name.as_str()) {
                self.flag(
                    m.path.span(),
                    format!("forbidden macro — `{name}!` reads filesystem at compile time"),
                );
            }
        }
        syn::visit::visit_macro(self, m);
    }
}

// `syn::Path::span()` from the `Spanned` trait — bring it into method-call
// position via the import above. The duplicate trait below is left out — we
// use `syn::spanned::Spanned` directly.

fn gather_use_paths(tree: &syn::UseTree, prefix: &mut Vec<String>, out: &mut Vec<Vec<String>>) {
    match tree {
        syn::UseTree::Path(p) => {
            prefix.push(p.ident.to_string());
            gather_use_paths(&p.tree, prefix, out);
            prefix.pop();
        }
        syn::UseTree::Name(n) => {
            let mut full = prefix.clone();
            full.push(n.ident.to_string());
            out.push(full);
        }
        syn::UseTree::Rename(r) => {
            let mut full = prefix.clone();
            full.push(r.ident.to_string());
            out.push(full);
        }
        syn::UseTree::Glob(_) => {
            out.push(prefix.clone());
        }
        syn::UseTree::Group(g) => {
            for item in &g.items {
                gather_use_paths(item, prefix, out);
            }
        }
    }
}

fn has_prefix(path: &[String], prefix: &[&str]) -> bool {
    if path.len() < prefix.len() {
        return false;
    }
    path.iter()
        .zip(prefix.iter())
        .all(|(a, b)| a.as_str() == *b)
}
