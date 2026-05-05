//! Invariant 14 — no backwards-compatibility shims. When a name, type, or
//! signature changes, delete the old in the same PR. Standing project rule:
//! "처음부터 이렇게 설계된 것처럼 흔적 0".
//!
//! Flagged shapes:
//!   1. `#[deprecated]` / `#[deprecated(...)]` attributes on any item.
//!   2. `pub use OldName as NewName` re-exports where the alias signals
//!      a backcompat rename — `*Old` / `*Legacy` / `*_old`.
//!   3. Line comments — `// deprecated`, `// formerly`,
//!      `// removed for backcompat` (and variants).
//!
//! `syn` does not expose ordinary line comments; the comment scan operates
//! on raw source text. Doc comments (`///` / `//!`) are excluded because
//! they document the name a user is reading right now and never carry
//! shim semantics.

use anyhow::Result;
use syn::spanned::Spanned;
use syn::visit::Visit;

use crate::visitor::{Violation, parse, repo_root, report, rust_source_files, span_loc};

const ALIAS_TAILS: &[&str] = &["Old", "Legacy"];

const COMMENT_PATTERNS: &[&str] = &[
    "// deprecated",
    "//deprecated",
    "// formerly ",
    "// removed for backcompat",
    "// removed for backwards",
    "// removed for compat",
    "// removed for migration",
];

pub(crate) fn run() -> Result<()> {
    let root = repo_root()?;
    let files = rust_source_files(&root);

    let mut violations = Vec::new();
    for file in &files {
        let (src, ast) = parse(file)?;
        let mut v = NoShimsVisitor {
            file: file.clone(),
            violations: &mut violations,
        };
        v.visit_file(&ast);

        // Comment scan — line by line, skip doc comments.
        for (idx, raw) in src.lines().enumerate() {
            let trimmed = raw.trim_start();
            if trimmed.starts_with("///") || trimmed.starts_with("//!") {
                continue;
            }
            for pat in COMMENT_PATTERNS {
                if trimmed.contains(pat) {
                    violations.push(Violation::new(
                        file.clone(),
                        idx + 1,
                        raw.find(pat).unwrap_or(0) + 1,
                        format!("legacy comment marker — `{pat}`"),
                    ));
                    break;
                }
            }
        }
    }

    // Top-level docs (CLAUDE.md, README, ADRs) host the rule rationale and
    // include the literal patterns inside fenced examples — those are not
    // shims. We restrict the comment scan to `crates/` source already.

    report(
        "no-shims (invariant 14)",
        violations,
        "Delete the old name / type / signature in the same PR. No\n\
         `#[deprecated]`, no `pub use OldName as NewName`, no `// formerly`\n\
         comments. If a flagged comment is genuinely descriptive prose\n\
         (not a backcompat marker), rephrase to avoid the trigger words.",
    )
}

struct NoShimsVisitor<'v> {
    file: std::path::PathBuf,
    violations: &'v mut Vec<Violation>,
}

impl<'ast, 'v> Visit<'ast> for NoShimsVisitor<'v> {
    fn visit_attribute(&mut self, attr: &'ast syn::Attribute) {
        if attr.path().is_ident("deprecated") {
            let (line, col) = span_loc(attr.bracket_token.span.span());
            self.violations.push(Violation::new(
                self.file.clone(),
                line,
                col,
                "`#[deprecated]` attribute — invariant 14 forbids deprecation periods",
            ));
        }
        syn::visit::visit_attribute(self, attr);
    }

    fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
        scan_use_for_legacy_alias(&item.tree, self);
        syn::visit::visit_item_use(self, item);
    }
}

fn scan_use_for_legacy_alias(tree: &syn::UseTree, v: &mut NoShimsVisitor<'_>) {
    match tree {
        syn::UseTree::Rename(r) => {
            let name = r.rename.to_string();
            if ALIAS_TAILS.iter().any(|tail| name.ends_with(tail))
                || name.starts_with("Old")
                || name.starts_with("Legacy")
                || name.contains("_old")
                || name.contains("_legacy")
            {
                let (line, col) = span_loc(r.rename.span());
                v.violations.push(Violation::new(
                    v.file.clone(),
                    line,
                    col,
                    format!("legacy alias `as {name}` — rename in place instead"),
                ));
            }
        }
        syn::UseTree::Path(p) => scan_use_for_legacy_alias(&p.tree, v),
        syn::UseTree::Group(g) => {
            for item in &g.items {
                scan_use_for_legacy_alias(item, v);
            }
        }
        syn::UseTree::Name(_) | syn::UseTree::Glob(_) => {}
    }
}
