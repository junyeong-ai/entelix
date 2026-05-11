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

use std::path::Path;

use anyhow::Result;
use syn::spanned::Spanned;
use syn::visit::Visit;

use crate::visitor::{FileGate, Violation, run_invariants, span_loc};

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

const REMEDIATION: &str = "Delete the old name / type / signature in the same PR. No\n\
     `#[deprecated]`, no `pub use OldName as NewName`, no `// formerly`\n\
     comments. If a flagged comment is genuinely descriptive prose\n\
     (not a backcompat marker), rephrase to avoid the trigger words.";

pub(crate) struct NoShimsGate;

impl FileGate for NoShimsGate {
    fn name(&self) -> &'static str {
        "no-shims (invariant 14)"
    }

    fn visit(&self, rel_path: &Path, src: &str, ast: &syn::File, violations: &mut Vec<Violation>) {
        let mut v = NoShimsVisitor {
            file: rel_path.to_path_buf(),
            violations,
        };
        v.visit_file(ast);

        // Comment scan — line by line, skip doc comments.
        for (idx, raw) in src.lines().enumerate() {
            let trimmed = raw.trim_start();
            if trimmed.starts_with("///") || trimmed.starts_with("//!") {
                continue;
            }
            for pat in COMMENT_PATTERNS {
                if trimmed.contains(pat) {
                    v.violations.push(Violation::new(
                        rel_path.to_path_buf(),
                        idx + 1,
                        raw.find(pat).unwrap_or(0) + 1,
                        format!("legacy comment marker — `{pat}`"),
                    ));
                    break;
                }
            }
        }
    }

    fn remediation(&self) -> &'static str {
        REMEDIATION
    }
}

pub(crate) fn file_gates() -> Vec<Box<dyn FileGate>> {
    vec![Box::new(NoShimsGate)]
}

pub(crate) fn run() -> Result<()> {
    run_invariants(&file_gates(), &[])
}

struct NoShimsVisitor<'v> {
    file: std::path::PathBuf,
    violations: &'v mut Vec<Violation>,
}

impl<'ast, 'v> Visit<'ast> for NoShimsVisitor<'v> {
    fn visit_attribute(&mut self, attr: &'ast syn::Attribute) {
        let path = attr.path();
        if path.is_ident("deprecated") {
            let (line, col) = span_loc(attr.bracket_token.span.span());
            self.violations.push(Violation::new(
                self.file.clone(),
                line,
                col,
                "`#[deprecated]` attribute — invariant 14 forbids deprecation periods",
            ));
        } else if path.is_ident("cfg_attr") && cfg_attr_contains_deprecated(attr) {
            let (line, col) = span_loc(attr.bracket_token.span.span());
            self.violations.push(Violation::new(
                self.file.clone(),
                line,
                col,
                "`#[cfg_attr(_, deprecated)]` — invariant 14 forbids deprecation periods even under cfg gating",
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
            if is_legacy_alias_name(&name) {
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

/// True when an alias name is a backcompat shim. Recognises the `Old`
/// / `Legacy` family **and** versioned trailers (`FooV1`, `FooV2`,
/// `foo_v1`) — both shapes are the canonical "I renamed the new
/// thing, here's a redirect for the old name" pattern.
fn is_legacy_alias_name(name: &str) -> bool {
    if ALIAS_TAILS.iter().any(|tail| name.ends_with(tail))
        || name.starts_with("Old")
        || name.starts_with("Legacy")
        || name.contains("_old")
        || name.contains("_legacy")
    {
        return true;
    }
    has_versioned_suffix(name)
}

/// True when `name` ends in `V<digit>+` (PascalCase, `FooV1`) or
/// `_v<digit>+` (snake_case, `foo_v1`). The version digits are at
/// least one character — `V` alone is not enough.
fn has_versioned_suffix(name: &str) -> bool {
    fn ends_with_digit_run(slice: &str) -> bool {
        let mut chars = slice.chars().rev();
        let mut saw_digit = false;
        for c in chars.by_ref() {
            if c.is_ascii_digit() {
                saw_digit = true;
            } else {
                return saw_digit && (c == 'V' || c == 'v');
            }
        }
        false
    }
    ends_with_digit_run(name)
        && (name.contains("_v") || name.chars().rev().find(|c| !c.is_ascii_digit()) == Some('V'))
}

/// True when `attr` is `#[cfg_attr(_, deprecated, ...)]` — any inner
/// meta whose path is `deprecated` triggers the gate.
fn cfg_attr_contains_deprecated(attr: &syn::Attribute) -> bool {
    let Ok(list) = attr.meta.require_list() else {
        return false;
    };
    list.tokens.clone().into_iter().any(|tt| match tt {
        proc_macro2::TokenTree::Ident(i) => i == "deprecated",
        _ => false,
    })
}
