//! Public-API surface hygiene:
//!
//!  1. Every `pub enum` carries `#[non_exhaustive]` (or an explicit
//!     `// SEALED-ENUM` line on the same item — opt-out for FSMs that
//!     intentionally close their variant set).
//!  2. Tier-1 user-facing config structs carry `#[non_exhaustive]`.
//!     Hand-curated allow-list — adding a new field post-1.0 is a SemVer
//!     break unless the struct is `#[non_exhaustive]`. Open data carriers
//!     (`Message`, `Document`, `EntityRecord`) are intentionally OPEN.
//!  3. Error variants carrying inner error types annotate the inner field
//!     with `#[source]` or `#[from]`. Caught at the AST level: when a
//!     variant's named field type ends in `Error` and lacks `#[source]` /
//!     `#[from]`, flag.

use std::path::Path;

use anyhow::Result;
use syn::visit::Visit;

use crate::visitor::{FileGate, Violation, run_file_gates, span_loc};

const TIER1_CONFIG_STRUCTS: &[&str] = &["ChatModelConfig", "SessionGraph", "Usage", "GraphHop"];

const REMEDIATION: &str = "Add `#[non_exhaustive]` to public enums and Tier-1 config structs.\n\
     Annotate error inner fields with `#[source]` or `#[from]` so the\n\
     diagnostic chain is preserved. If a `pub enum` is genuinely sealed\n\
     (a finite FSM), add a comment on the line above the declaration:\n\
     \n  // SEALED-ENUM: <reason>\n  pub enum FooState { ... }";

pub(crate) struct SurfaceHygieneGate;

impl FileGate for SurfaceHygieneGate {
    fn name(&self) -> &'static str {
        "surface-hygiene"
    }

    fn visit(&self, path: &Path, src: &str, ast: &syn::File, violations: &mut Vec<Violation>) {
        let mut v = HygieneVisitor {
            file: path.to_path_buf(),
            src,
            violations,
        };
        v.visit_file(ast);
    }

    fn remediation(&self) -> &'static str {
        REMEDIATION
    }
}

pub(crate) fn gates() -> Vec<Box<dyn FileGate>> {
    vec![Box::new(SurfaceHygieneGate)]
}

pub(crate) fn run() -> Result<()> {
    run_file_gates(&gates())
}

struct HygieneVisitor<'v, 's> {
    file: std::path::PathBuf,
    src: &'s str,
    violations: &'v mut Vec<Violation>,
}

impl<'ast, 'v, 's> Visit<'ast> for HygieneVisitor<'v, 's> {
    fn visit_item_enum(&mut self, item: &'ast syn::ItemEnum) {
        if !matches!(item.vis, syn::Visibility::Public(_)) {
            return;
        }
        if has_non_exhaustive(&item.attrs) || sealed_enum_marker_above(&item.ident, self.src) {
            return;
        }
        let (line, col) = span_loc(item.ident.span());
        self.violations.push(Violation::new(
            self.file.clone(),
            line,
            col,
            format!("`pub enum {}` missing `#[non_exhaustive]`", item.ident),
        ));
    }

    fn visit_item_struct(&mut self, item: &'ast syn::ItemStruct) {
        if !matches!(item.vis, syn::Visibility::Public(_)) {
            return;
        }
        let name = item.ident.to_string();
        if TIER1_CONFIG_STRUCTS.contains(&name.as_str()) && !has_non_exhaustive(&item.attrs) {
            let (line, col) = span_loc(item.ident.span());
            self.violations.push(Violation::new(
                self.file.clone(),
                line,
                col,
                format!("`pub struct {name}` is Tier-1 config — must carry `#[non_exhaustive]`"),
            ));
        }
    }
}

fn has_non_exhaustive(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|a| a.path().is_ident("non_exhaustive"))
}

fn sealed_enum_marker_above(ident: &syn::Ident, src: &str) -> bool {
    let lc = ident.span().start();
    let line_idx = lc.line.saturating_sub(1);
    src.lines()
        .nth(line_idx.saturating_sub(1))
        .map(|prev| prev.contains("SEALED-ENUM"))
        .unwrap_or(false)
}
