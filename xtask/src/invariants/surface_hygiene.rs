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
//!     with `#[source]` or `#[from]` so the diagnostic chain is preserved
//!     for `std::error::Error::source()` traversal. Scope: any `pub enum`
//!     whose ident is `Error` or ends in `Error`. For every field in
//!     every variant whose type's last segment is `Error` or ends in
//!     `Error`, require `#[source]` or `#[from]` on the field — OR
//!     `#[error(transparent)]` on the variant, which thiserror expands
//!     into source-forwarding for the inner field. `Box<X>` / `Arc<X>`
//!     / `Option<X>` wrappers are unwrapped one level so
//!     `Option<reqwest::Error>` still trips the check.

use std::path::Path;

use anyhow::Result;
use syn::visit::Visit;

use crate::visitor::{FileGate, Violation, run_invariants, span_loc};

const TIER1_CONFIG_STRUCTS: &[&str] = &["ChatModelConfig", "SessionGraph", "Usage", "GraphHop"];

/// Single-arity generic wrappers whose inner type still counts as the
/// "carried" error for `#[source]` / `#[from]` purposes. `Box<reqwest::Error>`
/// or `Option<AuthError>` is semantically the same as the bare error
/// from a diagnostic-chain standpoint.
const TRANSPARENT_WRAPPERS: &[&str] = &["Box", "Arc", "Rc", "Option"];

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

    fn visit(&self, rel_path: &Path, src: &str, ast: &syn::File, violations: &mut Vec<Violation>) {
        let mut v = HygieneVisitor {
            file: rel_path.to_path_buf(),
            src,
            violations,
        };
        v.visit_file(ast);
    }

    fn remediation(&self) -> &'static str {
        REMEDIATION
    }
}

pub(crate) fn file_gates() -> Vec<Box<dyn FileGate>> {
    vec![Box::new(SurfaceHygieneGate)]
}

pub(crate) fn run() -> Result<()> {
    run_invariants(&file_gates(), &[])
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
        if !has_non_exhaustive(&item.attrs) && !sealed_enum_marker_above(&item.ident, self.src) {
            let (line, col) = span_loc(item.ident.span());
            self.violations.push(Violation::new(
                self.file.clone(),
                line,
                col,
                format!("`pub enum {}` missing `#[non_exhaustive]`", item.ident),
            ));
        }
        if is_error_ident(&item.ident.to_string()) {
            self.check_error_variants(item);
        }
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

impl<'v, 's> HygieneVisitor<'v, 's> {
    fn check_error_variants(&mut self, item: &syn::ItemEnum) {
        let enum_name = item.ident.to_string();
        for variant in &item.variants {
            // `#[error(transparent)]` on the variant is thiserror's
            // documented spelling for "forward source to the inner
            // field". Treat the whole variant as satisfied — thiserror
            // generates the `Error::source()` plumbing.
            if variant_is_transparent(&variant.attrs) {
                continue;
            }
            for (idx, field) in variant.fields.iter().enumerate() {
                if !field_carries_error_type(&field.ty) {
                    continue;
                }
                if field_has_source_or_from(&field.attrs) {
                    continue;
                }
                let span = field.ident.as_ref().map_or_else(
                    || syn::spanned::Spanned::span(&field.ty),
                    syn::spanned::Spanned::span,
                );
                let (line, col) = span_loc(span);
                let where_ = match &field.ident {
                    Some(id) => format!("field `{id}`"),
                    None => format!("tuple field {idx}"),
                };
                self.violations.push(Violation::new(
                    self.file.clone(),
                    line,
                    col,
                    format!(
                        "`{enum_name}::{}` {where_} carries an inner error type without `#[source]` / `#[from]` / `#[error(transparent)]` — diagnostic chain breaks",
                        variant.ident
                    ),
                ));
            }
        }
    }
}

/// Detect `#[error(transparent)]` on a variant. The attr path is
/// `error` and the argument tokens are exactly the `transparent`
/// keyword — `parse_args::<syn::Ident>()` succeeds and yields that
/// identifier.
fn variant_is_transparent(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|a| {
        if !a.path().is_ident("error") {
            return false;
        }
        a.parse_args::<syn::Ident>()
            .map(|i| i == "transparent")
            .unwrap_or(false)
    })
}

fn is_error_ident(name: &str) -> bool {
    name == "Error" || name.ends_with("Error")
}

/// True when the field's type carries an error. Unwraps one level of
/// single-arity wrapper (`Box<E>`, `Arc<E>`, `Option<E>`) so a wrapped
/// error still trips the diagnostic-chain requirement.
fn field_carries_error_type(ty: &syn::Type) -> bool {
    let Some(seg) = path_last_segment(ty) else {
        return false;
    };
    let name = seg.ident.to_string();
    if is_error_ident(&name) {
        return true;
    }
    if TRANSPARENT_WRAPPERS.contains(&name.as_str())
        && let syn::PathArguments::AngleBracketed(args) = &seg.arguments
        && let Some(syn::GenericArgument::Type(inner)) = args.args.first()
    {
        return field_carries_error_type(inner);
    }
    false
}

fn path_last_segment(ty: &syn::Type) -> Option<&syn::PathSegment> {
    match ty {
        syn::Type::Path(tp) => tp.path.segments.last(),
        syn::Type::Reference(r) => path_last_segment(&r.elem),
        _ => None,
    }
}

fn field_has_source_or_from(attrs: &[syn::Attribute]) -> bool {
    attrs
        .iter()
        .any(|a| a.path().is_ident("source") || a.path().is_ident("from"))
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
