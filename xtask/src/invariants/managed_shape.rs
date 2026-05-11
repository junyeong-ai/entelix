//! Anthropic managed-agent shape — invariants 1, 2, 4, 10 +
//! Each rule is a single AST predicate, executed only against the file it
//! targets. Each predicate is its own [`FileGate`] with a tight
//! `applies_to` scope so the share-parse orchestrator dispatches the
//! per-file `syn::File` to whichever predicates target that zone.

use std::path::Path;

use anyhow::Result;
use syn::spanned::Spanned;
use syn::visit::Visit;

use crate::visitor::{FileGate, Violation, run_file_gates, span_loc};

const REMEDIATION: &str = "Anthropic managed-agent shape is non-negotiable. See CLAUDE.md\n\
     §\"Anthropic managed-agent shape\".";

const CORE_SRC: &str = "crates/entelix-core/src";
const SESSION_SRC: &str = "crates/entelix-session/src";
const SUBAGENT_FILE: &str = "crates/entelix-agents/src/subagent.rs";

fn under(path: &Path, zone: &str) -> bool {
    path.to_string_lossy().contains(zone)
}

// ── Invariant 2 — Agent must not own a `Persistence` field ──────────

pub(crate) struct NoAgentPersistenceGate;

impl FileGate for NoAgentPersistenceGate {
    fn name(&self) -> &'static str {
        "managed-shape (invariants 1, 2, 4, 10)"
    }

    fn applies_to(&self, path: &Path) -> bool {
        under(path, CORE_SRC)
    }

    fn visit(&self, path: &Path, _src: &str, ast: &syn::File, violations: &mut Vec<Violation>) {
        for item in &ast.items {
            if let syn::Item::Struct(s) = item {
                let name = s.ident.to_string();
                if !(name == "Agent" || name.ends_with("Agent")) {
                    continue;
                }
                for field in &s.fields {
                    let Some(ident) = &field.ident else {
                        continue;
                    };
                    let ty = type_last_segment(&field.ty).unwrap_or_default();
                    if ty.ends_with("Persistence") {
                        let (line, col) = span_loc(ident.span());
                        violations.push(Violation::new(
                            path.to_path_buf(),
                            line,
                            col,
                            format!(
                                "`{name}.{}: {ty}` — Harness must be stateless (invariant 2)",
                                ident
                            ),
                        ));
                    }
                }
            }
        }
    }

    fn remediation(&self) -> &'static str {
        REMEDIATION
    }
}

// ── Invariant 10 — ExecutionContext must not embed CredentialProvider ──

pub(crate) struct NoCredentialInCtxGate;

impl FileGate for NoCredentialInCtxGate {
    fn name(&self) -> &'static str {
        "managed-shape (invariants 1, 2, 4, 10)"
    }

    fn applies_to(&self, path: &Path) -> bool {
        under(path, CORE_SRC)
    }

    fn visit(&self, path: &Path, _src: &str, ast: &syn::File, violations: &mut Vec<Violation>) {
        for item in &ast.items {
            if let syn::Item::Struct(s) = item {
                if s.ident != "ExecutionContext" {
                    continue;
                }
                for field in &s.fields {
                    let ty = type_last_segment(&field.ty).unwrap_or_default();
                    if ty == "CredentialProvider" || ty.ends_with("CredentialProvider") {
                        let (line, col) = span_loc(field.ty.span());
                        violations.push(Violation::new(
                            path.to_path_buf(),
                            line,
                            col,
                            format!(
                                "`ExecutionContext` embeds `{ty}` — tokens must never reach Tool input (invariant 10)"
                            ),
                        ));
                    }
                }
            }
        }
    }

    fn remediation(&self) -> &'static str {
        REMEDIATION
    }
}

// ── Invariant 1 — SessionGraph must hold `events: Vec<GraphEvent>` ──

pub(crate) struct SessionEventFieldGate;

impl FileGate for SessionEventFieldGate {
    fn name(&self) -> &'static str {
        "managed-shape (invariants 1, 2, 4, 10)"
    }

    fn applies_to(&self, path: &Path) -> bool {
        under(path, SESSION_SRC)
    }

    fn visit(&self, path: &Path, _src: &str, ast: &syn::File, violations: &mut Vec<Violation>) {
        for item in &ast.items {
            if let syn::Item::Struct(s) = item {
                if s.ident != "SessionGraph" {
                    continue;
                }
                let mut has_events_vec_graphevent = false;
                for field in &s.fields {
                    let Some(ident) = &field.ident else {
                        continue;
                    };
                    if ident == "events" {
                        if let syn::Type::Path(tp) = &field.ty {
                            if let Some(last) = tp.path.segments.last() {
                                if last.ident == "Vec" {
                                    if let syn::PathArguments::AngleBracketed(a) = &last.arguments {
                                        if let Some(syn::GenericArgument::Type(inner)) =
                                            a.args.first()
                                        {
                                            if type_last_segment(inner).as_deref()
                                                == Some("GraphEvent")
                                            {
                                                has_events_vec_graphevent = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if !has_events_vec_graphevent {
                    let (line, col) = span_loc(s.ident.span());
                    violations.push(Violation::new(
                        path.to_path_buf(),
                        line,
                        col,
                        "`SessionGraph` must hold `events: Vec<GraphEvent>` (invariant 1 — event SSoT)",
                    ));
                }
            }
        }
    }

    fn remediation(&self) -> &'static str {
        REMEDIATION
    }
}

// ── Invariant 4 — Tool trait must expose `execute` ──

pub(crate) struct ToolExecuteGate;

impl FileGate for ToolExecuteGate {
    fn name(&self) -> &'static str {
        "managed-shape (invariants 1, 2, 4, 10)"
    }

    fn applies_to(&self, path: &Path) -> bool {
        under(path, CORE_SRC)
    }

    fn visit(&self, path: &Path, _src: &str, ast: &syn::File, violations: &mut Vec<Violation>) {
        for item in &ast.items {
            if let syn::Item::Trait(t) = item {
                if t.ident != "Tool" {
                    continue;
                }
                let has_execute = t.items.iter().any(|i| {
                    if let syn::TraitItem::Fn(f) = i {
                        f.sig.ident == "execute"
                    } else {
                        false
                    }
                });
                if !has_execute {
                    let (line, col) = span_loc(t.ident.span());
                    violations.push(Violation::new(
                        path.to_path_buf(),
                        line,
                        col,
                        "`Tool` trait missing `execute` method (invariant 4)",
                    ));
                }
            }
        }
    }

    fn remediation(&self) -> &'static str {
        REMEDIATION
    }
}

// ── Subagent must not construct fresh ToolRegistry ──

pub(crate) struct SubagentLayerInheritanceGate;

impl FileGate for SubagentLayerInheritanceGate {
    fn name(&self) -> &'static str {
        "managed-shape (invariants 1, 2, 4, 10)"
    }

    fn applies_to(&self, path: &Path) -> bool {
        under(path, SUBAGENT_FILE)
    }

    fn visit(&self, path: &Path, _src: &str, ast: &syn::File, violations: &mut Vec<Violation>) {
        let mut v = ToolRegistryNewVisitor {
            file: path.to_path_buf(),
            violations,
        };
        v.visit_file(ast);
    }

    fn remediation(&self) -> &'static str {
        REMEDIATION
    }
}

struct ToolRegistryNewVisitor<'v> {
    file: std::path::PathBuf,
    violations: &'v mut Vec<Violation>,
}
impl<'ast, 'v> Visit<'ast> for ToolRegistryNewVisitor<'v> {
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if let syn::Expr::Path(p) = &*call.func {
            let segs: Vec<String> = p
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect();
            if segs.len() >= 2
                && segs[segs.len() - 2] == "ToolRegistry"
                && segs[segs.len() - 1] == "new"
            {
                let (line, col) = span_loc(p.path.segments.first().unwrap().ident.span());
                self.violations.push(Violation::new(
                    self.file.clone(),
                    line,
                    col,
                    "`ToolRegistry::new` in subagent.rs — drops parent layer stack. \
                     Use parent_registry.with_only(allowed) or .filter(predicate)",
                ));
            }
        }
        syn::visit::visit_expr_call(self, call);
    }
}

fn type_last_segment(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(tp) => tp.path.segments.last().map(|s| s.ident.to_string()),
        syn::Type::Reference(r) => type_last_segment(&r.elem),
        _ => None,
    }
}

pub(crate) fn gates() -> Vec<Box<dyn FileGate>> {
    vec![
        Box::new(NoAgentPersistenceGate),
        Box::new(NoCredentialInCtxGate),
        Box::new(SessionEventFieldGate),
        Box::new(ToolExecuteGate),
        Box::new(SubagentLayerInheritanceGate),
    ]
}

pub(crate) fn run() -> Result<()> {
    run_file_gates(&gates())
}
