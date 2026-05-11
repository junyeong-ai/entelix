//! Anthropic managed-agent shape — invariants 1, 2, 4, 10 + subagent
//! layer-inheritance. Each rule is a single AST predicate scoped to the
//! file(s) it targets via [`FileGate::applies_to`], so the share-parse
//! orchestrator dispatches the per-file `syn::File` to whichever
//! predicates apply.
//!
//! Each sub-gate carries its own distinct `name()` so the report
//! identifies which managed-shape rule fired — collapsing all five
//! under one umbrella label hides which invariant the violation
//! actually breaks.
//!
//! ## Enforcement boundary
//!
//! The gates here enforce the **structurally detectable** projections
//! of invariants 1/2/4/10. CLAUDE.md states broader claims that the
//! AST cannot mechanically prove:
//!
//!   * Invariant 1's "no message cache anywhere" — the gate enforces
//!     the SSoT shape (`SessionGraph` holds `events: Vec<GraphEvent>`)
//!     but cannot prove that no *other* struct in the workspace
//!     accidentally caches messages. A `Vec<Message>` field could be a
//!     legitimate `BufferMemory` working-memory shape or an illegitimate
//!     side-cache; distinguishing them needs semantic context the
//!     visitor doesn't carry. Reviewer-enforced.
//!   * Invariant 2's "no persistent state beyond in-memory request
//!     scope" — the gate catches `Agent` fields whose type carries
//!     `*Persistence` / `*Checkpointer` / `*SessionLog` (the three
//!     persistence-layer trait families), which covers the common
//!     regression. Subtler shapes (a custom cache type, a stray
//!     handle to a DB pool) are reviewer-enforced.
//!
//! The structural projections catch every regression we've seen in
//! practice. The reviewer-enforced clauses are documented here so
//! future-Claude knows where the gate's enforcement ends and code
//! review picks up.

use std::path::Path;

use anyhow::Result;
use syn::spanned::Spanned;
use syn::visit::Visit;

use crate::visitor::{FileGate, Violation, run_invariants, span_loc, type_carries};

const REMEDIATION: &str = "Anthropic managed-agent shape is non-negotiable. See CLAUDE.md\n\
     §\"Anthropic managed-agent shape\".";

const CORE_SRC: &str = "crates/entelix-core/src";
const SESSION_SRC: &str = "crates/entelix-session/src";
const SUBAGENT_FILE: &str = "crates/entelix-agents/src/subagent.rs";

// ── Invariant 2 — Agent must not own persistent-state fields ────────
//
// Statelessness covers every persistence-layer trait, not just the
// `*Persistence` aggregate facade: a stray `*Checkpointer` /
// `*SessionLog` reference on an `Agent` would also defeat
// crash → wake(thread_id) → resume.

pub(crate) struct NoAgentPersistenceGate;

const PERSISTENT_STATE_SUFFIXES: &[&str] = &["Persistence", "Checkpointer", "SessionLog"];

fn carries_persistent_state(ty: &syn::Type) -> bool {
    type_carries(ty, &|seg: &str| {
        PERSISTENT_STATE_SUFFIXES
            .iter()
            .any(|suffix| seg.ends_with(suffix))
    })
}

impl FileGate for NoAgentPersistenceGate {
    fn name(&self) -> &'static str {
        "managed-shape: invariant 2 (Agent is stateless)"
    }

    fn applies_to(&self, rel_path: &Path) -> bool {
        rel_path.starts_with(CORE_SRC)
    }

    fn visit(&self, rel_path: &Path, _src: &str, ast: &syn::File, violations: &mut Vec<Violation>) {
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
                    if carries_persistent_state(&field.ty) {
                        let (line, col) = span_loc(ident.span());
                        violations.push(Violation::new(
                            rel_path.to_path_buf(),
                            line,
                            col,
                            format!(
                                "`{name}.{ident}` carries a persistence-layer type (`*Persistence` / `*Checkpointer` / `*SessionLog`, directly or via `Arc` / `Box<dyn …>` / `Option`) — Harness must be stateless (invariant 2)"
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
        "managed-shape: invariant 10 (no credentials in ExecutionContext)"
    }

    fn applies_to(&self, rel_path: &Path) -> bool {
        rel_path.starts_with(CORE_SRC)
    }

    fn visit(&self, rel_path: &Path, _src: &str, ast: &syn::File, violations: &mut Vec<Violation>) {
        for item in &ast.items {
            if let syn::Item::Struct(s) = item {
                if s.ident != "ExecutionContext" {
                    continue;
                }
                for field in &s.fields {
                    if type_carries(&field.ty, &|seg: &str| {
                        seg == "CredentialProvider" || seg.ends_with("CredentialProvider")
                    }) {
                        let (line, col) = span_loc(field.ty.span());
                        violations.push(Violation::new(
                            rel_path.to_path_buf(),
                            line,
                            col,
                            "`ExecutionContext` field carries a `CredentialProvider` (directly or via `Arc` / `Box<dyn …>` / `Option`) — tokens must never reach Tool input (invariant 10)",
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
        "managed-shape: invariant 1 (SessionGraph events SSoT)"
    }

    fn applies_to(&self, rel_path: &Path) -> bool {
        rel_path.starts_with(SESSION_SRC)
    }

    fn visit(&self, rel_path: &Path, _src: &str, ast: &syn::File, violations: &mut Vec<Violation>) {
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
                        rel_path.to_path_buf(),
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
        "managed-shape: invariant 4 (Tool::execute hand contract)"
    }

    fn applies_to(&self, rel_path: &Path) -> bool {
        rel_path.starts_with(CORE_SRC)
    }

    fn visit(&self, rel_path: &Path, _src: &str, ast: &syn::File, violations: &mut Vec<Violation>) {
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
                        rel_path.to_path_buf(),
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
        "managed-shape: subagent layer inheritance"
    }

    fn applies_to(&self, rel_path: &Path) -> bool {
        rel_path == Path::new(SUBAGENT_FILE)
    }

    fn visit(&self, rel_path: &Path, _src: &str, ast: &syn::File, violations: &mut Vec<Violation>) {
        let mut v = ToolRegistryNewVisitor {
            file: rel_path.to_path_buf(),
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

pub(crate) fn file_gates() -> Vec<Box<dyn FileGate>> {
    vec![
        Box::new(NoAgentPersistenceGate),
        Box::new(NoCredentialInCtxGate),
        Box::new(SessionEventFieldGate),
        Box::new(ToolExecuteGate),
        Box::new(SubagentLayerInheritanceGate),
    ]
}

pub(crate) fn run() -> Result<()> {
    run_invariants(&file_gates(), &[])
}
