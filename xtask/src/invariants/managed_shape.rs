//! Anthropic managed-agent shape — invariants 1, 2, 4, 10 +
//! Each rule is a single AST predicate, executed only against the file it
//! targets.

use anyhow::Result;
use syn::spanned::Spanned;
use syn::visit::Visit;

use crate::visitor::{Violation, parse, repo_root, report, span_loc};

pub(crate) fn run() -> Result<()> {
    let root = repo_root()?;
    let mut violations = Vec::new();

    // ── Invariant 2 — Agent must not own a `Persistence` field ──
    inv2_no_agent_persistence(&root, &mut violations)?;
    // ── Invariant 10 — ExecutionContext must not embed CredentialProvider ──
    inv10_no_credential_in_ctx(&root, &mut violations)?;
    // ── Invariant 1 — SessionGraph must hold `events: Vec<GraphEvent>` ──
    inv1_session_event_field(&root, &mut violations)?;
    // ── Invariant 4 — Tool trait must expose `execute` ──
    inv4_tool_execute(&root, &mut violations)?;
    // ── — Subagent must not construct fresh ToolRegistry ──
    adr35_subagent_layer_inheritance(&root, &mut violations)?;

    report(
        "managed-shape (invariants 1, 2, 4, 10)",
        violations,
        "Anthropic managed-agent shape is non-negotiable. See CLAUDE.md\n\
         §\"Anthropic managed-agent shape\".",
    )
}

fn inv2_no_agent_persistence(
    root: &std::path::Path,
    violations: &mut Vec<Violation>,
) -> Result<()> {
    // Walk every entelix-core source file. Flag any field whose type ends
    // in `*Persistence` directly on a struct named `Agent` or `*Agent`.
    let dir = root.join("crates/entelix-core/src");
    if !dir.exists() {
        return Ok(());
    }
    for entry in walkdir::WalkDir::new(&dir)
        .into_iter()
        .filter_map(std::result::Result::ok)
    {
        let p = entry.path();
        if !p.is_file() || p.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        let (_, ast) = parse(p)?;
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
                            p.to_path_buf(),
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
    Ok(())
}

fn inv10_no_credential_in_ctx(
    root: &std::path::Path,
    violations: &mut Vec<Violation>,
) -> Result<()> {
    let dir = root.join("crates/entelix-core/src");
    if !dir.exists() {
        return Ok(());
    }
    for entry in walkdir::WalkDir::new(&dir)
        .into_iter()
        .filter_map(std::result::Result::ok)
    {
        let p = entry.path();
        if !p.is_file() || p.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        let (_, ast) = parse(p)?;
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
                            p.to_path_buf(),
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
    Ok(())
}

fn inv1_session_event_field(root: &std::path::Path, violations: &mut Vec<Violation>) -> Result<()> {
    let dir = root.join("crates/entelix-session/src");
    if !dir.exists() {
        return Ok(());
    }
    let mut session_struct: Option<(std::path::PathBuf, proc_macro2::Span, bool)> = None;
    for entry in walkdir::WalkDir::new(&dir)
        .into_iter()
        .filter_map(std::result::Result::ok)
    {
        let p = entry.path();
        if !p.is_file() || p.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        let (_, ast) = parse(p)?;
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
                        // Expect Vec<GraphEvent>.
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
                session_struct = Some((p.to_path_buf(), s.ident.span(), has_events_vec_graphevent));
            }
        }
    }
    if let Some((path, span, ok)) = session_struct {
        if !ok {
            let (line, col) = span_loc(span);
            violations.push(Violation::new(
                path,
                line,
                col,
                "`SessionGraph` must hold `events: Vec<GraphEvent>` (invariant 1 — event SSoT)",
            ));
        }
    }
    Ok(())
}

fn inv4_tool_execute(root: &std::path::Path, violations: &mut Vec<Violation>) -> Result<()> {
    let dir = root.join("crates/entelix-core/src");
    if !dir.exists() {
        return Ok(());
    }
    for entry in walkdir::WalkDir::new(&dir)
        .into_iter()
        .filter_map(std::result::Result::ok)
    {
        let p = entry.path();
        if !p.is_file() || p.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        let (_, ast) = parse(p)?;
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
                        p.to_path_buf(),
                        line,
                        col,
                        "`Tool` trait missing `execute` method (invariant 4)",
                    ));
                }
            }
        }
    }
    Ok(())
}

fn adr35_subagent_layer_inheritance(
    root: &std::path::Path,
    violations: &mut Vec<Violation>,
) -> Result<()> {
    let path = root.join("crates/entelix-agents/src/subagent.rs");
    if !path.exists() {
        return Ok(());
    }
    let (src, ast) = parse(&path)?;
    let mut v = ToolRegistryNewVisitor {
        file: path.clone(),
        violations,
    };
    v.visit_file(&ast);
    let _ = src;
    Ok(())
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
