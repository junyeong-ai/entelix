//! naming taxonomy enforcement. Five rules:
//!
//!  1. **Forbidden suffixes** on `pub struct/enum/trait`:
//!     `*Engine` / `*Wrapper` / `*Handler` / `*Helper` / `*Util`.
//!  2. **`*Service`** is reserved for types that directly impl
//!     `tower::Service<R>`. Other names use `*Manager` / `*Client` / etc.
//!  3. **No `get_*`** accessors — use the bare name. Catches `pub fn`,
//!     `pub async fn`, **and** trait-default methods (the prior regex
//!     missed `pub async fn` and trait-default `async fn`).
//!  4. **Builder verb-prefix** — every `pub fn` returning `Self` /
//!     `Result<Self>` inside `impl <X>Builder` must start with
//!     `with_` / `add_` / `set_` / `register`. Value-level combinators
//!     (`Annotated::reduced`, etc.) are not in scope because they live
//!     outside `impl <X>Builder`.
//!  5. **`with_*(&self, …)`** — borrow disguised as builder. Bare
//!     accessor (`region(&self)`) for borrows; domain verb
//!     (`restricted_to(&self, …)`) for derivative views.
//!  6. **ctx parameter ordering** — split convention by trait family.

use std::collections::BTreeSet;

use anyhow::Result;
use syn::spanned::Spanned;
use syn::visit::Visit;

use crate::visitor::{Violation, parse, read, repo_root, report, rust_source_files, span_loc};
// `read` is used by check_ctx_in_trait_files via the helper closure below; keep
// alongside the rest of the visitor utilities.

const FORBIDDEN_SUFFIXES: &[&str] = &["Engine", "Wrapper", "Handler", "Helper", "Util"];

/// ctx-first traits — memory / persistence backends. Every method that
/// takes `&ExecutionContext` must place it as the first non-self argument.
const CTX_FIRST_TRAITS: &[&str] = &[
    "Store",
    "VectorStore",
    "GraphMemory",
    "Checkpointer",
    "SessionLog",
];

/// ctx-last traits — computation / dispatch. Every method that takes
/// `&ExecutionContext` must place it as the last argument.
const CTX_LAST_TRAITS: &[&str] = &[
    "Tool",
    "Embedder",
    "Retriever",
    "Reranker",
    "Runnable",
    "AgentObserver",
    "Approver",
    "Sandbox",
    "CostCalculator",
    "ToolCostCalculator",
    "EmbeddingCostCalculator",
    "Summarizer",
];

/// Memory pattern files where every method taking `&ExecutionContext` is
/// ctx-first by family rule (BufferMemory / SummaryMemory / EntityMemory /
/// EpisodicMemory / SemanticMemory / ConsolidatingBufferMemory).
const MEMORY_PATTERN_FILES: &[&str] = &[
    "crates/entelix-memory/src/buffer.rs",
    "crates/entelix-memory/src/summary.rs",
    "crates/entelix-memory/src/entity.rs",
    "crates/entelix-memory/src/episodic.rs",
    "crates/entelix-memory/src/semantic.rs",
    "crates/entelix-memory/src/consolidating.rs",
];

pub(crate) fn run() -> Result<()> {
    let root = repo_root()?;
    let files = rust_source_files(&root);
    let mut violations = Vec::new();

    for file in &files {
        let (src, ast) = parse(file)?;
        let mut v = NamingVisitor {
            file: file.clone(),
            file_uses_tower_service: file_imports_tower_service(&src),
            in_builder_impl_depth: 0,
            current_trait: None,
            violations: &mut violations,
        };
        v.visit_file(&ast);
    }

    // ctx ordering — explicit per-trait pass.
    for trait_name in CTX_FIRST_TRAITS {
        check_ctx_in_trait_files(&root, trait_name, CtxExpect::First, &mut violations)?;
    }
    for trait_name in CTX_LAST_TRAITS {
        check_ctx_in_trait_files(&root, trait_name, CtxExpect::Last, &mut violations)?;
    }
    // memory pattern files — every ctx-bearing method ctx-first, **except**
    // methods that belong to a ctx-last trait whose impl/definition is also
    // hosted in this file (e.g. `Summarizer` lives in `consolidating.rs` and
    // is ctx-last by trait family). Those are governed by their explicit
    // CTX_LAST_TRAITS pass and the file-level pass must not re-flag them.
    for rel in MEMORY_PATTERN_FILES {
        let path = root.join(rel);
        if !path.exists() {
            continue;
        }
        let (_, ast) = parse(&path)?;
        let mut v = CtxFileVisitor {
            file: path.clone(),
            expected: CtxExpect::First,
            ctx_last_trait_method_names: collect_ctx_last_trait_methods(&ast),
            current_ctx_last_trait: None,
            in_trait_or_impl: 0,
            violations: &mut violations,
        };
        v.visit_file(&ast);
    }

    report(
        "naming",
        violations,
        "Replacement guide:\n\
         *Engine    → say what it does (OrchestrationLoop, RetryStrategy)\n\
         *Wrapper   → say what it wraps (ToolToRunnableAdapter)\n\
         *Handler   → say what it handles (RequestProcessor, EventConsumer)\n\
         *Helper / *Util → fold into module\n\
         *Service   → *Manager (lifecycle) / *Client (HTTP)\n\
         get_X()    → X()\n\
         with_X(&self, …) → X(&self) bare accessor or restricted_to / filter / etc.\n\
         builder method → with_<noun> / add_<element> / set_<role> / register\n\
         ctx-first traits — memory/persistence: (&self, ctx, scope, payload, …)\n\
         ctx-last traits — computation/dispatch: (&self, input, …, ctx)",
    )
}

fn file_imports_tower_service(src: &str) -> bool {
    // Cheap text test — full AST visit happens elsewhere. tower::Service is
    // the universal middleware contract; whichever crate genuinely
    // implements it imports the trait. False positives here would only
    // **silence** a *Service-suffix violation, never fabricate one.
    src.contains("use tower::Service")
        || src.contains("tower::Service<")
        || src.contains("impl Service<")
        || src.contains(" Service<") && src.contains("tower")
}

struct NamingVisitor<'v> {
    file: std::path::PathBuf,
    file_uses_tower_service: bool,
    in_builder_impl_depth: usize,
    current_trait: Option<String>,
    violations: &'v mut Vec<Violation>,
}

impl<'v> NamingVisitor<'v> {
    fn check_pub_type_name(&mut self, vis: &syn::Visibility, ident: &syn::Ident) {
        if !matches!(vis, syn::Visibility::Public(_)) {
            return;
        }
        let name = ident.to_string();
        for suffix in FORBIDDEN_SUFFIXES {
            if name.ends_with(suffix) && name.len() > suffix.len() {
                let (line, col) = span_loc(ident.span());
                self.violations.push(Violation::new(
                    self.file.clone(),
                    line,
                    col,
                    format!("`{name}` — forbidden suffix `*{suffix}`"),
                ));
                return;
            }
        }
        if name.ends_with("Service") && !self.file_uses_tower_service {
            let (line, col) = span_loc(ident.span());
            self.violations.push(Violation::new(
                self.file.clone(),
                line,
                col,
                format!(
                    "`{name}` — `*Service` reserved for tower::Service impls; use *Manager / *Client"
                ),
            ));
        }
    }
}

impl<'ast, 'v> Visit<'ast> for NamingVisitor<'v> {
    fn visit_item_struct(&mut self, item: &'ast syn::ItemStruct) {
        self.check_pub_type_name(&item.vis, &item.ident);
        syn::visit::visit_item_struct(self, item);
    }
    fn visit_item_enum(&mut self, item: &'ast syn::ItemEnum) {
        self.check_pub_type_name(&item.vis, &item.ident);
        syn::visit::visit_item_enum(self, item);
    }
    fn visit_item_trait(&mut self, item: &'ast syn::ItemTrait) {
        self.check_pub_type_name(&item.vis, &item.ident);
        // Recurse into trait body with current_trait set so trait-default
        // methods see the get_* / with_*(&self) / ctx rules.
        let prev = self.current_trait.replace(item.ident.to_string());
        for trait_item in &item.items {
            self.visit_trait_item(trait_item);
        }
        self.current_trait = prev;
    }
    fn visit_item_impl(&mut self, item: &'ast syn::ItemImpl) {
        // `impl <X>Builder` — rest of the methods inside go through builder
        // verb-prefix enforcement.
        let mut is_builder_impl = false;
        if item.trait_.is_none() {
            if let syn::Type::Path(tp) = &*item.self_ty {
                if let Some(last) = tp.path.segments.last() {
                    let name = last.ident.to_string();
                    if name.ends_with("Builder") {
                        is_builder_impl = true;
                    }
                }
            }
        }
        if is_builder_impl {
            self.in_builder_impl_depth += 1;
        }
        syn::visit::visit_item_impl(self, item);
        if is_builder_impl {
            self.in_builder_impl_depth -= 1;
        }
    }
    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        check_method(self, &item.vis, &item.sig, item.span());
        syn::visit::visit_impl_item_fn(self, item);
    }
    fn visit_trait_item_fn(&mut self, item: &'ast syn::TraitItemFn) {
        // Trait methods: visibility is implicitly public; treat as `pub`.
        let pub_vis = syn::Visibility::Public(syn::token::Pub::default());
        check_method(self, &pub_vis, &item.sig, item.span());
        syn::visit::visit_trait_item_fn(self, item);
    }
    fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
        // Free fns: only check forbidden type suffix has nothing to do here;
        // get_*-on-self is impossible without a receiver, so skip get_*
        // detection — but builder verb-prefix is still impl-scoped.
        syn::visit::visit_item_fn(self, item);
    }
}

fn check_method(
    v: &mut NamingVisitor<'_>,
    vis: &syn::Visibility,
    sig: &syn::Signature,
    _full_span: proc_macro2::Span,
) {
    let is_pub_or_trait = matches!(vis, syn::Visibility::Public(_)) || v.current_trait.is_some();
    if !is_pub_or_trait {
        return;
    }
    let name = sig.ident.to_string();
    let (line, col) = span_loc(sig.ident.span());

    // ── Rule 3: `get_*` accessors with self receiver ──
    //
    // Forbidden: `fn get_name(&self) -> &str` — bare field accessor
    // shape, must use the bare name (`name`) per naming taxonomy.
    //
    // Allowed: `fn get_node(&self, ctx, ns, id) -> Result<Option<N>>`
    // — persistence-read verb-family per `.claude/rules/naming.md`.
    // Parameterized lookups returning `Option` are not field
    // accessors; the `get_*` prefix is the canonical persistence
    // read shape.
    if name.starts_with("get_") && has_self_receiver(sig) && sig.inputs.len() == 1 {
        v.violations.push(Violation::new(
            v.file.clone(),
            line,
            col,
            format!(
                "`{name}` — forbids `get_*` accessors; use the bare name (`{}`)",
                &name[4..]
            ),
        ));
    }

    // ── Rule 5: `with_*(&self, …)` masquerade ──
    if name.starts_with("with_") && has_borrow_receiver(sig) {
        v.violations.push(Violation::new(
            v.file.clone(),
            line,
            col,
            format!(
                "`{name}(&self, …)` — borrow disguised as builder; use a bare accessor or domain verb"
            ),
        ));
    }

    // ── Rule 4: builder verb-prefix on `impl <X>Builder` methods ──
    if v.in_builder_impl_depth > 0
        && returns_self_or_result_self(sig)
        && consumes_self(sig)
        && !is_builder_verb_prefix(&name)
    {
        v.violations.push(Violation::new(
            v.file.clone(),
            line,
            col,
            format!(
                "builder method `{name}` — must start with `with_` / `add_` / `set_` / `register`"
            ),
        ));
    }
}

fn is_builder_verb_prefix(name: &str) -> bool {
    name.starts_with("with_")
        || name.starts_with("add_")
        || name.starts_with("set_")
        || name == "register"
        || name.starts_with("register_")
        || name == "build"
        || name == "new"
        || name == "default"
        // Narrowing / selection verbs — analogous to
        // `ToolRegistry::restricted_to` / `Iterator::filter`. They
        // describe *which subset* the builder selects, not a
        // configuration value, so the `with_*` prefix would read
        // worse (`with_restriction` / `with_predicate` are vague).
        // The convention is documented in
        || name == "restrict_to"
        || name == "filter"
}

fn has_self_receiver(sig: &syn::Signature) -> bool {
    sig.inputs
        .iter()
        .next()
        .map(|a| matches!(a, syn::FnArg::Receiver(_)))
        .unwrap_or(false)
}

fn has_borrow_receiver(sig: &syn::Signature) -> bool {
    if let Some(syn::FnArg::Receiver(r)) = sig.inputs.iter().next() {
        r.reference.is_some()
    } else {
        false
    }
}

fn consumes_self(sig: &syn::Signature) -> bool {
    if let Some(syn::FnArg::Receiver(r)) = sig.inputs.iter().next() {
        r.reference.is_none()
    } else {
        false
    }
}

fn returns_self_or_result_self(sig: &syn::Signature) -> bool {
    let syn::ReturnType::Type(_, ty) = &sig.output else {
        return false;
    };
    let ty = &**ty;
    if is_self_type(ty) {
        return true;
    }
    if let syn::Type::Path(tp) = ty {
        if let Some(last) = tp.path.segments.last() {
            if last.ident == "Result" {
                if let syn::PathArguments::AngleBracketed(args) = &last.arguments {
                    if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                        return is_self_type(inner);
                    }
                }
            }
        }
    }
    false
}

fn is_self_type(ty: &syn::Type) -> bool {
    if let syn::Type::Path(tp) = ty {
        if let Some(last) = tp.path.segments.last() {
            return last.ident == "Self";
        }
    }
    false
}

#[derive(Clone, Copy)]
enum CtxExpect {
    First,
    Last,
}

/// Find the trait declaration in any source file, then walk its methods.
fn check_ctx_in_trait_files(
    root: &std::path::Path,
    trait_name: &str,
    expected: CtxExpect,
    violations: &mut Vec<Violation>,
) -> Result<()> {
    // Search every `crates/*/src` for `pub trait <name>`.
    let candidates = rust_source_files_under_src(root);
    for file in &candidates {
        let src = read(file)?;
        if !src.contains(&format!("trait {trait_name}")) {
            continue;
        }
        let ast: syn::File = match syn::parse_file(&src) {
            Ok(a) => a,
            Err(_) => continue,
        };
        let mut v = TraitCtxVisitor {
            file: file.clone(),
            target: trait_name,
            expected,
            seen: BTreeSet::new(),
            violations,
        };
        v.visit_file(&ast);
    }
    Ok(())
}

fn rust_source_files_under_src(root: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    for entry in walkdir::WalkDir::new(root.join("crates"))
        .into_iter()
        .filter_map(std::result::Result::ok)
    {
        let p = entry.path();
        if !p.is_file() || p.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        if p.components().any(|c| {
            let s = c.as_os_str().to_string_lossy();
            s == "target" || s == "tests"
        }) {
            continue;
        }
        out.push(p.to_path_buf());
    }
    out
}

struct TraitCtxVisitor<'v> {
    file: std::path::PathBuf,
    target: &'v str,
    expected: CtxExpect,
    seen: BTreeSet<String>,
    violations: &'v mut Vec<Violation>,
}

impl<'ast, 'v> Visit<'ast> for TraitCtxVisitor<'v> {
    fn visit_item_trait(&mut self, item: &'ast syn::ItemTrait) {
        if item.ident == self.target {
            for trait_item in &item.items {
                if let syn::TraitItem::Fn(f) = trait_item {
                    let key = format!("{}::{}", self.target, f.sig.ident);
                    if !self.seen.insert(key) {
                        continue;
                    }
                    if let Some(violation) = check_ctx_position(&self.file, &f.sig, self.expected) {
                        self.violations.push(violation);
                    }
                }
            }
        }
        syn::visit::visit_item_trait(self, item);
    }
}

struct CtxFileVisitor<'v> {
    file: std::path::PathBuf,
    expected: CtxExpect,
    /// Method names defined inside a trait declaration that's listed in
    /// CTX_LAST_TRAITS. Used to skip those methods during the file-level
    /// ctx-first sweep — they're already covered by their trait pass.
    ctx_last_trait_method_names: BTreeSet<String>,
    /// True while walking inside a trait declaration or `impl Trait for T`
    /// block whose trait is in CTX_LAST_TRAITS — every method inside is
    /// ctx-last by trait family, not ctx-first.
    current_ctx_last_trait: Option<String>,
    in_trait_or_impl: usize,
    violations: &'v mut Vec<Violation>,
}

impl<'ast, 'v> Visit<'ast> for CtxFileVisitor<'v> {
    fn visit_item_impl(&mut self, item: &'ast syn::ItemImpl) {
        self.in_trait_or_impl += 1;
        let prev = self.current_ctx_last_trait.take();
        if let Some((_, path, _)) = &item.trait_ {
            if let Some(last) = path.segments.last() {
                let name = last.ident.to_string();
                if CTX_LAST_TRAITS.contains(&name.as_str()) {
                    self.current_ctx_last_trait = Some(name);
                }
            }
        }
        syn::visit::visit_item_impl(self, item);
        self.current_ctx_last_trait = prev;
        self.in_trait_or_impl -= 1;
    }
    fn visit_item_trait(&mut self, item: &'ast syn::ItemTrait) {
        self.in_trait_or_impl += 1;
        let prev = self.current_ctx_last_trait.take();
        if CTX_LAST_TRAITS.contains(&item.ident.to_string().as_str()) {
            self.current_ctx_last_trait = Some(item.ident.to_string());
        }
        syn::visit::visit_item_trait(self, item);
        self.current_ctx_last_trait = prev;
        self.in_trait_or_impl -= 1;
    }
    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        if self.should_check(&item.sig.ident)
            && let Some(violation) = check_ctx_position(&self.file, &item.sig, self.expected)
        {
            self.violations.push(violation);
        }
        syn::visit::visit_impl_item_fn(self, item);
    }
    fn visit_trait_item_fn(&mut self, item: &'ast syn::TraitItemFn) {
        if self.should_check(&item.sig.ident)
            && let Some(violation) = check_ctx_position(&self.file, &item.sig, self.expected)
        {
            self.violations.push(violation);
        }
        syn::visit::visit_trait_item_fn(self, item);
    }
}

impl<'v> CtxFileVisitor<'v> {
    fn should_check(&self, ident: &syn::Ident) -> bool {
        if self.in_trait_or_impl == 0 {
            return false;
        }
        if self.current_ctx_last_trait.is_some() {
            return false;
        }
        if self
            .ctx_last_trait_method_names
            .contains(&ident.to_string())
        {
            return false;
        }
        true
    }
}

/// Walk one parsed file and collect method names declared inside any trait
/// listed in `CTX_LAST_TRAITS`. Used by the memory-pattern pass to skip
/// methods governed by an explicit ctx-last trait pass.
fn collect_ctx_last_trait_methods(ast: &syn::File) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for item in &ast.items {
        if let syn::Item::Trait(t) = item {
            if !CTX_LAST_TRAITS.contains(&t.ident.to_string().as_str()) {
                continue;
            }
            for ti in &t.items {
                if let syn::TraitItem::Fn(f) = ti {
                    out.insert(f.sig.ident.to_string());
                }
            }
        }
    }
    out
}

fn check_ctx_position(
    file: &std::path::Path,
    sig: &syn::Signature,
    expected: CtxExpect,
) -> Option<Violation> {
    let non_self: Vec<&syn::FnArg> = sig
        .inputs
        .iter()
        .filter(|a| !matches!(a, syn::FnArg::Receiver(_)))
        .collect();
    let ctx_pos = non_self
        .iter()
        .position(|a| arg_mentions_execution_context(a))?;
    let n = non_self.len();
    let ok = match expected {
        CtxExpect::First => ctx_pos == 0,
        CtxExpect::Last => ctx_pos + 1 == n,
    };
    if ok {
        return None;
    }
    let (line, col) = span_loc(sig.ident.span());
    let kind = match expected {
        CtxExpect::First => "ctx-first",
        CtxExpect::Last => "ctx-last",
    };
    Some(Violation::new(
        file.to_path_buf(),
        line,
        col,
        format!(
            "`{}` — expected {kind}; ctx is at position {} of {}",
            sig.ident, ctx_pos, n
        ),
    ))
}

fn arg_mentions_execution_context(arg: &syn::FnArg) -> bool {
    let syn::FnArg::Typed(pt) = arg else {
        return false;
    };
    type_mentions(&pt.ty, "ExecutionContext")
}

fn type_mentions(ty: &syn::Type, name: &str) -> bool {
    match ty {
        syn::Type::Reference(r) => type_mentions(&r.elem, name),
        syn::Type::Path(tp) => tp.path.segments.iter().any(|s| s.ident == name),
        _ => false,
    }
}
