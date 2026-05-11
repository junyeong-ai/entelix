//! Shared visitor utilities for invariant gates. Centralises file walking,
//! source loading, and the `Violation` reporting shape so each per-invariant
//! module reads as a focused predicate.
//!
//! One pipeline drives every AST invariant: [`run_invariants`] parses
//! the workspace once into a shared [`WorkspaceParse`] cache, runs
//! every [`FileGate`] file-by-file against that cache, then runs every
//! [`WorkspaceGate`] over the same cache for cross-file checks. Each
//! workspace `.rs` is read + parsed exactly once per invocation
//! regardless of how many gates inspect it — no double-parse, no
//! "walk + reparse per trait name" regression.
//!
//! The pipeline is single-threaded because the syn / proc-macro2
//! toolchain marks `syn::File` as `!Send` — the wrapping `Span` enum
//! holds a `proc_macro::Span` variant whose runtime data is
//! thread-local. ASTs therefore cannot cross thread boundaries today
//! and the cache cannot fan out across rayon workers. Empirical cost
//! is ~1-2 ms per file × ~300 files = ~0.45 s warm, which keeps the
//! wall-time profile within the cadence budget; correctness +
//! share-parse hygiene beats the modest parallelism win the prior
//! shape extracted. A future migration to a `Send`-able parser
//! (rust-analyzer's `rowan` IR, or proc-macro2's nightly `Send`
//! variant) would unlock cross-thread fan-out without changing the
//! gate trait surface.
//!
//! [`report`] sorts violations by `(path, line, col, message)` before
//! printing so two runs against the same code produce byte-identical
//! output regardless of the visit order.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use walkdir::WalkDir;

/// One detected violation. Always carries `path:line:col` so editors can jump
/// straight to it; `message` is the actionable fragment a reviewer sees.
/// `path` is workspace-relative so the report renders the same form a
/// reviewer would type — `crates/entelix-core/src/lib.rs:12:3`.
#[derive(Clone)]
pub(crate) struct Violation {
    pub(crate) path: PathBuf,
    pub(crate) line: usize,
    pub(crate) col: usize,
    pub(crate) message: String,
}

impl Violation {
    pub(crate) fn new(
        path: impl Into<PathBuf>,
        line: usize,
        col: usize,
        message: impl Into<String>,
    ) -> Self {
        Self {
            path: path.into(),
            line,
            col,
            message: message.into(),
        }
    }
    /// File-level violation (no source line). Used by manifest checks.
    pub(crate) fn file(path: impl Into<PathBuf>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            line: 0,
            col: 0,
            message: message.into(),
        }
    }
}

/// Print every violation in the canonical `path:line:col: msg` form,
/// prefix the reviewer-facing remediation block, and return `Err` so
/// the binary exits 1. Sorts violations by `(path, line, col,
/// message)` so the report is deterministic — same input, same
/// output, every run.
pub(crate) fn report(invariant: &str, violations: Vec<Violation>, remediation: &str) -> Result<()> {
    if violations.is_empty() {
        return Ok(());
    }
    let mut violations = violations;
    violations.sort_by(|a, b| {
        a.path
            .cmp(&b.path)
            .then(a.line.cmp(&b.line))
            .then(a.col.cmp(&b.col))
            .then_with(|| a.message.cmp(&b.message))
    });
    let mut out = String::new();
    writeln!(out, "{invariant} — {} violation(s):", violations.len()).ok();
    for v in &violations {
        if v.line == 0 {
            writeln!(out, "  {}: {}", v.path.display(), v.message).ok();
        } else {
            writeln!(
                out,
                "  {}:{}:{}: {}",
                v.path.display(),
                v.line,
                v.col,
                v.message
            )
            .ok();
        }
    }
    writeln!(out).ok();
    writeln!(out, "{remediation}").ok();
    anyhow::bail!("{out}");
}

/// All `.rs` files under `crates/`, skipping `target/` and `tests/`
/// trees — invariant gates apply to first-party production code.
///
/// Returns `Err` if `crates/` is missing (wrong `CARGO_MANIFEST_DIR`,
/// stale checkout, fresh workspace) so an empty file list never
/// silently reports every gate as green. Pruning happens at the
/// directory level via `filter_entry`, so `WalkDir` never descends
/// into a multi-gigabyte `target/` before discovering it should be
/// skipped.
pub(crate) fn rust_source_files(repo_root: &Path) -> Result<Vec<PathBuf>> {
    let crates = repo_root.join("crates");
    if !crates.exists() {
        anyhow::bail!(
            "no `crates/` directory under {} — is `CARGO_MANIFEST_DIR` pointing at the workspace root?",
            repo_root.display()
        );
    }
    let mut out = Vec::new();
    let walker = WalkDir::new(&crates).into_iter().filter_entry(|entry| {
        // Prune `target/` and `tests/` before descending. WalkDir's
        // depth-0 root entry has no skip-worthy name; the filter
        // applies from depth-1 inward.
        if entry.depth() == 0 {
            return true;
        }
        let name = entry.file_name().to_string_lossy();
        name != "target" && name != "tests"
    });
    for entry in walker.filter_map(Result::ok) {
        let path = entry.path();
        if !path.is_file() || path.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        out.push(path.to_path_buf());
    }
    if out.is_empty() {
        anyhow::bail!(
            "no `.rs` files found under {} — invariant gates would falsely report green; check `CARGO_MANIFEST_DIR`.",
            crates.display()
        );
    }
    out.sort();
    Ok(out)
}

/// Read a source file from disk.
pub(crate) fn read(path: &Path) -> Result<String> {
    fs::read_to_string(path).with_context(|| format!("read {}", path.display()))
}

/// Parse a Rust source file into a `syn::File`. Caller decides whether
/// to keep the source string (for raw-line scans) or drop it. Kept
/// alongside [`read`] because non-pipeline gates (e.g.
/// `facade_completeness`'s targeted re-parse of a single facade
/// `lib.rs`) need an ad-hoc parse path outside the orchestrator.
pub(crate) fn parse(path: &Path) -> Result<(String, syn::File)> {
    let src = read(path)?;
    let ast = syn::parse_file(&src).with_context(|| format!("parse {}", path.display()))?;
    Ok((src, ast))
}

/// Resolve `(line, col)` from a byte offset into `source`. `proc_macro2::Span`
/// is the more idiomatic carrier but file-relative byte offsets are what
/// `proc_macro2::LineColumn` reports through the `extra-traits` feature; this
/// helper exists for the few sites where we resolve our own offsets (comments,
/// macro fragments).
pub(crate) fn line_col(source: &str, offset: usize) -> (usize, usize) {
    let mut line = 1;
    let mut col = 1;
    for (i, ch) in source.char_indices() {
        if i >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}

/// Repo root — every subcommand assumes the binary is invoked via the
/// `cargo xtask` alias which sets `CARGO` and starts at workspace root.
pub(crate) fn repo_root() -> Result<PathBuf> {
    // `CARGO_MANIFEST_DIR` for xtask is `<repo>/xtask`, so take its parent.
    let manifest = std::env::var("CARGO_MANIFEST_DIR").context("CARGO_MANIFEST_DIR unset")?;
    let path = PathBuf::from(manifest)
        .parent()
        .context("xtask manifest has no parent")?
        .to_path_buf();
    Ok(path)
}

/// Span → file-relative `(line, col)`. `proc_macro2::Span::start` returns
/// `LineColumn { line, column }` where line is 1-indexed and column is
/// 0-indexed; we normalise column to 1-indexed for editor parity.
pub(crate) fn span_loc(span: proc_macro2::Span) -> (usize, usize) {
    let lc = span.start();
    (lc.line, lc.column + 1)
}

/// True when `ty` carries a type whose last path segment matches
/// `predicate`. Recurses uniformly through every containment shape
/// `syn::Type` exposes:
///
///   * References (`&T`, `&mut T`) and parenthesised / grouped
///     wrappers — unwrap one level.
///   * Trait objects (`dyn Trait`) and `impl Trait` — any bound whose
///     last path segment satisfies the predicate counts.
///   * Tuples, fixed-size arrays, slices — recurse into each element
///     type.
///   * Path types (`Foo`, `Bar<T>`, `HashMap<K, V>`, `Box<T>`,
///     `Vec<T>`, …) — predicate first, then recurse into every
///     `<T>` generic argument the path carries.
///
/// The recursion is uniform — no hand-curated "transparent wrapper"
/// list to drift out of sync with stdlib + ecosystem container
/// types. Shared by every gate that ends in "field X carries an
/// inner Y" — `surface_hygiene` (`*Error`), `managed_shape`
/// (`*Persistence` / `*Checkpointer` / `*SessionLog`) — so the
/// recursion semantics are auditable in one place.
pub(crate) fn type_carries(ty: &syn::Type, predicate: &dyn Fn(&str) -> bool) -> bool {
    match ty {
        syn::Type::Reference(r) => type_carries(&r.elem, predicate),
        syn::Type::Paren(p) => type_carries(&p.elem, predicate),
        syn::Type::Group(g) => type_carries(&g.elem, predicate),
        syn::Type::TraitObject(to) => to.bounds.iter().any(|bound| {
            if let syn::TypeParamBound::Trait(t) = bound {
                t.path
                    .segments
                    .last()
                    .map(|s| predicate(&s.ident.to_string()))
                    .unwrap_or(false)
            } else {
                false
            }
        }),
        syn::Type::ImplTrait(it) => it.bounds.iter().any(|bound| {
            if let syn::TypeParamBound::Trait(t) = bound {
                t.path
                    .segments
                    .last()
                    .map(|s| predicate(&s.ident.to_string()))
                    .unwrap_or(false)
            } else {
                false
            }
        }),
        syn::Type::Tuple(t) => t.elems.iter().any(|e| type_carries(e, predicate)),
        syn::Type::Array(a) => type_carries(&a.elem, predicate),
        syn::Type::Slice(s) => type_carries(&s.elem, predicate),
        syn::Type::Path(tp) => {
            let Some(seg) = tp.path.segments.last() else {
                return false;
            };
            if predicate(&seg.ident.to_string()) {
                return true;
            }
            // Recurse through every generic argument — covers `Vec<T>`,
            // `HashMap<K, V>`, `Result<T, E>`, `Box<T>`, `Arc<T>`,
            // `Option<T>`, `Pin<P>`, plus any future container.
            if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                for arg in &args.args {
                    if let syn::GenericArgument::Type(inner) = arg
                        && type_carries(inner, predicate)
                    {
                        return true;
                    }
                }
            }
            false
        }
        _ => false,
    }
}

/// One sequenced cadence step — name shown in the banner, closure that
/// runs the underlying check. Failure short-circuits the cadence.
/// Distinct shape from [`FileGate`] / [`WorkspaceGate`] (those are
/// AST dispatch contracts); a `CadenceStep` is one entry in the
/// outer bundled-cadence sequence (`fmt → clippy → test → invariants`).
pub(crate) type CadenceStep = (&'static str, fn() -> Result<()>);

/// One parsed workspace file in the shared cache.
pub(crate) struct ParsedFile {
    /// Path relative to the workspace root (e.g.
    /// `crates/entelix-core/src/lib.rs`). Stable across machines.
    pub(crate) rel: PathBuf,
    /// File source text — retained so file gates that scan raw lines
    /// (`no-shims` comment markers, `silent-fallback` marker
    /// detection) operate on the same string the parser consumed
    /// without a second `read_to_string`.
    pub(crate) src: String,
    /// Parsed Rust AST. Every gate that inspects this file borrows
    /// the same instance — no double-parse, no thread-shipping
    /// (`syn::File` is `!Send`).
    pub(crate) ast: syn::File,
}

/// Single-threaded parse cache shared across every gate in one
/// pipeline invocation. Built once by [`run_invariants`] and iterated
/// by every [`FileGate`] and [`WorkspaceGate`] in turn.
pub(crate) struct WorkspaceParse {
    files: Vec<ParsedFile>,
}

impl WorkspaceParse {
    pub(crate) fn iter(&self) -> impl Iterator<Item = &ParsedFile> {
        self.files.iter()
    }
}

/// Single-file invariant visitor. The orchestrator dispatches the
/// shared AST to every gate whose [`Self::applies_to`] matches —
/// each file is parsed once across the whole pipeline.
///
/// `applies_to` operates on the **workspace-relative** path
/// (`crates/<crate>/src/<file>.rs`) so component-aware
/// [`Path::starts_with`] gives correct zone scoping. Substring
/// matching on absolute paths is reviewer-rejected because an
/// unrelated path component could coincidentally embed the zone
/// string.
pub(crate) trait FileGate: Send + Sync {
    /// Identifier reported on violation. Matches the corresponding
    /// `cargo xtask <name>` subcommand for editor / CI jump.
    fn name(&self) -> &'static str;

    /// Filter — return `false` to skip the gate for this path
    /// (zone-scoped invariants narrow to e.g. `crates/entelix-core/src`).
    /// Default: applies to every Rust source file the workspace walker
    /// surfaces.
    fn applies_to(&self, rel_path: &Path) -> bool {
        let _ = rel_path;
        true
    }

    /// Inspect one parsed file, push every detected violation into
    /// `violations`. `rel_path` is the workspace-relative form —
    /// visitors stash it verbatim in [`Violation::new`] so reports
    /// render `crates/.../file.rs:L:C`.
    fn visit(&self, rel_path: &Path, src: &str, ast: &syn::File, violations: &mut Vec<Violation>);

    /// Reviewer-facing remediation block printed beneath the
    /// per-violation list. Independent of `name` so the gate's
    /// remediation can survive a rename without docstring drift.
    fn remediation(&self) -> &'static str;
}

/// Cross-file invariant gate — runs after every [`FileGate`] against
/// the same shared [`WorkspaceParse`] cache. Use this shape for
/// checks that need a whole-workspace view (which file declares
/// trait X, what methods does it carry, …).
pub(crate) trait WorkspaceGate: Send + Sync {
    fn name(&self) -> &'static str;

    /// Inspect every parsed file (or a relevant subset) and push every
    /// detected violation into `violations`.
    fn check(&self, parse: &WorkspaceParse, violations: &mut Vec<Violation>);

    fn remediation(&self) -> &'static str;
}

/// Run the full invariant pipeline against the workspace.
///
/// Stages, all sharing one `WorkspaceParse` cache:
///
/// 1. **Parse** — every `.rs` file under `crates/` is read + parsed
///    once. A parse error surfaces immediately as `Err`; a malformed
///    Rust file is a genuine signal, not a "gate had nothing to
///    report" outcome.
/// 2. **File pass** — for each cached file, every [`FileGate`] whose
///    `applies_to` matches receives the shared AST. Violations
///    accumulate per gate.
/// 3. **Workspace pass** — every [`WorkspaceGate`] receives the full
///    cache. Violations accumulate per gate.
///
/// Fails fast at the first gate (file or workspace) that emitted
/// violations, in input order. Single-threaded — see module docs for
/// the `syn::File: !Send` constraint that forces this.
pub(crate) fn run_invariants(
    file_gates: &[Box<dyn FileGate>],
    workspace_gates: &[Box<dyn WorkspaceGate>],
) -> Result<()> {
    if file_gates.is_empty() && workspace_gates.is_empty() {
        return Ok(());
    }
    let root = repo_root()?;
    let files = rust_source_files(&root)?;

    // ── Stage 1: parse the workspace once ──
    let mut cache: Vec<ParsedFile> = Vec::with_capacity(files.len());
    for abs in &files {
        let rel = abs
            .strip_prefix(&root)
            .with_context(|| format!("strip repo root from {}", abs.display()))?
            .to_path_buf();
        let (src, ast) = parse(abs)?;
        cache.push(ParsedFile { rel, src, ast });
    }
    let parse_cache = WorkspaceParse { files: cache };

    // ── Stage 2: file pass over the shared cache ──
    if !file_gates.is_empty() {
        let mut per_gate: HashMap<&'static str, Vec<Violation>> = HashMap::new();
        for file in parse_cache.iter() {
            for gate in file_gates {
                if !gate.applies_to(&file.rel) {
                    continue;
                }
                let mut vs = Vec::new();
                gate.visit(&file.rel, &file.src, &file.ast, &mut vs);
                if !vs.is_empty() {
                    per_gate.entry(gate.name()).or_default().extend(vs);
                }
            }
        }
        for gate in file_gates {
            if let Some(violations) = per_gate.get(gate.name())
                && !violations.is_empty()
            {
                return report(gate.name(), violations.clone(), gate.remediation());
            }
        }
    }

    // ── Stage 3: workspace pass over the same cache ──
    for gate in workspace_gates {
        let mut violations = Vec::new();
        gate.check(&parse_cache, &mut violations);
        if !violations.is_empty() {
            return report(gate.name(), violations, gate.remediation());
        }
    }

    Ok(())
}

#[cfg(test)]
mod type_carries_tests {
    //! Adversarial fixture corpus for [`type_carries`]. The recursion
    //! semantics — references, parens, groups, trait objects, impl
    //! Trait, tuples, arrays, slices, paths with arbitrary generic
    //! arguments — are load-bearing for both `surface_hygiene`
    //! (`*Error`) and `managed_shape` (`*Persistence` / `*Checkpointer`
    //! / `*SessionLog`). One regression in this helper silently
    //! widens every gate's coverage hole; the fixtures below pin the
    //! recursion shape so a future tweak surfaces here first.

    use super::type_carries;

    fn parse(src: &str) -> syn::Type {
        syn::parse_str(src).unwrap_or_else(|e| panic!("parse `{src}`: {e}"))
    }

    fn ends_in_error(s: &str) -> bool {
        s == "Error" || s.ends_with("Error")
    }

    fn carries_error(ty: &str) -> bool {
        type_carries(&parse(ty), &ends_in_error)
    }

    #[test]
    fn bare_path_with_matching_last_segment() {
        assert!(carries_error("Error"));
        assert!(carries_error("MyError"));
        assert!(carries_error("crate::auth::AuthError"));
    }

    #[test]
    fn bare_path_without_match_is_false() {
        assert!(!carries_error("String"));
        assert!(!carries_error("u64"));
        assert!(!carries_error("std::collections::HashMap"));
    }

    #[test]
    fn single_arity_wrappers_recurse() {
        assert!(carries_error("Box<MyError>"));
        assert!(carries_error("Arc<MyError>"));
        assert!(carries_error("Rc<MyError>"));
        assert!(carries_error("Option<MyError>"));
        assert!(carries_error("Pin<Box<MyError>>"));
    }

    #[test]
    fn multi_arity_paths_recurse_through_every_arg() {
        // `HashMap<K, V>` — error in the value slot.
        assert!(carries_error("HashMap<String, MyError>"));
        // …and in the key slot.
        assert!(carries_error("HashMap<MyError, String>"));
        // `Result<T, E>` — error in the error slot is the common case.
        assert!(carries_error("Result<String, MyError>"));
        // Both slots clear → false.
        assert!(!carries_error("HashMap<String, u32>"));
        assert!(!carries_error("Result<String, u32>"));
    }

    #[test]
    fn references_unwrap_one_level() {
        assert!(carries_error("&MyError"));
        assert!(carries_error("&mut MyError"));
        assert!(carries_error("&'a MyError"));
        assert!(!carries_error("&String"));
    }

    #[test]
    fn trait_objects_recognise_bounds() {
        assert!(carries_error("dyn MyError"));
        assert!(carries_error("dyn std::error::Error + Send + Sync"));
        assert!(carries_error(
            "Box<dyn std::error::Error + Send + Sync + 'static>"
        ));
        assert!(carries_error("dyn Send + Sync + MyError"));
        assert!(!carries_error("dyn Send + Sync"));
    }

    #[test]
    fn impl_trait_bounds_recognised() {
        assert!(carries_error("impl MyError"));
        assert!(carries_error("impl Send + MyError"));
        assert!(!carries_error("impl Send + Sync"));
    }

    #[test]
    fn tuples_arrays_slices_recurse() {
        assert!(carries_error("(String, MyError)"));
        assert!(carries_error("(MyError,)"));
        assert!(carries_error("[MyError; 5]"));
        assert!(carries_error("&[MyError]"));
        assert!(carries_error("&[Box<dyn std::error::Error>]"));
        assert!(!carries_error("(String, u32)"));
        assert!(!carries_error("[u8; 32]"));
    }

    #[test]
    fn paren_and_group_are_transparent() {
        assert!(carries_error("(MyError)"));
        assert!(carries_error("((MyError))"));
    }

    #[test]
    fn deep_nesting_recurses_all_the_way() {
        assert!(carries_error("Vec<Option<Box<dyn std::error::Error>>>"));
        assert!(carries_error("Result<Vec<u8>, Arc<MyError>>"));
        assert!(carries_error("Option<(String, &MyError)>"));
        assert!(carries_error("HashMap<String, Vec<Box<dyn Error>>>"));
    }

    #[test]
    fn unsupported_type_variants_fall_through_to_false() {
        // Function pointers and raw pointers are not error carriers
        // semantically and fall through to the catch-all.
        assert!(!carries_error("fn(MyError) -> ()"));
        assert!(!carries_error("*const MyError"));
        assert!(!carries_error("*mut MyError"));
    }

    #[test]
    fn hrtb_trait_objects_recurse_bounds() {
        // Higher-ranked trait bounds are valid trait-object shapes
        // (`Box<dyn for<'r> Visit<'r>>`). The bound's last path
        // segment is what predicate matching keys on; HRTB wrapping
        // must not hide the trait ident.
        assert!(carries_error(
            "Box<dyn for<'r> std::error::Error + Send + Sync + 'r>"
        ));
        assert!(carries_error("&dyn for<'a> MyError"));
    }

    #[test]
    fn const_generic_args_dont_mask_inner_types() {
        // Const-generic args carry no type info, so they're skipped
        // during recursion; the type-shaped sibling args still
        // recurse normally.
        assert!(carries_error("Foo<MyError, 32>"));
        assert!(!carries_error("Foo<u8, 32>"));
        // Array-as-generic-arg: `Foo<[u8; 32]>` — Foo's first
        // generic arg is a Type::Array which recurses to u8 (not
        // error). Should be false.
        assert!(!carries_error("Foo<[u8; 32]>"));
        // Same shape with an error element should fire.
        assert!(carries_error("Foo<[MyError; 32]>"));
    }

    #[test]
    fn lifetime_parameterized_paths_recurse() {
        // Lifetime args are skipped during generic-arg recursion;
        // the type-shaped sibling args still surface the predicate.
        assert!(carries_error("Foo<'a, MyError>"));
        assert!(carries_error("&'a Foo<'b, MyError>"));
        assert!(!carries_error("Foo<'a, 'b>"));
    }

    #[test]
    fn associated_type_bindings_in_trait_objects() {
        // `Box<dyn Trait<Output = MyError>>` — the binding's RHS is
        // a Type that recursion currently does not unwrap. This is
        // a known limit: the visitor inspects bound paths only, not
        // bound bindings. Documented expected behaviour today.
        assert!(!carries_error("Box<dyn Trait<Output = MyError>>"));
    }

    #[test]
    fn predicate_is_arbitrary() {
        // Same helper must work for any suffix the gates care about —
        // `*Persistence`, `*Checkpointer`, `*SessionLog`.
        assert!(type_carries(&parse("Arc<dyn Persistence>"), &|s: &str| s
            .ends_with("Persistence"),));
        assert!(type_carries(
            &parse("Vec<Arc<PostgresCheckpointer>>"),
            &|s: &str| s.ends_with("Checkpointer"),
        ));
        assert!(type_carries(
            &parse("Option<Box<dyn SessionLog>>"),
            &|s: &str| s.ends_with("SessionLog")
        ));
        assert!(!type_carries(
            &parse("Vec<Arc<HashMap<String, u32>>>"),
            &|s: &str| s.ends_with("Persistence"),
        ));
    }
}
