//! Invariant 15 — silent fallback prohibition. In the codec,
//! transport, and cost-meter hot zones, every `.unwrap_or*` / `or_else`
//! site requires an audited `// silent-fallback-ok: <reason>` marker on
//! the same line.
//!
//! No baseline counts. The marker is the unique authentication channel —
//! presence on the line reading the call, or the call is rejected. This
//! removes the regex-script class of regression where a baseline number
//! drifted up over time and the reviewer never saw it.
//!
//! ## Hot-zone maintenance contract
//!
//! [`HOT_ZONES`] is the curated list of zones the rule enforces:
//! codecs, transports, cost-meter. The list is editable from one place
//! and visible at every review of a new codec / transport / cost
//! source. When a new codec or transport lands — particularly in a new
//! crate — the reviewer's checklist item is "does this zone belong on
//! the silent-fallback list?". A structural alternative (detect any
//! file declaring `impl Codec for …` / `impl Transport for …` and
//! enrol it automatically) would dissolve the maintenance contract
//! into the code, at the cost of a workspace-wide pre-pass on every
//! invocation. The hardcoded list is deliberate: the cost of one new
//! line on a new codec PR is lower than the cost of a workspace-wide
//! impl scan on every cadence run, and the review-time prompt is the
//! intended catch.

use std::path::Path;

use anyhow::Result;
use syn::visit::Visit;

use crate::visitor::{FileGate, Violation, run_invariants};

const HOT_ZONES: &[&str] = &[
    "crates/entelix-core/src/codecs",
    "crates/entelix-core/src/transports",
    "crates/entelix-cloud/src/bedrock",
    "crates/entelix-cloud/src/vertex",
    "crates/entelix-cloud/src/foundry",
    "crates/entelix-policy/src/cost.rs",
];

const FALLBACK_METHODS: &[&str] = &[
    "unwrap_or",
    "unwrap_or_default",
    "unwrap_or_else",
    "or_else",
];

const MARKER: &str = "silent-fallback-ok";

const REMEDIATION: &str = "Invariant 15 — surface every information-loss event through a typed\n\
     channel (`ModelWarning::LossyEncode { field, detail }` or\n\
     `StopReason::Other { raw }`). Default-injecting a value\n\
     (`max_tokens.unwrap_or(4096)`, `cache_rate.unwrap_or(input/10)`) is\n\
     a bug regardless of how reasonable the default looks.\n\
     \n\
     If a fallback is genuinely safe (e.g. a missing-string accessor\n\
     that defaults to \"\"), add the marker on the same line:\n\
     \n  .unwrap_or(\"\") // silent-fallback-ok: missing optional field";

pub(crate) struct SilentFallbackGate;

impl FileGate for SilentFallbackGate {
    fn name(&self) -> &'static str {
        "silent-fallback (invariant 15)"
    }

    fn applies_to(&self, rel_path: &Path) -> bool {
        HOT_ZONES.iter().any(|zone| rel_path.starts_with(zone))
    }

    fn visit(&self, rel_path: &Path, src: &str, ast: &syn::File, violations: &mut Vec<Violation>) {
        let lines: Vec<&str> = src.lines().collect();
        let mut v = FallbackVisitor {
            file: rel_path.to_path_buf(),
            lines: &lines,
            violations,
        };
        v.visit_file(ast);
    }

    fn remediation(&self) -> &'static str {
        REMEDIATION
    }
}

pub(crate) fn file_gates() -> Vec<Box<dyn FileGate>> {
    vec![Box::new(SilentFallbackGate)]
}

pub(crate) fn run() -> Result<()> {
    run_invariants(&file_gates(), &[])
}

struct FallbackVisitor<'v, 'l> {
    file: std::path::PathBuf,
    lines: &'l [&'l str],
    violations: &'v mut Vec<Violation>,
}

impl<'ast, 'v, 'l> Visit<'ast> for FallbackVisitor<'v, 'l> {
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        let method = call.method.to_string();
        if FALLBACK_METHODS.contains(&method.as_str()) {
            let method_span = call.method.span();
            let report_line = method_span.start().line;
            let report_col = method_span.start().column + 1;
            // The marker may sit anywhere inside the call expression — on
            // the method-name line itself, on the closure body of an
            // `unwrap_or_else`, or on the trailing `;` line that closes a
            // multi-line call. Check the full span line range so reviewers
            // can keep the marker next to whichever fragment of the
            // construct it documents.
            use syn::spanned::Spanned;
            let full = syn::Expr::MethodCall(call.clone()).span();
            let start = full.start().line.saturating_sub(1);
            let end = full.end().line;
            let mut found = false;
            for idx in start..end.min(self.lines.len()) {
                if self.lines[idx].contains(MARKER) {
                    found = true;
                    break;
                }
            }
            if !found {
                self.violations.push(Violation::new(
                    self.file.clone(),
                    report_line,
                    report_col,
                    format!("`.{method}()` without `// {MARKER}: <reason>` marker"),
                ));
            }
        }
        syn::visit::visit_expr_method_call(self, call);
    }
}
