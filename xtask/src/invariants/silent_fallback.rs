//! Invariant 15 — silent fallback prohibition (ADR-0032). In the codec,
//! transport, and cost-meter hot zones, every `.unwrap_or*` site requires
//! an audited `// silent-fallback-ok: <reason>` marker on the same line.
//!
//! No baseline counts. The marker is the unique authentication channel —
//! presence on the line reading the call, or the call is rejected. This
//! removes the regex-script class of regression where a baseline number
//! drifted up over time and the reviewer never saw it.

use anyhow::Result;
use syn::visit::Visit;

use crate::visitor::{Violation, parse, repo_root, report, rust_source_files_in};

const HOT_ZONES: &[&str] = &[
    "crates/entelix-core/src/codecs",
    "crates/entelix-core/src/transports",
    "crates/entelix-cloud/src/bedrock",
    "crates/entelix-cloud/src/vertex",
    "crates/entelix-cloud/src/foundry",
    "crates/entelix-policy/src/cost.rs",
];

const FALLBACK_METHODS: &[&str] = &["unwrap_or", "unwrap_or_default", "unwrap_or_else"];

const MARKER: &str = "silent-fallback-ok";

pub(crate) fn run() -> Result<()> {
    let root = repo_root()?;
    let mut violations = Vec::new();

    for zone in HOT_ZONES {
        let zone_path = root.join(zone);
        let files = rust_source_files_in(&zone_path);
        for file in &files {
            let (src, ast) = parse(file)?;
            let lines: Vec<&str> = src.lines().collect();
            let mut v = FallbackVisitor {
                file: file.clone(),
                lines: &lines,
                violations: &mut violations,
            };
            v.visit_file(&ast);
        }
    }

    report(
        "silent-fallback (invariant 15)",
        violations,
        "ADR-0032 — surface every information-loss event through a typed\n\
         channel (`ModelWarning::LossyEncode { field, detail }` or\n\
         `StopReason::Other { raw }`). Default-injecting a value\n\
         (`max_tokens.unwrap_or(4096)`, `cache_rate.unwrap_or(input/10)`) is\n\
         a bug regardless of how reasonable the default looks.\n\
         \n\
         If a fallback is genuinely safe (e.g. a missing-string accessor\n\
         that defaults to \"\"), add the marker on the same line:\n\
         \n  .unwrap_or(\"\") // silent-fallback-ok: missing optional field",
    )
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
