//! Invariant 17 — heuristic policy externalisation (ADR-0034). Forbids
//! probability-shaped float literals (`0.X`) in heuristic-prone code paths.
//! These literals are policy decisions (jitter ratios, MMR lambdas,
//! retry-multiplier fractions, summarisation thresholds) and belong on
//! `*Policy` structs operators override, not in dispatch hot paths.
//!
//! The visitor walks `Lit::Float`; integer literals (`100`, `4096`) and
//! whole-number floats (`1.0`, `100.0`) are excluded — those are usually
//! durations or counts, with their own gates. Doc and ordinary comments
//! never reach the AST so prose mentioning thresholds is automatically
//! safe.

use anyhow::Result;
use syn::visit::Visit;

use crate::visitor::{Violation, parse, repo_root, report, rust_source_files_in, span_loc};

const ZONES: &[&str] = &[
    "crates/entelix-core/src/codecs",
    "crates/entelix-core/src/transports",
    "crates/entelix-cloud/src",
    "crates/entelix-agents/src",
    "crates/entelix-policy/src/cost.rs",
];

const MARKER: &str = "magic-ok";

pub(crate) fn run() -> Result<()> {
    let root = repo_root()?;
    let mut violations = Vec::new();

    for zone in ZONES {
        let zone_path = root.join(zone);
        let files = rust_source_files_in(&zone_path);
        for file in &files {
            let (src, ast) = parse(file)?;
            let lines: Vec<&str> = src.lines().collect();
            let mut v = MagicVisitor {
                file: file.clone(),
                lines: &lines,
                violations: &mut violations,
            };
            v.visit_file(&ast);
        }
    }

    report(
        "magic-constants (invariant 17)",
        violations,
        "ADR-0034 — move the literal onto a `*Policy` struct the operator can\n\
         override. Existing patterns: `RetryPolicy`, `MmrPolicy`,\n\
         `ConsolidationPolicy`. Genuinely safe literals (e.g. an exact ratio\n\
         fixed by a vendor wire format) carry an inline marker:\n\
         \n  let ratio = 0.5; // magic-ok: vendor-fixed ratio",
    )
}

struct MagicVisitor<'v, 'l> {
    file: std::path::PathBuf,
    lines: &'l [&'l str],
    violations: &'v mut Vec<Violation>,
}

impl<'ast, 'v, 'l> Visit<'ast> for MagicVisitor<'v, 'l> {
    fn visit_lit_float(&mut self, lit: &'ast syn::LitFloat) {
        // syn keeps the literal's textual representation — `0.5_f64`,
        // `0.001`, `1.0`, `100.0`. We match `0.<digit>+` only.
        let text = lit.base10_digits().to_string();
        if !is_probability_literal(&text) {
            return;
        }
        let (line, col) = span_loc(lit.span());
        let line_text = self
            .lines
            .get(line.saturating_sub(1))
            .copied()
            .unwrap_or("");
        if line_text.contains(MARKER) {
            return;
        }
        self.violations.push(Violation::new(
            self.file.clone(),
            line,
            col,
            format!("probability-shaped literal `{text}` — externalise onto a *Policy"),
        ));
    }
}

fn is_probability_literal(text: &str) -> bool {
    // base10_digits() returns the literal without type suffix — `0.5`, `1.0`.
    // We match `0.<digit>+` exactly. Anything starting with a digit other
    // than 0 (`1.0`, `100.0`) is excluded.
    let mut chars = text.chars();
    let Some('0') = chars.next() else {
        return false;
    };
    let Some('.') = chars.next() else {
        return false;
    };
    let rest: String = chars.collect();
    !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit())
}
