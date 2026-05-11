//! Invariant 17 — heuristic policy externalisation. Forbids
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

use std::path::Path;

use anyhow::Result;
use syn::visit::Visit;

use crate::visitor::{FileGate, Violation, run_file_gates, span_loc};

const ZONES: &[&str] = &[
    "crates/entelix-core/src/codecs",
    "crates/entelix-core/src/transports",
    "crates/entelix-cloud/src",
    "crates/entelix-agents/src",
    "crates/entelix-policy/src/cost.rs",
];

const MARKER: &str = "magic-ok";

const REMEDIATION: &str = "Move the literal onto a `*Policy` struct the operator can\n\
     override. Existing patterns: `RetryPolicy`, `MmrPolicy`,\n\
     `ConsolidationPolicy`. Genuinely safe literals (e.g. an exact ratio\n\
     fixed by a vendor wire format) carry an inline marker:\n\
     \n  let ratio = 0.5; // magic-ok: vendor-fixed ratio";

pub(crate) struct MagicConstantsGate;

impl FileGate for MagicConstantsGate {
    fn name(&self) -> &'static str {
        "magic-constants (invariant 17)"
    }

    fn applies_to(&self, path: &Path) -> bool {
        let s = path.to_string_lossy();
        ZONES.iter().any(|zone| s.contains(zone))
    }

    fn visit(&self, path: &Path, src: &str, ast: &syn::File, violations: &mut Vec<Violation>) {
        let lines: Vec<&str> = src.lines().collect();
        let mut v = MagicVisitor {
            file: path.to_path_buf(),
            lines: &lines,
            violations,
        };
        v.visit_file(ast);
    }

    fn remediation(&self) -> &'static str {
        REMEDIATION
    }
}

pub(crate) fn gates() -> Vec<Box<dyn FileGate>> {
    vec![Box::new(MagicConstantsGate)]
}

pub(crate) fn run() -> Result<()> {
    run_file_gates(&gates())
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
