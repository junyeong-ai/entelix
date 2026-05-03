#!/usr/bin/env bash
# scripts/check-magic-constants.sh — invariant #17 enforcement
# (CLAUDE.md §"Heuristic policy externalisation" + ADR-0034).
#
# Forbids embedded probability literals (`0.X`) in heuristic-prone
# code paths — codecs, transports, recipe agents, and the cost meter.
# These literals are policy decisions (jitter ratios, MMR lambdas,
# retry-multiplier fractions, summarisation thresholds) and belong on
# `*Policy` structs the operator can override, not in the dispatch
# hot path.
#
# Other classes of magic numbers (HTTP status codes, byte caps, token
# budgets) carry their own, more precise gates — silent-fallback,
# codec consistency, lossy-warning completeness. Probability literals
# are the canonical "heuristic in disguise" tell, so this gate
# narrows on them.
#
# Doc lines (`///`, `//!`) and full-line comments (`//`) are excluded
# so prose can mention numeric thresholds without tripping the gate.
# String literals and crate version doc(html_root_url = "...") are
# also excluded.
#
# Exit 0 = clean. Exit 1 = at least one violation.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if command -v rg >/dev/null 2>&1; then
    SEARCH=(rg --no-heading --line-number --color=never --type rust)
else
    SEARCH=(grep -rn --include='*.rs' --color=never -E)
fi

ZONES=(
    "crates/entelix-core/src/codecs/"
    "crates/entelix-core/src/transports/"
    "crates/entelix-cloud/src/"
    "crates/entelix-agents/src/"
    "crates/entelix-policy/src/cost.rs"
)

# Probability-shaped literal: `0.<digit>+`, NOT preceded by another
# digit (so `1.0` / `100.0` etc. are not matches — those are usually
# durations in seconds or counts).
PATTERN='(^|[^0-9])0\.[0-9]+'

violations=0
for zone in "${ZONES[@]}"; do
    candidates="$("${SEARCH[@]}" --regexp "${PATTERN}" "${zone}" 2>/dev/null || true)"
    if [[ -z "${candidates}" ]]; then
        continue
    fi
    # Drop doc lines, full-line comments, and version literals.
    filtered="$(printf '%s\n' "${candidates}" \
        | grep -Ev ':[[:space:]]*(///|//!|//)' \
        | grep -v 'html_root_url' \
        | grep -v 'magic-ok' \
        || true)"
    if [[ -z "${filtered}" ]]; then
        continue
    fi
    echo "::error::invariant #17 violation — probability-shaped literal in ${zone}"
    printf '%s\n' "${filtered}" | head -20 | sed 's/^/    /'
    echo
    echo "  Move the literal onto a *Policy struct the operator can override:"
    echo "    - jitter / mmr lambda / cooldown fractions  → existing *Policy"
    echo "    - new heuristic                              → introduce a *Policy"
    echo "  ADR-0034 §\"Heuristic policy externalisation\" lists the canonical"
    echo "  patterns. Genuinely safe literals (e.g. an exact ratio fixed by a"
    echo "  vendor wire format) carry an inline marker:"
    echo "    let ratio = 0.5; // magic-ok: vendor-fixed ratio"
    echo
    violations=$((violations + 1))
done

if (( violations > 0 )); then
    cat <<'EOF'

—————————————————————————————————————————————————————————————————————————
Invariant #17 — heuristic policy externalisation (CLAUDE.md + ADR-0034).

Probability-shaped literals (jitter ratios, retry multipliers, MMR
lambdas, summarisation thresholds) are policy decisions. They belong
on `*Policy` structs operators can override, not buried in the
dispatch hot path. The pattern that already lives in the codebase:
RetryPolicy, MmrPolicy, ConsolidationPolicy. Follow it.
—————————————————————————————————————————————————————————————————————————
EOF
    exit 1
fi

echo "check-magic-constants: 0 violations across ${#ZONES[@]} zone(s)."
