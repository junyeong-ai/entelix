#!/usr/bin/env bash
# scripts/check-managed-shape.sh — Invariants 1, 2, 3, 4 enforcement
#
# Enforces Anthropic managed-agent shape constraints that cannot be expressed
# at compile time alone:
#
#   inv 1 — Session is event SSoT   → SessionGraph must hold `events: Vec<GraphEvent>`
#   inv 2 — Harness is stateless    → Agent struct must NOT carry Persistence
#   inv 4 — Hand contract           → Tool trait must expose `execute` and nothing else
#                                     (informational; the trait is checked once defined)
#   inv 10 — Tokens never reach     → ExecutionContext must NOT embed CredentialProvider
#            Tool input
#
#   Brain passes hand (ADR-0035) — `Subagent` must NOT construct a fresh
#   `ToolRegistry::new()`. The narrowed registry the sub-agent dispatches
#   through is built via `parent_registry.with_only(...)` /
#   `parent_registry.filter(...)` so the parent's layer stack rides over
#   verbatim. A fresh registry would silently drop `PolicyLayer` /
#   `OtelLayer` / retry middleware — silent-loss bug at the worst possible
#   boundary.
#
# Phase 0: most targets do not exist yet (stub crates). The script returns 0
# when expected items are absent and only fires on actively-broken state.
#
# Exit 0 = healthy. Exit 1 = at least one violation.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if command -v rg >/dev/null 2>&1; then
    SEARCH=(rg --no-heading --line-number --color=never --type rust)
else
    SEARCH=(grep -rn --include='*.rs' --color=never)
fi

violations=0

fail() {
    echo "::error::$1"
    shift
    if (( $# > 0 )); then
        printf '%s\n' "$@"
    fi
    echo
    violations=$((violations + 1))
}

# ── inv 2 — Agent must not own a Persistence field ─────────────────────────
# Look only inside agent-related files in entelix-core (Phase 1+).
agent_dir="crates/entelix-core/src"
if [[ -d "${agent_dir}" ]]; then
    if hits="$("${SEARCH[@]}" --regexp 'pub[[:space:]]+persistence[[:space:]]*:[[:space:]]*[A-Za-z_]*Persistence' "${agent_dir}" 2>/dev/null || true)"; then
        if [[ -n "${hits}" ]]; then
            fail "Invariant 2 violation — Agent owns a Persistence field (Harness must be stateless)" \
                 "${hits}" \
                 "Move persistence behind Persistence trait passed per-call, not embedded in Agent."
        fi
    fi
fi

# ── inv 10 — ExecutionContext must not embed CredentialProvider ────────────
ctx_files="$(find crates/entelix-core/src -type f -name '*.rs' 2>/dev/null || true)"
if [[ -n "${ctx_files}" ]]; then
    # Match struct ExecutionContext { ... credential ...: CredentialProvider ... }
    # Two-step: locate ExecutionContext defs, then scan within for forbidden fields.
    while IFS= read -r file; do
        [[ -z "${file}" ]] && continue
        if awk '
            /pub[[:space:]]+struct[[:space:]]+ExecutionContext/ { in_ctx = 1; depth = 0 }
            in_ctx {
                # crude brace counter — works for the typical struct body
                for (i = 1; i <= length($0); i++) {
                    c = substr($0, i, 1)
                    if (c == "{") depth++
                    if (c == "}") { depth--; if (depth == 0) { in_ctx = 0 } }
                }
                if (/CredentialProvider/) { print FILENAME ":" NR ": " $0; found = 1 }
            }
            END { exit found ? 1 : 0 }
        ' "${file}" 2>/dev/null; then
            : # exit 0 → no hit
        else
            fail "Invariant 10 violation — ExecutionContext embeds CredentialProvider (tokens must never reach Tool input)" \
                 "Hit in ${file}" \
                 "Keep credentials inside Transport. ExecutionContext gets cancellation, deadline, tenant — never tokens."
        fi
    done <<< "${ctx_files}"
fi

# ── inv 1 — SessionGraph must hold events: Vec<GraphEvent> ─────────────────
session_dir="crates/entelix-session/src"
if [[ -d "${session_dir}" ]]; then
    # Only enforce once SessionGraph has been defined.
    if "${SEARCH[@]}" --regexp 'pub[[:space:]]+struct[[:space:]]+SessionGraph' "${session_dir}" >/dev/null 2>&1; then
        if ! "${SEARCH[@]}" --regexp 'pub[[:space:]]+events[[:space:]]*:[[:space:]]*Vec<[[:space:]]*GraphEvent[[:space:]]*>' "${session_dir}" >/dev/null 2>&1; then
            fail "Invariant 1 violation — SessionGraph defined without 'pub events: Vec<GraphEvent>'" \
                 "Session is the event SSoT. The events log is the single source of audit truth — no derived caches."
        fi
    fi
fi

# ── inv 4 — Tool trait shape (informational; activates once Tool is defined) ─
tools_dir="crates/entelix-core/src"
if [[ -d "${tools_dir}" ]]; then
    if tool_hits="$("${SEARCH[@]}" --regexp '^[[:space:]]*pub[[:space:]]+trait[[:space:]]+Tool\b' "${tools_dir}" 2>/dev/null || true)"; then
        if [[ -n "${tool_hits}" ]]; then
            # If Tool trait exists, ensure execute is present.
            if ! "${SEARCH[@]}" --regexp 'fn[[:space:]]+execute[[:space:]]*\(' "${tools_dir}" >/dev/null 2>&1; then
                fail "Invariant 4 violation — Tool trait defined but no 'execute' method found" \
                     "Tool's only method is execute(input, ctx) → output. Add nothing else."
            fi
        fi
    fi
fi

# ── ADR-0035 — Sub-agent must inherit parent ToolRegistry layer stack ─────
# `Subagent` lives in `entelix-agents`. Constructing a fresh
# `ToolRegistry::new()` inside this module would re-introduce D1 (the
# layer stack would silently disappear at the brain↔hand boundary).
# The narrowed registry must come from `parent_registry.with_only(...)`
# / `parent_registry.filter(...)` so layers ride over.
subagent_file="crates/entelix-agents/src/subagent.rs"
if [[ -f "${subagent_file}" ]]; then
    # Match `ToolRegistry::new` in code — exclude doc lines (`///`,
    # `//!`) where the symbol legitimately appears in prose.
    raw="$("${SEARCH[@]}" --regexp 'ToolRegistry::new\b' "${subagent_file}" 2>/dev/null || true)"
    if [[ -n "${raw}" ]]; then
        hits="$(printf '%s\n' "${raw}" | grep -Ev ':[[:space:]]*(///|//!)' || true)"
        if [[ -n "${hits}" ]]; then
            fail "ADR-0035 violation — Subagent constructs a fresh ToolRegistry (drops parent layer stack)" \
                 "${hits}" \
                 "Use parent_registry.with_only(allowed) or parent_registry.filter(predicate) instead." \
                 "The narrowed view inherits the parent's PolicyLayer / OtelLayer / retry middleware via Arc-shared factory." \
                 "See ADR-0035 §\"Sub-agent layer-stack inheritance\" + tests/subagent_layer_inheritance.rs."
        fi
    fi
fi

if (( violations > 0 )); then
    cat <<'EOF'

—————————————————————————————————————————————————————————————————————————
Managed-agent shape (Anthropic) is non-negotiable. See:
  - CLAUDE.md §"Anthropic Managed-Agent Shape"
  - docs/architecture/managed-agents.md
  - ADR-0005, ADR-0035

To fix: address each violation above. If a violation reflects a real design
need, write a new ADR — never weaken this gate.
—————————————————————————————————————————————————————————————————————————
EOF
    exit 1
fi

echo "check-managed-shape: clean."
