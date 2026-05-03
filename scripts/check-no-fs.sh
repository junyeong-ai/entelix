#!/usr/bin/env bash
# scripts/check-no-fs.sh — Invariant 9 enforcement
#
# Forbids filesystem / shell / sandbox imports in first-party crates.
# Canon: CLAUDE.md §"12 Architecture Invariants" #9, ADR-0003.
#
# Exit 0 = no hits. Exit 1 = at least one hit (CI fails).

set -euo pipefail

# ── Locate repo root regardless of caller's cwd ────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── Pick a grep engine ─────────────────────────────────────────────────────
if command -v rg >/dev/null 2>&1; then
    SEARCH=(rg --no-heading --line-number --color=never --type rust)
else
    # POSIX grep fallback. -r recurses, --include filters, -n shows line numbers.
    SEARCH=(grep -rn --include='*.rs' --color=never)
fi

# ── Forbidden import patterns (Rust-only — match `use ...` lines) ──────────
PATTERNS=(
    '^[[:space:]]*use[[:space:]]+std::fs(::|;|[[:space:]])'
    '^[[:space:]]*use[[:space:]]+std::process(::|;|[[:space:]])'
    '^[[:space:]]*use[[:space:]]+std::os::unix::process(::|;|[[:space:]])'
    '^[[:space:]]*use[[:space:]]+tokio::fs(::|;|[[:space:]])'
    '^[[:space:]]*use[[:space:]]+tokio::process(::|;|[[:space:]])'
    '^[[:space:]]*use[[:space:]]+landlock(::|;|[[:space:]])'
    '^[[:space:]]*use[[:space:]]+seatbelt(::|;|[[:space:]])'
    '^[[:space:]]*use[[:space:]]+tree_sitter(::|;|[[:space:]])'
    '^[[:space:]]*use[[:space:]]+nix(::|;|[[:space:]])'
)

violations=0
for pat in "${PATTERNS[@]}"; do
    if hits="$("${SEARCH[@]}" --regexp "${pat}" crates/ 2>/dev/null || true)"; then
        if [[ -n "${hits}" ]]; then
            echo "::error::Invariant 9 violation — forbidden import:"
            echo "${hits}"
            echo
            violations=$((violations + 1))
        fi
    fi
done

if (( violations > 0 )); then
    cat <<'EOF'

—————————————————————————————————————————————————————————————————————————
entelix is web-service-native. No filesystem, no shell, no local sandbox.
Reason — ADR-0003 + invariant 9. Even tokio::fs is forbidden in first-party
crates; agent file/process work belongs in user code or external tools.

To fix:
  - Replace `std::fs` reads with HTTP / Store / explicit user-supplied bytes.
  - Replace `std::process` shell-outs with first-class tools or MCP servers.
  - If you genuinely need filesystem access, write a new ADR proposing the
    invariant change and seek explicit approval — never bypass this gate.
—————————————————————————————————————————————————————————————————————————
EOF
    exit 1
fi

echo "check-no-fs: 0 violations across $(find crates -name '*.rs' 2>/dev/null | wc -l | tr -d ' ') Rust file(s)."
