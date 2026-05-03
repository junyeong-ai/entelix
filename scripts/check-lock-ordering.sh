#!/usr/bin/env bash
# scripts/check-lock-ordering.sh — CLAUDE.md §"Lock ordering" enforcement.
#
# Verifies the workspace clippy `await_holding_lock` and
# `await_holding_refcell_ref` lints are pinned at deny level. Together
# they statically forbid any guard from `parking_lot::Mutex`,
# `std::sync::Mutex`, `tokio::sync::Mutex`, `RefCell`, etc. crossing an
# `.await` point — the failure mode CLAUDE.md §"Lock ordering" calls
# out ("Never hold any lock across `.await` on a user-supplied future").
#
# A contributor downgrading either lint to "warn" is the only way the
# gate could regress, and that drift is exactly what this script
# catches.
#
# The lock-acquisition *order* across distinct mutexes is not a
# property the type system or clippy can decide. CLAUDE.md documents
# the canonical order (`tenant > session > checkpoint > memory >
# tool_registry > orchestrator`); reviewers verify it during code
# review. A grep-shaped "double-acquire" heuristic would generate
# false positives without parsing Rust, so this script does not
# attempt one.
#
# Exit 0 = clean. Exit 1 = at least one violation.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

violations=0

required_lints=(
    'await_holding_lock = "deny"'
    'await_holding_refcell_ref = "deny"'
)
for lint in "${required_lints[@]}"; do
    if ! grep -qF "${lint}" Cargo.toml; then
        echo "::error::Missing required clippy lint in workspace Cargo.toml: ${lint}"
        echo "  CLAUDE.md §\"Lock ordering\" requires this lint at deny level."
        echo "  Add it back under [workspace.lints.clippy]."
        violations=$((violations + 1))
    fi
done

if (( violations > 0 )); then
    echo
    echo "check-lock-ordering: ${violations} violation(s)."
    exit 1
fi
echo "check-lock-ordering: workspace clippy await-holding lints pinned at deny level."
