#!/usr/bin/env bash
# scripts/check-doc-canonical-paths.sh — live-doc canonical path gate.
#
# Live user-facing docs (README, docs/architecture, docs/migrations) must
# reference the facade path (`entelix::Type`) rather than the underlying
# crate path (`entelix_core::Type` / `entelix_persistence::postgres::Type`
# etc.). The facade is the canonical 90% surface (ADR-0064); doc examples
# that reach through underlying crate paths teach users a non-canonical
# pattern.
#
# Catches the regression class where a sub-crate name appears in a doc
# code block or prose (e.g., `entelix_server::AgentRouterBuilder`) when
# the facade re-export (`entelix::AgentRouterBuilder`) is the canonical
# form. slice 98 cleaned the live-doc baseline; this gate freezes it.
#
# Scope:
#   IN  — README.md, PLAN.md, docs/architecture/*.md, docs/migrations/*.md
#   OUT — docs/adr/*.md (historical retrospectives + design records may
#         legitimately reference underlying crate paths), CHANGELOG.md
#         (historical entries pin past type names), docs/public-api/*.txt
#         (auto-generated), per-crate CLAUDE.md (sub-crate authors
#         intentionally use their own paths).
#
# Exit 0 = clean. Exit 1 = at least one underlying-crate path in a live doc.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Files in scope.
files=(README.md PLAN.md)
while IFS= read -r f; do files+=("${f}"); done < <(find docs/architecture docs/migrations -type f -name '*.md' 2>/dev/null)

# Sub-crate names → patterns (with `::` suffix, only path uses, not bare crate-name mentions).
patterns=(
    'entelix_core::'
    'entelix_runnable::'
    'entelix_prompt::'
    'entelix_graph::'
    'entelix_graph_derive::'
    'entelix_session::'
    'entelix_memory::'
    'entelix_memory_openai::'
    'entelix_memory_qdrant::'
    'entelix_memory_pgvector::'
    'entelix_graphmemory_pg::'
    'entelix_persistence::'
    'entelix_tools::'
    'entelix_mcp::'
    'entelix_mcp_chatmodel::'
    'entelix_cloud::'
    'entelix_policy::'
    'entelix_otel::'
    'entelix_server::'
    'entelix_agents::'
)

violations=""
for f in "${files[@]}"; do
    [[ -f "${f}" ]] || continue
    for p in "${patterns[@]}"; do
        hits="$(grep -n "${p}" "${f}" 2>/dev/null || true)"
        if [[ -n "${hits}" ]]; then
            while IFS= read -r line; do
                violations="${violations}${f}:${line}"$'\n'
            done <<< "${hits}"
        fi
    done
done

if [[ -n "${violations}" ]]; then
    count="$(printf '%s' "${violations}" | grep -c .)"
    echo "::error::live docs reference ${count} underlying-crate paths instead of facade canonical:"
    printf '%s' "${violations}"
    cat <<'EOF'

—————————————————————————————————————————————————————————————————————————
README, `docs/architecture/`, and `docs/migrations/` are user-facing
canonical references — they should demonstrate facade paths
(`entelix::Type` / `entelix::module::Type`), not underlying crate paths
(`entelix_persistence::postgres::Type`). The facade is the canonical 90%
surface per ADR-0064; doc examples reaching through underlying crate
paths teach users a non-canonical pattern.

Replace each `entelix_<crate>::Type` reference with `entelix::Type` (the
facade re-export). If the type is genuinely missing from the facade,
add the re-export to `crates/entelix/src/lib.rs` first and let
`scripts/check-facade-completeness.sh` confirm.

Historical retrospectives in `docs/adr/`, `CHANGELOG.md` entries, and
per-crate `CLAUDE.md` files are intentionally exempt — they pin past
type names or document the underlying crate's own surface.
—————————————————————————————————————————————————————————————————————————
EOF
    exit 1
fi

echo "check-doc-canonical-paths: ${#files[@]} live docs use only facade canonical paths."
