#!/usr/bin/env bash
# scripts/freeze-public-api.sh
#
# Generate per-crate public API baseline snapshots into
# `docs/public-api/<crate>.txt`. Run this only when you intentionally
# want to refreeze the surface (e.g. after a deliberate API change
# that you've documented in an ADR).
#
# CI invokes `scripts/check-public-api.sh` instead — that one fails
# on any drift between the live surface and these baselines, so
# accidental breakage shows up at PR time rather than at release.
#
# Requirements:
#   cargo install cargo-public-api  (>= 0.51)
#
# Usage:
#   scripts/freeze-public-api.sh             # refreeze every crate
#   scripts/freeze-public-api.sh entelix-mcp  # refreeze one crate

set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${WORKSPACE_ROOT}/docs/public-api"
mkdir -p "${OUT_DIR}"

# Workspace members that ship a public surface. The `entelix` facade
# crate is intentionally excluded — its surface is just re-exports of
# the sub-crates, which already have their own baselines.
CRATES=(
    entelix-core
    entelix-runnable
    entelix-prompt
    entelix-graph
    entelix-graph-derive
    entelix-session
    entelix-memory
    entelix-memory-openai
    entelix-memory-pgvector
    entelix-memory-qdrant
    entelix-graphmemory-pg
    entelix-persistence
    entelix-tools
    entelix-mcp
    entelix-mcp-chatmodel
    entelix-cloud
    entelix-policy
    entelix-otel
    entelix-server
    entelix-agents
)

if [[ $# -gt 0 ]]; then
    # Refreeze a single crate.
    CRATES=("$@")
fi

for crate in "${CRATES[@]}"; do
    echo "── ${crate} ─────────────────────────────"
    out="${OUT_DIR}/${crate}.txt"
    cargo public-api -p "${crate}" --simplified > "${out}"
    lines=$(wc -l < "${out}" | tr -d ' ')
    echo "  → ${out} (${lines} lines)"
done

echo
echo "✓ All baselines refreshed under docs/public-api/"
echo "  Commit any changes; CI's check-public-api.sh diffs against these files."
