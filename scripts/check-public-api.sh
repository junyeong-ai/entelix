#!/usr/bin/env bash
# scripts/check-public-api.sh
#
# Diff the live public API surface of each workspace crate against
# the committed baseline under `docs/public-api/<crate>.txt`. Any
# drift fails CI — refreeze deliberately via
# `scripts/freeze-public-api.sh` after the change is approved.
#
# This script is a soft gate when `cargo-public-api` isn't installed
# (skip with a warning) so contributors don't need the tool locally;
# CI installs it via `cargo install cargo-public-api` and enforces.
#
# Requirements:
#   cargo install cargo-public-api  (>= 0.51)

set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASELINE_DIR="${WORKSPACE_ROOT}/docs/public-api"

if ! command -v cargo-public-api > /dev/null; then
    echo "check-public-api: cargo-public-api not installed; skipping (install with 'cargo install cargo-public-api')."
    exit 0
fi

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

failures=()
for crate in "${CRATES[@]}"; do
    baseline="${BASELINE_DIR}/${crate}.txt"
    if [[ ! -f "${baseline}" ]]; then
        echo "check-public-api: baseline missing for ${crate} — run scripts/freeze-public-api.sh ${crate}"
        failures+=("${crate}")
        continue
    fi
    live="$(mktemp)"
    cargo public-api -p "${crate}" --simplified > "${live}"
    if ! diff -u "${baseline}" "${live}" > /dev/null; then
        echo "── drift in ${crate} ────────────────────"
        diff -u "${baseline}" "${live}" || true
        failures+=("${crate}")
    fi
    rm -f "${live}"
done

if (( ${#failures[@]} > 0 )); then
    echo
    echo "✗ public API drift detected in: ${failures[*]}"
    echo "  If the change is intentional, refreeze with:"
    echo "    scripts/freeze-public-api.sh ${failures[*]}"
    exit 1
fi

echo "✓ public API surface unchanged across all ${#CRATES[@]} baseline crates (facade excluded by design)."
