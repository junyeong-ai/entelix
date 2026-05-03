#!/usr/bin/env bash
# scripts/check-feature-matrix.sh — facade feature-isolation gate.
#
# Verifies that every facade feature compiles in isolation. Catches
# the class of regression where `entelix`'s feature flag enables a
# sub-crate dep but forgets to pass through the corresponding
# sub-crate feature, e.g. `postgres = ["dep:entelix-persistence"]`
# silently dropping the underlying `entelix-persistence/postgres`
# pass-through and leaving `entelix_persistence::postgres` module
# missing. cargo's default `--all-features` masks this because the
# union of all features happens to enable the right combination.
#
# Exit 0 = clean. Exit 1 = at least one feature fails to compile in
# isolation.
#
# Runtime: ~1-2 minutes on a warm cache. CI already pays this cost
# on the ` --all-features` build; the marginal cost here is the
# per-feature `cargo check` rebuild which is fast on a warm target/.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Every facade feature listed in `crates/entelix/Cargo.toml::[features]`
# except `default` (empty) and `full` (union — covered separately).
FEATURES=(
    mcp
    mcp-chatmodel
    postgres
    redis
    otel
    aws
    gcp
    azure
    policy
    server
    embedders-openai
    vectorstores-qdrant
    vectorstores-pgvector
    graphmemory-pg
)

failures=()

# Per-feature isolation check.
for f in "${FEATURES[@]}"; do
    if ! cargo check -p entelix --no-default-features --features "${f}" --quiet 2>/dev/null; then
        failures+=("${f}")
    fi
done

# `full` should be a clean union of every feature.
if ! cargo check -p entelix --no-default-features --features full --quiet 2>/dev/null; then
    failures+=("full")
fi

# `--no-default-features` (empty feature set) must also build.
if ! cargo check -p entelix --no-default-features --quiet 2>/dev/null; then
    failures+=("no-default-features")
fi

if (( ${#failures[@]} > 0 )); then
    echo "::error::facade feature isolation broken — features that fail to compile alone:"
    for f in "${failures[@]}"; do
        echo "  - ${f}"
    done
    echo
    cat <<'EOF'
—————————————————————————————————————————————————————————————————————————
facade `entelix` features must compile in isolation. The most common
cause: `feature = ["dep:foo"]` enables the dep but omits the pass-through
to `foo`'s own internal feature, e.g.

  postgres = ["dep:entelix-persistence"]                              # broken
  postgres = ["dep:entelix-persistence", "entelix-persistence/postgres"]  # ok

Re-run a single failing feature locally to see the compiler error:

  cargo check -p entelix --no-default-features --features <FEATURE>
—————————————————————————————————————————————————————————————————————————
EOF
    exit 1
fi

echo "check-feature-matrix: ${#FEATURES[@]} facade features + 'full' + 'no-default-features' all compile in isolation."
