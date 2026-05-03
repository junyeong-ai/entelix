#!/usr/bin/env bash
# scripts/check-dead-deps.sh — workspace.dependencies dead-entry gate.
#
# Flags entries declared in `Cargo.toml::[workspace.dependencies]` that
# zero crates inherit (`<crate>/Cargo.toml` 안 `name = { workspace = true }`
# reference 0). Dead deps inflate `Cargo.lock` resolution time, leak into
# `cargo audit` reports, and surface as confusing "where is this used?"
# noise during dependency review.
#
# Exit 0 = clean. Exit 1 = at least one dead entry.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Internal entelix-* refs are intentionally listed in
# `[workspace.dependencies]` even when no consumer exists yet — they
# pin the workspace lockstep version. Skip them.
INTERNAL_PREFIX='^entelix-'

# Extract every workspace.dependencies key. Format variants:
#   foo = "1.0"
#   foo = { version = "1", features = [...] }
keys="$(
  awk '
    /^\[workspace\.dependencies\]/ { in_block = 1; next }
    /^\[/ { in_block = 0 }
    in_block && /^[a-zA-Z0-9_-]+[[:space:]]*=/ {
      n = $1
      sub(/^[[:space:]]*/, "", n)
      sub(/[[:space:]]*=.*$/, "", n)
      print n
    }
  ' Cargo.toml
)"

dead=()
for key in ${keys}; do
    if [[ "${key}" =~ ${INTERNAL_PREFIX} ]]; then
        continue
    fi
    # Look for `<key> = { workspace = true ... }` or
    # `<key> = { workspace = true, ... }` in any sub-crate Cargo.toml.
    if ! rg -q "^${key}[[:space:]]*=[[:space:]]*\\{[[:space:]]*workspace[[:space:]]*=[[:space:]]*true" crates/*/Cargo.toml 2>/dev/null; then
        dead+=("${key}")
    fi
done

if (( ${#dead[@]} > 0 )); then
    echo "::error::workspace.dependencies declares entries no crate inherits:"
    for d in "${dead[@]}"; do
        echo "  - ${d}"
    done
    echo
    cat <<'EOF'
—————————————————————————————————————————————————————————————————————————
Each entry in `Cargo.toml::[workspace.dependencies]` should be inherited
by at least one sub-crate via `<crate-name> = { workspace = true }`. Dead
entries inflate Cargo.lock resolution and confuse dependency review.

Either:
  (a) Remove the entry from workspace.dependencies, or
  (b) Wire it into the crate that needs it via `[dependencies]` or
      `[dev-dependencies]` with `{ workspace = true }`.

Internal `entelix-*` refs are exempt — they pin workspace lockstep.
—————————————————————————————————————————————————————————————————————————
EOF
    exit 1
fi

count="$(printf '%s\n' "${keys}" | grep -v -E '^entelix-|^$' | wc -l | xargs)"
echo "check-dead-deps: ${count} workspace.dependencies entries — every one is inherited by at least one crate."
