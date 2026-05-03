#!/usr/bin/env bash
# scripts/check-surface-hygiene.sh — public-API surface hygiene gate.
#
# Enforces three structural rules across `crates/*/src/`:
#   1. Every `pub enum` (excluding `#[non_exhaustive]`-marked sealed FSMs
#      that explicitly opt out via `// SEALED-ENUM` annotation) must be
#      `#[non_exhaustive]`. SemVer hazard prevention.
#   2. Every error-typed variant carrying an inner error type must annotate
#      the source field with `#[source]` or `#[from]`. Diagnostic-chain
#      preservation.
#   3. No `pub fn get_*` accessor (already covered by check-naming.sh —
#      mirrored here so a single failing run surfaces the full picture).
#
# Exit 0 = clean. Exit 1 = at least one violation.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

violations=0

# ── Rule 1: pub enum without #[non_exhaustive] ────────────────────────────
missing_non_exhaustive="$(python3 - <<'PY'
import re, os
hits = []
DECL = re.compile(r'^[ \t]*pub enum (\w+)', re.MULTILINE)
for root, dirs, files in os.walk("crates"):
    if "/target/" in root or root.endswith("/target"):
        continue
    if "/tests/" in root:
        continue
    for f in files:
        if not f.endswith(".rs"):
            continue
        path = os.path.join(root, f)
        with open(path) as fh:
            lines = fh.readlines()
        for idx, line in enumerate(lines):
            m = DECL.match(line)
            if not m:
                continue
            name = m.group(1)
            # Walk backward through contiguous attr/doc block.
            i = idx - 1
            block = []
            while i >= 0:
                s = lines[i].strip()
                if s == "":
                    break
                if s.startswith('#[') or s.startswith('///') or s.startswith('//!'):
                    block.append(s)
                    i -= 1
                    continue
                break
            if any('non_exhaustive' in b for b in block):
                continue
            hits.append(f"{path}:{idx+1}:pub enum {name}")
for h in hits:
    print(h)
PY
)"
if [[ -n "${missing_non_exhaustive}" ]]; then
    echo "::error::Public enum missing #[non_exhaustive] (SemVer hazard):"
    printf '%s\n' "${missing_non_exhaustive}"
    echo
    echo "Add #[non_exhaustive] to every public enum so future variant additions are non-breaking."
    echo
    violations=$((violations + 1))
fi

# ── Rule 1b: Tier-1 user-facing structs require #[non_exhaustive] ─────────
# Hand-curated allow-list of struct names that operators construct via
# `Type { field: value, ... }` literally. Adding a new field to any of
# these post-1.0 is a SemVer break unless the struct is `#[non_exhaustive]`.
#
# Open data carriers (e.g., `Message`, `Document`, `EntityRecord`,
# `Episode`) are intentionally OPEN — their fields ARE the API and
# operators construct them literally. The line is "config struct" vs
# "data carrier" — config evolves with new knobs (non_exhaustive),
# data carriers evolve with new helper methods or new ContentPart
# variants (open struct).
TIER1_CONFIG_STRUCTS=(
    "ChatModelConfig"
    "SessionGraph"
    "Usage"
    "GraphHop"
)
struct_violations=()
for s in "${TIER1_CONFIG_STRUCTS[@]}"; do
    hits="$(grep -rn --include='*.rs' -B1 "^pub struct ${s}\b" crates/ 2>/dev/null \
        | grep -v '/tests/' \
        | grep -v 'non_exhaustive' \
        | grep "^pub struct ${s}\b" || true)"
    if [[ -n "${hits}" ]]; then
        struct_violations+=("${hits}")
    fi
done
if (( ${#struct_violations[@]} > 0 )); then
    echo "::error::Tier-1 config struct missing #[non_exhaustive] (SemVer hazard):"
    for v in "${struct_violations[@]}"; do echo "${v}"; done
    echo
    echo "Add #[non_exhaustive] to the listed struct, or remove it from"
    echo "TIER1_CONFIG_STRUCTS in scripts/check-surface-hygiene.sh if the"
    echo "design has shifted to an open-data-carrier shape."
    echo
    violations=$((violations + 1))
fi

# ── Rule 2: error variants dropping the source chain ──────────────────────
# Heuristic: a variant `Variant(SomeError)` or `Variant { ..., src: SomeError }`
# inside a `#[derive(thiserror::Error)]` enum must carry `#[source]` or
# `#[from]` on the inner error field. We grep for `.map_err(|e| ...(e.to_string()))`
# inside `.rs` files as a tell-tale "source dropped" signal.
chain_drops="$(grep -rn --include='*.rs' --color=never \
    -E 'map_err\(\|[A-Za-z_]+\| [A-Za-z_:]+::[A-Za-z_]+\([a-z_]+\.to_string\(\)\)' \
    crates/ 2>/dev/null | grep -v '/tests/' || true)"
if [[ -n "${chain_drops}" ]]; then
    echo "::error::Error chain dropped via .to_string() in map_err — preserve source:"
    printf '%s\n' "${chain_drops}"
    echo
    echo "Replace with a typed ctor that stores the source as #[source] or #[from]."
    echo "Example: McpError::network(err) instead of McpError::Network(err.to_string())."
    echo
    violations=$((violations + 1))
fi

# ── Rule 3: get_* accessors (mirror of check-naming.sh) ───────────────────
if command -v rg >/dev/null 2>&1; then
    SEARCH=(rg --no-heading --line-number --color=never --type rust)
else
    SEARCH=(grep -rn --include='*.rs' --color=never -E)
fi
GETTER_RE='^[[:space:]]*pub[[:space:]]+fn[[:space:]]+get_[A-Za-z_][A-Za-z0-9_]*[[:space:]]*\(&self'
if hits="$("${SEARCH[@]}" --regexp "${GETTER_RE}" crates/ 2>/dev/null || true)"; then
    hits="$(printf '%s\n' "${hits}" | grep -v '/tests/' || true)"
    if [[ -n "${hits}" ]]; then
        echo "::error::ADR-0010 violation — 'get_*' accessor:"
        printf '%s\n' "${hits}"
        echo
        violations=$((violations + 1))
    fi
fi

if (( violations > 0 )); then
    exit 1
fi

echo "check-surface-hygiene: 0 violations."
