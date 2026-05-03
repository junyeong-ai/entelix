#!/usr/bin/env bash
# scripts/check-supply-chain.sh — supply-chain hardening gate.
#
# Two layers:
#
# 1. **`cargo audit`** — RustSec advisory database (live fetch). Fails
#    on any unfixed CVE that doesn't have an explicit `[advisories.ignore]`
#    entry in `deny.toml` (each ignore must carry a comment justifying
#    why it's an acceptable transitive risk for entelix).
#
# 2. **`cargo deny check`** — three-layer supply-chain policy:
#    - advisories: same RustSec database, plus unmaintained / unsound /
#      yanked detection.
#    - licenses: explicit allow-list of OSI-approved permissive
#      licenses (MIT / Apache-2.0 / BSD-{2,3} / ISC / BSL-1.0 /
#      Unicode-3.0 / MPL-2.0). Anything else fails — protects against
#      AGPL / GPL / CC-BY-NC drift.
#    - bans: forbids `landlock` / `seatbelt` / `tree-sitter` from the
#      dependency graph (invariant 9 dependency-graph layer; companion
#      to `scripts/check-no-fs.sh` which catches first-party imports).
#
# Both tools are required dev-tooling. Install via:
#   cargo install cargo-audit cargo-deny --locked
#
# Exit 0 = clean. Exit 1 = at least one CVE / disallowed license / banned dep.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

failed=0

if ! command -v cargo-audit >/dev/null 2>&1; then
    echo "::error::cargo-audit not installed; run: cargo install cargo-audit --locked"
    exit 1
fi

if ! command -v cargo-deny >/dev/null 2>&1; then
    echo "::error::cargo-deny not installed; run: cargo install cargo-deny --locked"
    exit 1
fi

# cargo audit reads `.cargo/audit.toml` for the ignore list (cargo-audit
# does not consume `deny.toml` directly). The single source of truth for
# the ignored-advisories rationale stays in `deny.toml` — `audit.toml`
# mirrors only the IDs.
echo "── cargo audit"
if ! cargo audit --deny warnings --quiet >/dev/null 2>&1; then
    cargo audit --deny warnings 2>&1 | tail -50
    failed=$((failed + 1))
fi

echo "── cargo deny check"
deny_output="$(cargo deny check 2>&1)"
# `cargo deny` exits 0 when there are only warnings (duplicate-version,
# advisory-not-detected, license-not-encountered). Only escalate when
# the policy itself rejects something — confirmed by the four-line
# summary `advisories ok, bans ok, licenses ok, sources ok`.
if ! grep -qF 'advisories ok, bans ok, licenses ok, sources ok' <<< "${deny_output}"; then
    printf '%s\n' "${deny_output}" | tail -30
    failed=$((failed + 1))
fi

if (( failed > 0 )); then
    cat <<'EOF'

—————————————————————————————————————————————————————————————————————————
Supply-chain gate is the workspace's defense against CVE drift, license
contamination, and dependency-graph regressions of invariant 9
(`landlock` / `seatbelt` / `tree-sitter` must not appear even
transitively).

Each unfixed CVE that lands here must either:
  (a) Be patched by upgrading the offending crate, or
  (b) Be added to `deny.toml [advisories.ignore]` with a comment
      explaining why the vulnerability is structurally absent in
      entelix's usage (e.g., "sqlx-mysql pulled transitively but
      no MySQL connection ever opens").

Disallowed licenses must be resolved by either:
  (a) Pinning to a different version of the offending crate that
      uses an allowed license, or
  (b) Adding the license to `deny.toml [licenses.allow]` after
      legal review.
—————————————————————————————————————————————————————————————————————————
EOF
    exit 1
fi

echo "check-supply-chain: cargo audit + cargo deny check both clean."
