#!/usr/bin/env bash
# scripts/check-silent-fallback.sh — invariant #15 enforcement
# (CLAUDE.md §"No silent fallback" + ADR-0032).
#
# Bounds the number of `.unwrap_or` / `.unwrap_or_default` /
# `.unwrap_or_else` sites in the silent-fallback hot zones — codecs,
# transports, and the cost meter. Adding a new site requires an
# explicit `// silent-fallback-ok: <reason>` marker on the same line
# (the pattern grep ignores) or bumps the baseline below.
#
# The patterns themselves are not bugs — `accessor.unwrap_or("")` is
# the correct way to extract a missing JSON string field. The risk is
# regressions like "max_tokens.unwrap_or(4096)" or "cache_read_per_1k.
# unwrap_or(input/10)" sneaking in. A baselined count is the cheapest
# guard that catches both classes without false-positiving on the
# safe accessor pattern.
#
# Exit 0 = clean (count matches baseline). Exit 1 = drift.
#
# After an intentional addition, audit the new site, append the
# `// silent-fallback-ok: <reason>` marker (preferred) OR bump the
# baseline constant below.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if command -v rg >/dev/null 2>&1; then
    SEARCH=(rg --no-heading --line-number --color=never --type rust)
else
    SEARCH=(grep -rn --include='*.rs' --color=never -E)
fi

# Hot zones — every line of these files is part of the
# silent-fallback surface.
ZONES=(
    "crates/entelix-core/src/codecs/"
    "crates/entelix-core/src/transports/"
    "crates/entelix-cloud/src/bedrock/"
    "crates/entelix-cloud/src/vertex/"
    "crates/entelix-cloud/src/foundry/"
    "crates/entelix-policy/src/cost.rs"
)

PATTERN='\.unwrap_or(_default|_else)?\b'

# Per-zone audited-baseline lookup. Encoded as a `case` so the script
# stays compatible with bash 3.2 (no associative arrays on macOS).
baseline_for() {
    case "$1" in
        "crates/entelix-core/src/codecs/")     echo 0 ;;
        "crates/entelix-core/src/transports/") echo 2 ;;
        "crates/entelix-cloud/src/bedrock/")   echo 0 ;;
        "crates/entelix-cloud/src/vertex/")    echo 0 ;;
        "crates/entelix-cloud/src/foundry/")   echo 0 ;;
        "crates/entelix-policy/src/cost.rs")   echo 0 ;;
        *)                                     echo 0 ;;
    esac
}

violations=0
for zone in "${ZONES[@]}"; do
    candidates="$("${SEARCH[@]}" --regexp "${PATTERN}" "${zone}" 2>/dev/null || true)"
    # Drop lines that carry the audited-OK marker.
    if [[ -n "${candidates}" ]]; then
        candidates="$(printf '%s\n' "${candidates}" | grep -v 'silent-fallback-ok' || true)"
    fi
    if [[ -z "${candidates}" ]]; then
        actual=0
    else
        actual="$(printf '%s\n' "${candidates}" | grep -c '.')"
    fi
    expected="$(baseline_for "${zone}")"
    if (( actual > expected )); then
        diff=$(( actual - expected ))
        echo "::error::invariant #15 silent-fallback baseline exceeded in ${zone}"
        echo "  expected at most ${expected} unaudited \`unwrap_or*\` sites; found ${actual} (+${diff} new)"
        echo
        echo "  New / changed lines:"
        printf '%s\n' "${candidates}" | head -40 | sed 's/^/    /'
        echo
        echo "  Audit the addition. Either:"
        echo "    (1) Replace the silent fallback with \`Error::invalid_request\`"
        echo "        (mandatory vendor field) or \`ModelWarning::LossyEncode\`"
        echo "        + \`StopReason::Other{raw}\` (decode-time loss). See ADR-0032."
        echo "    (2) If the fallback is genuinely safe (e.g. a missing-string"
        echo "        accessor that defaults to \"\"), add the marker on the same line:"
        echo "          .unwrap_or(\"\") // silent-fallback-ok: missing optional field"
        echo "    (3) If the addition is wholesale and audited, bump BASELINE in"
        echo "        scripts/check-silent-fallback.sh."
        echo
        violations=$((violations + 1))
    elif (( actual < expected )); then
        # Surface dropping is a good thing but the baseline is now
        # stale — fail loudly so we re-baseline at the lower value.
        echo "::warning::invariant #15 silent-fallback baseline is stale in ${zone}"
        echo "  expected ${expected} sites; found only ${actual}. Lower the baseline."
        violations=$((violations + 1))
    fi
done

if (( violations > 0 )); then
    cat <<'EOF'

—————————————————————————————————————————————————————————————————————————
Invariant #15 — silent fallback prohibition (CLAUDE.md + ADR-0032).

Codecs, transports, and the cost meter must surface every information-loss
event through one of two channels:

  1. ModelWarning::LossyEncode { field, detail } — coerced values.
  2. StopReason::Other { raw } — unknown vendor reasons.

Vendor-mandatory IR fields (Anthropic max_tokens, …) are rejected at
encode time with Error::invalid_request. Missing decode signals surface
as `Other{raw:"missing"}` plus a LossyEncode warning.

Default-injecting a value (max_tokens.unwrap_or(4096),
cache_rate.unwrap_or(input/10), stopReason.unwrap_or(EndTurn)) is a bug
regardless of how reasonable the default looks.
—————————————————————————————————————————————————————————————————————————
EOF
    exit 1
fi

echo "check-silent-fallback: 0 violations (baseline matched in ${#ZONES[@]} zone(s))."
