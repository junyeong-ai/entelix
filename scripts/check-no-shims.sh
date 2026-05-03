#!/usr/bin/env bash
# scripts/check-no-shims.sh — invariant #14 enforcement
# (CLAUDE.md §"Engineering" + ADR-0010 §"No backwards-compatibility shims").
#
# Forbids the shim shapes that invariant #14 names explicitly:
#
#   - `#[deprecated]` / `#[deprecated(...)]` annotations
#   - `// deprecated` line comments
#   - `// removed for backcompat` orphan markers (left in place
#     when code was deleted with a "we used to expose X" comment)
#   - `pub use OldName as NewName` re-exports that exist solely to
#     keep an old call path compiling
#
# Audit lifecycle:
# - When a name / type / signature changes, delete the old in the
#   *same* PR. No deprecation period, no shim.
# - The companion `pub use entelix_graph_derive::StateMerge` style
#   re-export is fine because it doesn't rename anything (the macro
#   keeps its original name); only `as Old` renames are flagged.
#
# Exit 0 = clean. Exit 1 = at least one shim found.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if command -v rg >/dev/null 2>&1; then
    SEARCH=(rg --no-heading --line-number --color=never --type rust)
else
    SEARCH=(grep -rn --include='*.rs' --color=never -E)
fi

# Each pattern is hunted with the same SEARCH backend; collect all
# hits then report a unified diagnostic.
declare -a PATTERNS=(
    '#\[deprecated(\(|\])'                                 # #[deprecated] / #[deprecated(...)]
    '//\s*deprecated\b'                                     # // deprecated <anything>
    '//\s*removed\s+for\s+(backcompat|backwards|compat|migration)'  # // removed for backcompat
    '//\s*formerly\s+'                                      # // formerly named X / // formerly in module Y
    '\bpub\s+use\s+[^;]*\s+as\s+[A-Z][a-zA-Z0-9_]*Old\b'    # pub use X as YOld
)

violations=0
for pattern in "${PATTERNS[@]}"; do
    hits="$("${SEARCH[@]}" --regexp "${pattern}" crates/ 2>/dev/null || true)"
    if [[ -n "${hits}" ]]; then
        echo "::error::invariant #14 shim found — pattern '${pattern}'"
        printf '%s\n' "${hits}" | sed 's/^/    /'
        echo
        violations=$((violations + 1))
    fi
done

if (( violations > 0 )); then
    cat <<'EOF'

—————————————————————————————————————————————————————————————————————————
Invariant #14 — no backwards-compatibility shims (CLAUDE.md + ADR-0010).

When a name, type, or signature changes, delete the old in the same PR.
No `// deprecated`, no `pub use OldName as NewName`, no fallback
constructors. The standing project rule is "처음부터 이렇게 설계된
것처럼 흔적 0" — deprecation periods are noise the codebase does not
keep.

If a flagged site is genuinely not a shim (e.g. an external library's
deprecation note inside a doc-comment block), narrow the source to the
specific pattern and audit individually.
—————————————————————————————————————————————————————————————————————————
EOF
    exit 1
fi

echo "check-no-shims: 0 violations across ${#PATTERNS[@]} pattern(s)."
