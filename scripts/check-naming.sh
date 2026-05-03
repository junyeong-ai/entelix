#!/usr/bin/env bash
# scripts/check-naming.sh — ADR-0010 enforcement (naming taxonomy)
#
# Forbids vague suffixes on public types and `get_*` accessors. Canonical
# rules: docs/adr/0010-naming-taxonomy.md + CLAUDE.md §"Naming Taxonomy".
#
# Exit 0 = clean. Exit 1 = at least one violation.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if command -v rg >/dev/null 2>&1; then
    SEARCH=(rg --no-heading --line-number --color=never --type rust)
else
    SEARCH=(grep -rn --include='*.rs' --color=never -E)
fi

violations=0

# ── Forbidden suffixes on public types (struct / enum / trait) ─────────────
# Targets the public surface only — `pub struct Foo*`, `pub enum Foo*`,
# `pub trait Foo*`. Tests are excluded by directory convention.
SUFFIX_RE='^[[:space:]]*pub[[:space:]]+(struct|enum|trait)[[:space:]]+[A-Za-z_][A-Za-z0-9_]*(Engine|Wrapper|Handler|Helper|Util)\b'
if hits="$("${SEARCH[@]}" --regexp "${SUFFIX_RE}" crates/ 2>/dev/null || true)"; then
    # Filter out tests directories.
    hits="$(printf '%s\n' "${hits}" | grep -v '/tests/' || true)"
    if [[ -n "${hits}" ]]; then
        echo "::error::ADR-0010 violation — forbidden type suffix (Engine | Wrapper | Handler | Helper | Util):"
        printf '%s\n' "${hits}"
        echo
        echo "Replace with a specific role (e.g., FooLoop, FooStrategy, FooAdapter, FooProcessor)."
        echo "See CLAUDE.md §\"Naming Taxonomy\" → \"Forbidden suffixes\"."
        echo
        violations=$((violations + 1))
    fi
fi

# ── *Service forbidden in core libs except for tower::Service impls ────────
# Per ADR-0010 (post-Phase-6.5): `*Service` is reserved for types that
# directly impl `tower::Service`. The check excludes any file that imports
# `tower::Service` or contains `impl ... Service<...>` referencing tower —
# false positives would prevent ecosystem-standard tower middleware naming.
SERVICE_RE='^[[:space:]]*pub[[:space:]]+(struct|enum|trait)[[:space:]]+[A-Za-z_][A-Za-z0-9_]*Service\b'
if hits="$("${SEARCH[@]}" --regexp "${SERVICE_RE}" crates/ 2>/dev/null || true)"; then
    hits="$(printf '%s\n' "${hits}" | grep -v '/tests/' || true)"
    if [[ -n "${hits}" ]]; then
        # For each remaining hit, verify the file uses tower::Service. If so,
        # it's a permitted middleware impl; drop from the violation list.
        filtered=""
        while IFS= read -r line; do
            [[ -z "${line}" ]] && continue
            file="${line%%:*}"
            if grep -qE '(use tower::Service|impl<[^>]*>[[:space:]]*Service<|impl[[:space:]]+Service<)' "${file}"; then
                continue
            fi
            filtered="${filtered}${line}"$'\n'
        done <<< "${hits}"
        if [[ -n "${filtered}" ]]; then
            echo "::error::ADR-0010 violation — '*Service' suffix without tower::Service impl:"
            printf '%s' "${filtered}"
            echo
            echo "Use *Manager (lifecycle), *Client (HTTP), or a more specific role."
            echo "(*Service is permitted only when the type directly impls tower::Service.)"
            echo
            violations=$((violations + 1))
        fi
    fi
fi

# ── get_* accessors forbidden (use bare name) ──────────────────────────────
GETTER_RE='^[[:space:]]*pub[[:space:]]+fn[[:space:]]+get_[A-Za-z_][A-Za-z0-9_]*[[:space:]]*\(&self'
if hits="$("${SEARCH[@]}" --regexp "${GETTER_RE}" crates/ 2>/dev/null || true)"; then
    hits="$(printf '%s\n' "${hits}" | grep -v '/tests/' || true)"
    if [[ -n "${hits}" ]]; then
        echo "::error::ADR-0010 violation — 'get_*' accessor (use bare name, e.g., name(&self) instead of get_name):"
        printf '%s\n' "${hits}"
        echo
        violations=$((violations + 1))
    fi
fi

# ── Builder verb-prefix enforcement ─────────────────────────────────────────
# Per ADR-0010 §"Builder verb-prefix exception":
#   - `with_<noun>` for configuration setters
#   - `add_<element>` for collection inserts
#   - `set_<role>` for designating named roles
#   - `register` for registry inserts (Result<Self>)
# Catches API drift where a new builder method slips in as
# `pub fn region(...)` instead of `pub fn with_region(...)`.
#
# Scoped to `impl <Type>Builder` (or `impl <generics> <Type>Builder<...>`)
# blocks only — value-level types (`Annotated`, etc.) accumulate via
# functional combinators (`reduced`, `mapped`, …) that intentionally
# don't follow the builder rule. Restricting by impl-receiver name keeps
# the signal high.
builder_violations="$(
  python3 - <<'PY'
import re, os, sys
impl_re = re.compile(r'^impl([\s<].*\b|\s+)([A-Za-z_][A-Za-z0-9_]*Builder)\b')
fn_re = re.compile(r'^\s*pub\s+fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((mut\s+)?self\b[^)]*\)\s*->\s*(Self\b|Result<Self\b)')
allowed = re.compile(r'^(with_|add_|set_|register)')

violations = []
for root, dirs, files in os.walk('crates'):
    if '/target/' in root or root.endswith('/target'):
        continue
    if '/tests' in root:
        continue
    for fname in files:
        if not fname.endswith('.rs'):
            continue
        path = os.path.join(root, fname)
        try:
            with open(path) as f: lines = f.readlines()
        except IOError:
            continue
        in_builder = False
        depth = 0
        for i, line in enumerate(lines, 1):
            if not in_builder and impl_re.search(line):
                in_builder = True
                depth = 0
            if in_builder:
                depth += line.count('{') - line.count('}')
                if depth <= 0 and ('}' in line or '{' in line):
                    # opening line may have braces; only exit when truly back to 0
                    if depth <= 0 and '{' not in line:
                        in_builder = False
                        continue
            if in_builder:
                m = fn_re.match(line)
                if m and not allowed.match(m.group(1)):
                    violations.append(f"{path}:{i}:{line.rstrip()}")
for v in violations: print(v)
PY
)"
if [[ -n "${builder_violations}" ]]; then
    echo "::error::ADR-0010 violation — builder method missing verb prefix (with_/add_/set_/register):"
    printf '%s\n' "${builder_violations}"
    echo
    echo "See CLAUDE.md §\"Naming Taxonomy\" → \"Builder verb-prefix exception\"."
    echo "  with_<noun>(self, x)     — configuration setter on a single target"
    echo "  add_<element>(self, …)   — append to an internal collection"
    echo "  set_<role>(self, …)      — designate one element as a named role"
    echo "  register(self, …)        — registry insert (Result<Self>)"
    echo
    violations=$((violations + 1))
fi

# ── with_*(&self) — borrow disguised as builder ────────────────────────────
# Per ADR-0010 §"Method names" + §"Builder verb-prefix exception":
#   - `with_<noun>(self, x) -> Self` — builder option setter (consumes self)
#   - `name(&self) -> &T`            — bare accessor (never `get_name`, never `with_name`)
#
# Any `with_*(&self, …)` is wrong by construction: the verb prefix promises
# configuration but the receiver shape is a borrow. Whether the return is
# `&T` (borrow accessor) or `Result<Self>` (derivative-view constructor),
# the call site reads as mutation when in fact nothing changes — that
# mismatch is the ban. Use a bare accessor (`region(&self) -> &str`) for
# borrows; use a domain verb (`restricted_to(&self, …) -> Result<Self>`,
# `filter(&self, …) -> Self`) for derivative views.
MASQUERADE_RE='^[[:space:]]*pub[[:space:]]+(const[[:space:]]+)?fn[[:space:]]+with_[A-Za-z_][A-Za-z0-9_]*[[:space:]]*\(&(mut[[:space:]]+)?self'
if hits="$("${SEARCH[@]}" --regexp "${MASQUERADE_RE}" crates/ 2>/dev/null || true)"; then
    hits="$(printf '%s\n' "${hits}" | grep -v '/tests/' || true)"
    if [[ -n "${hits}" ]]; then
        echo "::error::ADR-0010 violation — 'with_*(&self, …)' borrow disguised as builder:"
        printf '%s\n' "${hits}"
        echo
        echo "  with_region(&self) -> &str           ← bare accessor"
        echo "  with_only(&self, …) -> Result<Self>  ← domain verb"
        echo "                                          (restricted_to / filter / etc.)"
        echo
        echo "If the method consumes self (returns Self), use '(self, …) -> Self'."
        echo
        violations=$((violations + 1))
    fi
fi

# ── ctx parameter placement enforcement ────────────────────────────────────
# Per ADR-0010 §"Parameter ordering — ctx placement":
#   - ctx-first: memory/persistence backends — Store, VectorStore, GraphMemory,
#     Checkpointer, SessionLog, BufferMemory, SummaryMemory, EntityMemory,
#     EpisodicMemory, SemanticMemory, ConsolidatingBufferMemory.
#   - ctx-last:  computation/dispatch — Tool, Embedder, Retriever, Reranker,
#     Runnable, Approver, AgentObserver, Sandbox.
# Catches API drift where a new method puts ctx in the wrong position.
ctx_violations="$(
  python3 - <<'PY'
import re, os, sys
ctx_first_traits = {
  'Store': 'crates/entelix-memory/src/traits.rs',
  'VectorStore': 'crates/entelix-memory/src/traits.rs',
  'GraphMemory': 'crates/entelix-memory/src/graph.rs',
  'Checkpointer': 'crates/entelix-graph/src/checkpointer.rs',
  'SessionLog': 'crates/entelix-session/src/log.rs',
}
ctx_last_traits = {
  'Tool': 'crates/entelix-core/src/tools/mod.rs',
  'Embedder': 'crates/entelix-memory/src/traits.rs',
  'Retriever': 'crates/entelix-memory/src/traits.rs',
  'Reranker': 'crates/entelix-memory/src/traits.rs',
  'Runnable': 'crates/entelix-runnable/src/runnable.rs',
  'AgentObserver': 'crates/entelix-agents/src/agent/observer.rs',
  'Approver': 'crates/entelix-agents/src/agent/approver.rs',
  'Sandbox': 'crates/entelix-core/src/sandbox.rs',
  'CostCalculator': 'crates/entelix-core/src/cost.rs',
  'ToolCostCalculator': 'crates/entelix-core/src/cost.rs',
  'EmbeddingCostCalculator': 'crates/entelix-memory/src/metered.rs',
  'Summarizer': 'crates/entelix-memory/src/consolidating.rs',
}
memory_pattern_files = [
  'crates/entelix-memory/src/buffer.rs',
  'crates/entelix-memory/src/summary.rs',
  'crates/entelix-memory/src/entity.rs',
  'crates/entelix-memory/src/episodic.rs',
  'crates/entelix-memory/src/semantic.rs',
  'crates/entelix-memory/src/consolidating.rs',
]
def parse_args(s):
  parts, depth, cur = [], 0, ''
  for ch in s:
    if ch in '<([': depth += 1
    elif ch in '>)]': depth -= 1
    if ch == ',' and depth == 0:
      t = cur.strip()
      if t: parts.append(t)
      cur = ''
    else: cur += ch
  t = cur.strip()
  if t: parts.append(t)
  return parts
def find_violations(file, trait_or_none, expected, src=None, skip_methods=None):
  if src is None:
    if not os.path.exists(file): return []
    with open(file) as f: src = f.read()
  out = []
  if trait_or_none is None:
    block = src
  else:
    tr = re.compile(rf'(?:pub )?trait {trait_or_none}\b[^{{]*\{{', re.MULTILINE)
    m = tr.search(src)
    if not m: return []
    start = m.end(); depth = 1; i = start
    while depth > 0 and i < len(src):
      if src[i] == '{': depth += 1
      elif src[i] == '}': depth -= 1
      i += 1
    block = src[start:i]
  for m in re.finditer(r'(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*\(([^)]+)\)', block, re.DOTALL):
    name, args = m.group(1), m.group(2)
    if skip_methods and name in skip_methods: continue
    if 'ExecutionContext' not in args: continue
    args_split = parse_args(args)
    non_self = [a for a in args_split if not re.match(r'^&?(mut\s+)?self\b', a)]
    ctx_pos = next((i for i, a in enumerate(non_self) if 'ExecutionContext' in a), None)
    if ctx_pos is None: continue
    if expected == 'first' and ctx_pos != 0:
      out.append(f'{file}: {trait_or_none or "(impl)"}::{name} expected ctx-first, got {ctx_pos} of {len(non_self)}')
    elif expected == 'last' and ctx_pos != len(non_self) - 1:
      out.append(f'{file}: {trait_or_none or "(impl)"}::{name} expected ctx-last, got {ctx_pos} of {len(non_self)}')
  return out
all_v = []
# Per-file set of method names already governed by an explicit trait
# entry — those are checked once below and must be excluded from the
# memory-pattern-files glob pass so the glob does not double-check
# (and disagree with) them.
explicit_methods_by_file = {}
def trait_methods(file, trait):
  if not os.path.exists(file): return set()
  with open(file) as f: src = f.read()
  tr = re.compile(rf'(?:pub )?trait {trait}\b[^{{]*\{{', re.MULTILINE)
  m = tr.search(src)
  if not m: return set()
  start = m.end(); depth = 1; i = start
  while depth > 0 and i < len(src):
    if src[i] == '{': depth += 1
    elif src[i] == '}': depth -= 1
    i += 1
  block = src[start:i]
  return {m.group(1) for m in re.finditer(r'(?:async\s+)?fn\s+(\w+)\s*\(', block)}
for trait, file in ctx_first_traits.items():
  all_v += find_violations(file, trait, 'first')
  explicit_methods_by_file.setdefault(file, set()).update(trait_methods(file, trait))
for trait, file in ctx_last_traits.items():
  all_v += find_violations(file, trait, 'last')
  explicit_methods_by_file.setdefault(file, set()).update(trait_methods(file, trait))
for file in memory_pattern_files:
  all_v += find_violations(file, None, 'first', skip_methods=explicit_methods_by_file.get(file, set()))
for v in all_v: print(v)
PY
)"
if [[ -n "${ctx_violations}" ]]; then
    echo "::error::ADR-0010 violation — ctx parameter placement (must match naming.md spec):"
    printf '%s\n' "${ctx_violations}"
    echo
    echo "ctx-first: memory/persistence backends — '(&self, ctx, scope, payload...)'."
    echo "ctx-last:  computation/dispatch — '(&self, input, ..., ctx)'."
    echo "See CLAUDE.md §\"Naming Taxonomy\" → \"ctx parameter ordering — split convention\"."
    echo
    violations=$((violations + 1))
fi

if (( violations > 0 )); then
    cat <<'EOF'

—————————————————————————————————————————————————————————————————————————
Naming taxonomy is the consistency contract — see ADR-0010 + CLAUDE.md.
Each suffix carries semantics; vague names erode the API surface.

Replacement guide:
  *Engine    → say what it does    (OrchestrationLoop, RetryStrategy)
  *Wrapper   → say what it wraps   (ToolToRunnableAdapter)
  *Handler   → say what it handles (RequestProcessor, EventConsumer)
  *Helper    → fold into module    (just put the fn in foo::compute)
  *Util      → ditto
  *Service   → *Manager (lifecycle) / *Client (HTTP)
  get_X()    → X()
—————————————————————————————————————————————————————————————————————————
EOF
    exit 1
fi

echo "check-naming: 0 violations."
