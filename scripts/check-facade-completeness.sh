#!/usr/bin/env bash
# scripts/check-facade-completeness.sh — facade re-export coverage gate.
#
# Verifies that every `pub use` item in every sub-crate's `lib.rs` is
# reachable through the `entelix` facade — either via a flat
# re-export (`pub use entelix_X::Type`), via a sub-module re-export
# (`pub use entelix_persistence::postgres::Type`), or via a parent-
# module re-export (`pub use entelix_core::tools;` covers everything
# the `tools` module names).
#
# Catches the regression class where a sub-crate adds a new public
# type and forgets to wire it into the facade — every consumer who
# does `use entelix::*` then has to learn the underlying crate path,
# defeating the facade's "canonical 90% surface" contract (ADR-0064).
#
# Items that are intentionally excluded (advanced runtime internals
# like `entelix_graph::Dispatch`/`scatter`/`FinalizingStream` that
# operators reach via the underlying crate path) live in
# `scripts/facade-excludes.txt`. Adding a new exemption requires the
# same review discipline as adding a `// silent-fallback-ok:` marker.
#
# Exit 0 = clean. Exit 1 = at least one un-exempt missing item.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

EXCLUDES_FILE="scripts/facade-excludes.txt"

violations="$(
  python3 - <<PY
import os, re, sys

EXCLUDES_FILE = "${EXCLUDES_FILE}"

def parse_facade(path):
    """Returns (flat: dict crate -> set, modules: dict crate -> set)."""
    with open(path) as f: src = f.read()
    flat, modules = {}, {}
    # Multi-item: pub use entelix_X::{a, b as alias, ...};
    for m in re.finditer(r'pub use (entelix_\w+)::\{([^}]+)\};', src, re.DOTALL):
        crate = m.group(1)
        for it in m.group(2).split(','):
            it = it.strip()
            if not it: continue
            name = it.split(' as ')[0].strip()
            flat.setdefault(crate, set()).add(name)
    # Sub-module multi-item: pub use entelix_X::sub::{a, b, ...};
    for m in re.finditer(r'pub use (entelix_\w+)::(\w+)::\{([^}]+)\};', src, re.DOTALL):
        crate = m.group(1)
        for it in m.group(3).split(','):
            it = it.strip()
            if not it: continue
            name = it.split(' as ')[0].strip()
            flat.setdefault(crate, set()).add(name)
    # Single sub-module: pub use entelix_X::sub::Type;
    for m in re.finditer(r'pub use (entelix_\w+)::(\w+)::([\w_]+);', src):
        crate = m.group(1)
        flat.setdefault(crate, set()).add(m.group(3))
    # Top-level single: pub use entelix_X::Item;
    # Lowercase = module re-export; PascalCase / SCREAMING = item.
    for m in re.finditer(r'pub use (entelix_\w+)::([\w_]+);', src, re.MULTILINE):
        crate = m.group(1)
        item = m.group(2)
        if item[0].islower() and '_' not in item:  # likely module
            modules.setdefault(crate, set()).add(item)
        else:
            flat.setdefault(crate, set()).add(item)
    return flat, modules

def parse_subcrate(path):
    """Returns list of (submodule_name, [item_names]).
    submodule_name == None for crate-to-crate re-exports (e.g.
    entelix-graph re-exporting entelix-graph-derive)."""
    if not os.path.exists(path): return []
    with open(path) as f: src = f.read()
    blocks = []
    # pub use submod::{a, b, ...};   (single-segment path)
    for m in re.finditer(r'pub use (\w+)::\{([^}]+)\};', src, re.DOTALL):
        sub = m.group(1)
        items = [x.strip().split(' as ')[0].strip() for x in m.group(2).split(',') if x.strip()]
        blocks.append((sub, items))
    # pub use submod::Item;   (single-segment path)
    for m in re.finditer(r'^pub use (\w+)::([\w_]+);', src, re.MULTILINE):
        sub, item = m.group(1), m.group(2)
        blocks.append((sub, [item]))
    return blocks

def load_excludes():
    if not os.path.exists(EXCLUDES_FILE): return set()
    out = set()
    with open(EXCLUDES_FILE) as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if line: out.add(line)
    return out

def main():
    facade_flat, facade_mods = parse_facade('crates/entelix/src/lib.rs')
    excludes = load_excludes()
    violations = []
    for crate_dir in sorted(os.listdir('crates')):
        if crate_dir == 'entelix': continue
        if not crate_dir.startswith('entelix-'): continue
        crate = crate_dir.replace('-', '_')
        crate_flat = facade_flat.get(crate, set())
        crate_mods = facade_mods.get(crate, set())
        for sub, items in parse_subcrate(f'crates/{crate_dir}/src/lib.rs'):
            for item in items:
                key = f'{crate_dir}::{item}'
                if item in crate_flat: continue
                if sub in crate_mods: continue
                if key in excludes: continue
                violations.append(key)
    for v in violations: print(v)

main()
PY
)"

if [[ -n "${violations}" ]]; then
    count="$(printf '%s\n' "${violations}" | wc -l | xargs)"
    echo "::error::facade missing ${count} re-exports — sub-crates expose items the canonical \`entelix\` path can't reach:"
    printf '%s\n' "${violations}"
    echo
    cat <<'EOF'
—————————————————————————————————————————————————————————————————————————
Every sub-crate `pub use` item should be reachable through the
`entelix` facade — either by a flat re-export or by a module
re-export. Per ADR-0064, the facade is the canonical 90% surface;
items only reachable via underlying crate paths defeat it.

Two ways to resolve:

  (a) Add the item to `crates/entelix/src/lib.rs` under the matching
      `pub use entelix_X::{...}` block (maintain alphabetical sort —
      cargo fmt enforces case-sensitive ASCII order).

  (b) If the item is intentionally advanced/internal-only, add an
      entry to `scripts/facade-excludes.txt` in the form
      `entelix-<crate>::<Item>` with a comment explaining why.

Internal items that warrant exclusion: low-level dispatch primitives
(`Dispatch`, `scatter`, `FinalizingStream`) where operators reach
through the underlying crate path on purpose.
—————————————————————————————————————————————————————————————————————————
EOF
    exit 1
fi

# Count items in facade for the success message.
count="$(grep -E '^\s+[A-Z_a-z][A-Za-z0-9_]*' crates/entelix/src/lib.rs | wc -l | xargs)"
echo "check-facade-completeness: every sub-crate \`pub use\` item is reachable through the \`entelix\` facade."
