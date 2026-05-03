# scripts/

CI gate + invariant enforcement scripts. Every script is referenced
from `.github/workflows/ci.yml` and runs locally with `bash
scripts/<name>.sh` from the workspace root. Each one exits non-zero on
violation and prints `file:line` + remediation guidance.

The canonical list lives in `CLAUDE.md` §"Commands"; this README is
the per-script reference.

## Per-script summary

| Script | Enforces |
|---|---|
| `check-no-fs.sh` | Invariant 9 — no `std::fs` / `std::process` / `tokio::fs` / sandbox crates in first-party crates. |
| `check-managed-shape.sh` | Invariants 1-4 — Anthropic managed-agent shape (stateless harness, single `Tool::execute`, `SessionGraph` event SSoT, no credentials in `ExecutionContext`). |
| `check-naming.sh` | ADR-0010 naming taxonomy — forbidden suffixes (`*Engine` / `*Wrapper` / `*Handler` / `*Helper` / `*Util`), `get_*` accessors, `*Service` outside `entelix-server`, ctx-position split per trait. |
| `check-surface-hygiene.sh` | `#[non_exhaustive]` on every public enum + Tier-1 user-facing structs; `#[source]` / `#[from]` on error variants carrying inner errors. |
| `check-silent-fallback.sh` | Invariant 15 — no `unwrap_or*` in codecs / transports / cost meter without an audited `// silent-fallback-ok:` marker. ADR-0032. |
| `check-magic-constants.sh` | Invariant 17 — no probability-shaped literals (`0.X`) in heuristic-prone paths (codec / transport / agent / cost). ADR-0034. |
| `check-no-shims.sh` | Invariant 14 — no `#[deprecated]`, no `// deprecated`, no `pub use X as YOld`, no `// formerly …`. |
| `check-lock-ordering.sh` | `await_holding_lock` + `await_holding_refcell_ref` clippy lints pinned at deny level. |
| `check-feature-matrix.sh` | Every facade feature compiles in isolation (no implicit feature dependency). |
| `check-dead-deps.sh` | Every `[workspace.dependencies]` entry is inherited by at least one crate. |
| `check-facade-completeness.sh` | Every sub-crate `pub use` item is reachable through the `entelix` facade (intentional internals listed in `scripts/facade-excludes.txt`). |
| `check-doc-canonical-paths.sh` | Live operator-facing docs use facade paths (`entelix::Foo`), not underlying crate paths (`entelix_core::Foo`). |
| `check-supply-chain.sh` | `cargo audit` (RustSec CVE) + `cargo deny` (license + bans + transitive). |
| `check-public-api.sh` | Per-crate public-API drift against `docs/public-api/<crate>.txt`. Refreeze with `freeze-public-api.sh <crate>`. |

## Authoring conventions

- `#!/usr/bin/env bash` + `set -euo pipefail`.
- CWD-agnostic — `cd` to repo root via `${BASH_SOURCE[0]}` discovery.
- Prefer `rg` (ripgrep) with `grep -rE` fallback.
- Output: violation → `file:line` + actionable remediation. Exit 0 = clean.
- Every new script must be wired into `.github/workflows/ci.yml` in the same PR.
