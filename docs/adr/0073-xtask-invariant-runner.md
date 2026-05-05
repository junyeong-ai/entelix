# ADR 0073 — `cargo xtask` typed-AST invariant runner

**Status**: Accepted
**Date**: 2026-05-05
**Decision**: The 12 shell scripts that enforced workspace invariants migrate into one Rust binary at `xtask/`. Each invariant becomes a typed-AST visitor (`syn::File` for Rust source, `toml_edit::DocumentMut` for manifests) at `xtask/src/invariants/<name>.rs`, dispatched by a clap subcommand. The xtask crate lives in its own workspace so its dependencies and clippy regime do not bleed into production builds. CI calls `cargo xtask invariants` for the static-analysable suite and `cargo xtask <name>` for the heavier network / cargo-subprocess gates.

## Context

The 1.0 RC enforcement story landed across 12 shell scripts under `scripts/check-*.sh` — `check-no-fs.sh`, `check-managed-shape.sh`, `check-naming.sh`, `check-surface-hygiene.sh`, `check-silent-fallback.sh`, `check-magic-constants.sh`, `check-no-shims.sh`, `check-lock-ordering.sh`, `check-dead-deps.sh`, `check-facade-completeness.sh`, `check-doc-canonical-paths.sh`, `check-feature-matrix.sh`, plus `check-supply-chain.sh` and `check-public-api.sh` for the heavier gates and `freeze-public-api.sh` for the operator-side baseline refresh.

Each script was a regex / `rg` / `awk` pipeline over raw source text. That worked for the simplest gates (`rg 'pub use \w+ as \w+'` for shim detection) but the regex frame missed structural patterns the invariants actually care about:

- **`pub async fn` and trait-default methods** — `check-naming.sh` matched `fn get_\w+` but not `async fn get_…` reliably across multi-line signatures, and trait `default fn` bodies were invisible to the line-oriented scan.
- **Fully-qualified `std::fs` calls** — `check-no-fs.sh` matched `use std::fs` but not bare-call sites like `std::fs::read(p)?;` or `tokio::fs::read_to_string(p).await?;` that compile without an import.
- **Macro-expanded violations** — `include_str!("…")`, `include_bytes!("…")` are filesystem reads at compile time. The shell scan saw the macro name but not its semantic meaning, and any `#[cfg(test)] include_str!` was indistinguishable from a production-time include.
- **Trait `default` impls of forbidden methods** — `delete_edge` / `delete_node` no-op defaults (ADR-0065 closed this) were scanned by file path, so a default impl in a different file was missed.
- **`#[non_exhaustive]` on public enums** — line scan found the attribute but not its absence on every public enum, so adding a new public enum without the attribute slipped through.
- **`with_*(&self)` builders** — the `with_*` regex matched the prefix but couldn't tell whether the receiver was `self` (consumes — correct) or `&self` / `&mut self` (mutator misuse).
- **`#[derive(thiserror::Error)]` variant `#[source]` chains** — line scan couldn't bind a variant to its `#[error(transparent)]` or `#[from]` annotation across attribute lines.

The shell scripts also had operational pain that scaled badly:

- **No shared parser state**. Each script re-tokenised the workspace, and several scanned the same file 4-5x with different regexes.
- **No common output format**. One script printed `path:line`, another `path` only, another printed the offending line excerpt with no location at all. CI failure triage took longer than running the gates.
- **Bash portability cost**. macOS `sed -E` and GNU `sed -r`, `grep -P` PCRE, `rg` flags drifted between scripts; some used `awk` heredocs that broke under `set -euo pipefail` in unexpected ways.
- **Hidden dependencies**. `check-naming.sh` shelled out to a Python helper for builder-prefix scope detection (slice 79). `check-feature-matrix.sh` invoked `cargo metadata` and parsed JSON via `jq`. The "shell script" framing concealed a multi-language toolchain CI had to keep installed.
- **No type system**. A bug in `check-silent-fallback.sh` that misclassified an audited `unwrap_or_default` site silently allowed a real violation through for two slices before being noticed.

The shell-script frame had served the 0.x → 1.0-RC bring-up where the rules themselves were still being discovered. Once the rules stabilised at 18 invariants + the per-rule ADR backing, the regex frame stopped paying for itself — every new gate added another ~50-150 line shell file with its own argument parsing, output format, and fragile `set -e` semantics.

## Decision

### One binary, one subcommand per invariant

`xtask/src/main.rs` exposes a clap CLI:

```bash
cargo xtask invariants                # every static-analysable gate, canonical CI order
cargo xtask <name>                    # one gate by short name
cargo xtask freeze-public-api [crate] # operator-side baseline refresh
```

The `Cmd` enum maps 1:1 onto `xtask/src/invariants/<name>.rs`:

| Subcommand | Visitor module | CLAUDE.md invariant |
|---|---|---|
| `no-fs` | `no_fs.rs` | 9 |
| `managed-shape` | `managed_shape.rs` | 1, 2, 4, 10 + ADR-0035 |
| `naming` | `naming.rs` | ADR-0010 + ctx-position |
| `surface-hygiene` | `surface_hygiene.rs` | `#[non_exhaustive]` + `#[source]` / `#[from]` |
| `silent-fallback` | `silent_fallback.rs` | 15 + ADR-0032 |
| `magic-constants` | `magic_constants.rs` | 17 + ADR-0034 |
| `no-shims` | `no_shims.rs` | 14 |
| `lock-ordering` | `lock_ordering.rs` | "Lock ordering" §, `await_holding_*` |
| `dead-deps` | `dead_deps.rs` | `[workspace.dependencies]` hygiene |
| `facade-completeness` | `facade_completeness.rs` | every `pub use` reachable via `entelix::*` |
| `doc-canonical-paths` | `doc_canonical_paths.rs` | live docs use facade paths |
| `supply-chain` | `supply_chain.rs` | `cargo audit` + `cargo deny` (subprocess) |
| `feature-matrix` | `feature_matrix.rs` | each feature compiles alone (subprocess) |
| `public-api` | `public_api.rs` | per-crate baseline drift (subprocess) |

The `invariants` subcommand runs the 11 static-analysable gates in canonical CI order; `supply-chain`, `feature-matrix`, `public-api` are gated on heavier toolchain installs (`cargo-audit`, `cargo-deny`, `cargo-public-api`) and run as separate CI jobs.

### Visitors are typed-AST, not regex

Every visitor parses Rust source with `syn::parse_file(&src)?` and walks the resulting AST through a `syn::visit::Visit<'_>` impl. Every manifest is parsed with `toml_edit::DocumentMut`. Span resolution to `path:line:col` uses `proc_macro2::Span::start()` (gated on the `span-locations` feature, since the method is otherwise opaque outside proc-macro context — see `xtask/Cargo.toml`).

This buys back what the regex frame missed:

- **`visit_item_fn` + `visit_trait_item_fn`** see every function definition uniformly, including trait-default bodies, `pub async fn`, and `extern "C" fn`.
- **`visit_expr_path` + `visit_expr_call`** see `std::fs::read(...)` calls regardless of whether the segments came through `use` or were spelled fully.
- **`visit_macro`** sees `include_str!` / `include_bytes!` invocations and can branch on `#[cfg(test)]` parent attributes by walking up the visit stack.
- **`visit_item_enum` + attribute scan** confirms `#[non_exhaustive]` is present on every public enum, not just that the attribute appears somewhere in the file.
- **`Receiver` type-match on `ImplItemFn::sig.inputs[0]`** distinguishes `self` from `&self` / `&mut self` — `with_*` mutators get caught structurally.
- **`Variant.fields` walk** binds each `#[error(transparent)]` / `#[from]` attribute to its variant for the `surface-hygiene` chain check.

The visitors share one `walkdir::WalkDir` traversal, one `syn::parse_file` per file, and one error-collecting buffer (`Vec<Violation { path: PathBuf, line: usize, col: usize, msg: String }>`), so scanning the workspace happens once per gate instead of once per regex.

### Separate workspace for xtask

`xtask/Cargo.toml` declares its own `[workspace]` table, distinct from the main entelix workspace. This is intentional and load-bearing:

- **Dependency graph isolation**. `clap`, `syn`, `walkdir`, `toml_edit`, `regex`, `proc-macro2`, `anyhow` exist only inside the xtask graph. `cargo metadata` for the production workspace doesn't traverse them, so editor / language-server resolution stays narrow.
- **Clippy regime isolation**. The production workspace runs `clippy::pedantic` + `clippy::nursery` deny across every crate — appropriate for a public SDK but actively wrong for a CLI tool that prints, returns early, and dispatches via function pointers. xtask's `[lints.clippy]` keeps `correctness` / `suspicious` / `style` / `complexity` / `perf` deny but explicitly relaxes `collapsible_if` / `collapsible_match` because each visit method is more readable as one sentence per arm.
- **Build cache isolation**. CI caches `xtask/target/` under `Swatinem/rust-cache@v2`'s `workspaces: xtask` key, separate from the production cache key per OS. The production cache stays warm even when only xtask invariant logic changes (and vice versa).
- **`cargo build --workspace --all-features` stays narrow**. Nothing in the production workspace depends on xtask, so building the SDK doesn't compile xtask's binary.

The `.cargo/config.toml` alias `xtask = "run --quiet --release --package xtask --"` makes `cargo xtask <subcommand>` work uniformly from the repo root, regardless of which workspace the user's shell happens to be standing in.

### CI wiring

`.github/workflows/ci.yml` collapses the 11-step preflight scaffolding (one shell script per step) into a single `cargo xtask invariants` step. The supply-chain, feature-matrix, and public-api jobs each call their respective `cargo xtask <name>` subcommand. Cache key for the preflight job uses `workspaces: xtask`; the production jobs use per-OS keys.

The result: failure output is uniform (`path:line:col → message`), the canonical run order is encoded in `run_all` (Rust source you can grep), and adding a new invariant is one new file under `xtask/src/invariants/`, one `Cmd` variant, one dispatch-arm match, and one tuple in the `run_all` gate list. No new shell file, no new CI step.

## Consequences

**Positive**:

- Structural patterns the shell frame missed are now caught (FQP `std::fs` calls, `pub async fn` accessors, trait-default no-op overrides, `include_str!` macros, `#[non_exhaustive]` absence, `with_*(&self)` mutator misuse).
- One output format across every gate. CI triage drops to "open the path:line:col, read the message".
- Adding a new invariant lands in one place — `xtask/src/invariants/<name>.rs` — instead of a script + a CI step + a CLAUDE.md note + a README entry.
- Shared scan state means each file is parsed once per gate, not once per regex.
- The runner is unit-testable — visitors take an in-memory `&syn::File` so per-rule fixtures live next to the visitor.

**Negative**:

- One Rust binary to compile from cold. CI cold start adds ~2 minutes for the first xtask compilation; warm cache (Swatinem) brings it back to ~10 seconds. The production preflight previously paid no compile cost.
- Contributors who ran `bash scripts/check-no-fs.sh` directly need to learn `cargo xtask no-fs`. The alias keeps the muscle memory close (one word), but the shell-script-as-grep-context affordance is gone.
- xtask's separate workspace means `Cargo.lock` lives at `xtask/Cargo.lock`, separate from the production lock. Two lockfiles to update when bumping shared transitive deps. (`syn` is the only meaningful overlap — the production workspace uses it through `entelix-graph-derive` for proc-macro work, xtask uses it for invariant parsing. Lockstep is not required.)

**Migration outcome (one-shot, no shim)**: All 12 `scripts/check-*.sh`, `scripts/freeze-public-api.sh`, `scripts/README.md`, and `scripts/facade-excludes.txt` are deleted in the same commit that lands `xtask/`. CLAUDE.md `Commands` section, `.claude/rules/naming.md` self-check footer, and `.github/workflows/ci.yml` switch to `cargo xtask` references in the same commit. There is no transition period and no backwards-compatible shim — invariant 14 forbids it.

## References

- CLAUDE.md `Commands` § — public xtask subcommand list
- CLAUDE.md invariants 9, 14, 15, 17 — specific gate references that previously named scripts
- ADR-0010 — naming taxonomy that `cargo xtask naming` enforces
- ADR-0032 — silent-fallback bug class that `cargo xtask silent-fallback` enforces
- ADR-0034 — heuristic externalisation that `cargo xtask magic-constants` enforces
- ADR-0035 — managed-agent shape that `cargo xtask managed-shape` enforces
- ADR-0064 — 1.0 release charter (post-compile contract that the xtask suite enforces)
- `xtask/src/main.rs` — clap subcommand definitions
- `xtask/src/invariants/<name>.rs` — per-invariant visitor implementations
- `xtask/src/visitor.rs` — shared `Visit<'_>` helpers (path resolution, span → `line:col`, attribute walk)
