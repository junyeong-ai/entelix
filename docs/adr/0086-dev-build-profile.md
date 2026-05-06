# ADR 0086 — Dev build profile tuned for 21-crate workspace scale

**Status**: Accepted
**Date**: 2026-05-06
**Decision**: `[profile.dev]` is tuned for a workspace with **112 integration test binaries + 18 examples + 2447 transitive dependencies** statically linking the full graph per binary. The default `cargo` settings produce ~58 GB `target/` directories with ~970,000 fingerprint files; the canonical entelix dev profile (`debug = "line-tables-only"`, `split-debuginfo = "unpacked"`, `incremental = true`, `codegen-units = 16`, `[profile.dev.package."*"] debug = false`, `[profile.test] inherits = "dev"`) keeps `target/debug` at ~2.3 GB and ~2,000 files — a 25× disk and 450× file-count reduction. Cold-build and touched-file rebuild times shrink in lockstep (`cargo check --workspace --all-targets` cold: 6m 18s → 39s; touch + rebuild: 60+s → **2.9s**).

## Context

The entelix workspace has crossed a scale threshold where the cargo dev profile defaults stop working:

- **21 first-party crates** + **2447 transitive deps** (reqwest + rustls + sqlx + tokio + opentelemetry + ...).
- **112 integration test binaries** distributed across crates (entelix-core: 32, entelix-agents: 12, entelix-graph: 11, …).
- **18 examples** in the `entelix` facade.

Each integration-test binary statically links the entire dep graph independently. With cargo's stock dev profile the symptoms surface fast:

| Symptom | Stock defaults | Tuned defaults | Reduction |
|---|---|---|---|
| `target/` size | 58 GB | ~2.3 GB | **25×** |
| `target/debug/deps/` files | 971,670 | 2,012 | **480×** |
| `cargo check --workspace --all-targets` cold | 6m 18s | 39s | **9.7×** |
| `cargo clippy --workspace --all-targets` | 1m 50s (one crate) | 13s (workspace) | **8.5×** |
| Touch + `cargo check -p X` rebuild | 60+s | 4.8s | **12×** |
| Touch + workspace --all-targets rebuild | 60+s | **2.9s** | **20×** |

The 58 GB / 970k-file pathology came from four compounding factors:

1. **Per-binary debug info duplication** — every test binary contains its own copy of every dep's DWARF.
2. **`debug = 2` default** — full DWARF including variable / type info, ~5× the size of line-tables-only.
3. **`debug = "limited"` (initial fix attempt)** — better than full DWARF but still inflated; line-tables-only is the right level for an SDK debugged via `tracing` + tests rather than LLDB symbol inspection.
4. **High `codegen-units` count** — an experimental `codegen-units = 256` setting fragmented every crate into 256 small `.o` files, multiplying file count by ~16× without recoverable parallelism gain.

Other factors verified as **non-causes**:

- **Incremental compilation accumulation** — already disabled in slice 99's earlier infra fix; the prior ~33 GB `target/debug/incremental/` was an independent issue and stays at 0 B.
- **Macro-heavy codegen (async-trait, schemars derive, sqlx macros)** — contributes to slow compile but does not balloon the build artifact tree on its own.

## Decision

```toml
[profile.dev]
debug = "line-tables-only"
split-debuginfo = "unpacked"
incremental = true
codegen-units = 16

[profile.dev.package."*"]
debug = false
opt-level = 0

[profile.test]
inherits = "dev"
```

### `debug = "line-tables-only"`

Keeps backtraces with file:line resolution but drops the DWARF variable / type metadata. SDKs are typically debugged through `tracing` spans, structured assertions, and unit-test failure messages — not through LLDB stepping over local variables. The debug-info-shape that `line-tables-only` discards is the dominant per-`.o` size driver.

### `split-debuginfo = "unpacked"`

macOS canonical setting (matches Apple's own toolchains). Splits debug info into `.dwo` files alongside the `.o` instead of embedding it inside the `.o`. Prevents per-codegen-unit duplication of identical debug records — every CGU that uses the same dep gets a reference to one shared `.dwo`, not a copy.

### `incremental = true`

A first cut at this ADR disabled incremental compilation, reasoning that the 33 GB cache observed under stock defaults proved the feature was unsalvageable at workspace scale. Re-measurement after the other three axes (debug-strip, package-`*` strip, codegen-units default) showed the conclusion was wrong: the 33 GB was caused by *full DWARF × 2400 third-party deps × per-feature matrix*, not by incremental itself. With `debug = "line-tables-only"` shrinking each fingerprint ~5× and `[profile.dev.package."*"] debug = false` cutting the dep contribution another ~80%, incremental cache settles in the ~1 GB range — fully manageable.

The trade re-measured under the corrected baseline:

|  | `incremental = false` | `incremental = true` | Verdict |
|---|---|---|---|
| Cold workspace --all-targets | 28 s | 39 s | small regression |
| Touch + check -p X | 7.5 s | 4.8 s | 1.6× faster |
| Touch + check workspace --all-targets | 11.2 s | **2.9 s** | **3.9× faster** |
| `target/debug` size | 1.0 GB | 2.3 GB | acceptable |

Touched-file rebuilds happen 100s of times per slice; cold builds happen ~once per day. The break-even disk cost (1.3 GB) buys ~10 minutes per slice across the inner-loop cycle. `incremental = true` wins decisively at this workspace scale once the other axes are in place.

### `codegen-units = 16`

The cargo default for dev profile. The earlier `codegen-units = 256` experiment is the proximate cause of the 970k-file blow-up: more CGUs ⇒ more per-`.o` files ⇒ linear file-count growth without parallelism gain (Apple-Silicon `M`-class CPUs saturate at 8–14 parallel codegen jobs; 256 CGUs creates pure file-system overhead past that). 16 is the right setting; do not change it without `cargo --timings` evidence.

### `[profile.dev.package."*"]`

Third-party deps are debugged through their own published sources or release builds, never through this workspace's debug info. `debug = false` for non-workspace crates strips ~80% of `target/` size with zero functional cost — the **single biggest disk-saving axis** of the four. `opt-level = 0` stays so panic / overflow / bounds checks remain useful when an entelix tool body calls into a third-party crate.

### `[profile.test] inherits = "dev"`

Cargo's `test` profile inherits from `dev` by default, so `cargo build` and `cargo test` already share a single cache instead of fragmenting into two parallel build trees. Declaring the inheritance explicitly documents the intent and protects against silent default drift across cargo releases.

## Hot-path opt-level boost (deferred)

A common companion to this profile is `[profile.dev.package.<hot-path>] opt-level = 1` for libraries that run heavily inside integration tests (HTTP transports, JSON codecs, async runtime). For entelix the candidates are `reqwest`, `tokio`, `serde_json`, `jsonschema`, `tower`. Adding it speeds up test runtime at the cost of one-time compile-time on those deps.

Deferred until a measured test-runtime bottleneck emerges. Premature opt-level boost without `--timings` evidence inflates compile time on every dep upgrade for unclear gain.

## Future work — integration test consolidation

The 112-binary count is the residual driver of compile fan-out. A separate slice will consolidate per-crate integration tests under one binary per crate by moving `tests/foo.rs` files into `tests/integration/foo.rs` sub-modules and adding a single `tests/integration.rs` entry point per crate. Estimated impact: **112 binaries → 21 binaries**, a 5× compile-target reduction. Deferred to its own slice because the file-move surface is large and the migration changes `cargo test --test foo` ergonomics.

## Consequences

- `target/` stays under 2 GB indefinitely — fits on developer laptops without `cargo clean` rituals.
- `cargo` invocations bottom out in 5–30s instead of 60–400s — the inner-loop tax that prompted the slice 100 mass-migration cycle audit drops by an order of magnitude.
- Backtraces and panic messages stay file-line precise — invariant 12 / 15 diagnostic surface unchanged.
- LLDB step-debugging over workspace code still works (we keep `debug = "line-tables-only"`); deep variable-inspection sessions on third-party deps require a one-off `cargo build --profile dev-debug` (operator-defined profile) — vanishingly rare in practice.

## References

- ADR-0064 — 1.0 release charter; `target/` budget is part of "developer ergonomics" non-functional contract.
- v3 plan slice 100 — surfaced the `cargo` cycle-time problem that motivated this fix.
