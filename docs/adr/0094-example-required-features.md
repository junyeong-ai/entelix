# ADR 0094 — Examples declare `required-features` for feature-gated APIs

**Status**: Accepted
**Date**: 2026-05-06
**Decision**: Every `[[example]]` that imports facade items behind `#[cfg(feature = "X")]` declares the matching `required-features = ["X"]` in `crates/entelix/Cargo.toml`. The doc-comment build instruction at the top of each example mirrors the feature flag so a copy-paste user runs the right command. Bare `cargo build --examples -p entelix` (no features) skips the gated examples cleanly; `cargo build --examples -p entelix --all-features` builds the full set.

## Context

Examples 13/15/17/18 imported feature-gated facade items (`mcp`, `policy`, `mcp-chatmodel`) without declaring `required-features` in `Cargo.toml`. Symptoms:

- `cargo build --examples -p entelix` (no features) → 4 compile errors (`E0432: unresolved import`).
- `cargo build --example 13_mcp_tools -p entelix` → fails on `entelix::ChatModelSamplingProvider`-style imports.
- The doc-comments at the top of each example said `cargo build --example 13_mcp_tools -p entelix` — wrong without features.

CI didn't catch it because the workspace test job runs with `--all-features`, which always satisfies the gates by accident. Local devs running bare `cargo build --examples` hit the wall.

`scripts/check-feature-matrix.sh` (slice 84) covers feature isolation for the *facade build*, not for examples — examples are a separate Cargo target shape with their own `required-features` keyword. Two examples (`12_compat_matrix`, `14_serve_agent`) had `required-features` set; four did not.

## Decision

### `crates/entelix/Cargo.toml`

```toml
[[example]]
name = "13_mcp_tools"
required-features = ["mcp"]

[[example]]
name = "15_production_workflow"
required-features = ["policy"]

[[example]]
name = "17_mcp_sampling_provider"
required-features = ["mcp", "mcp-chatmodel"]

[[example]]
name = "18_tool_approval"
required-features = ["policy"]
```

### Doc-comment build instructions

Each example's `//!` header includes the feature flag in the build / run command:

```rust
//! Build: `cargo build --example 18_tool_approval -p entelix --features policy`
//! Run:   `cargo run   --example 18_tool_approval -p entelix --features policy`
```

Without that, copy-paste users hit the same compile error.

### Verification

- `cargo build --examples -p entelix` — finishes in 0.22s (skips 6 feature-gated examples cleanly).
- `cargo build --examples -p entelix --all-features` — finishes in ~13s (builds all 18).
- `cargo xtask invariants` — 11/11 clean.

## Consequences

- Examples are split into two cohorts by Cargo's `required-features`:
  - Always-on (12 examples): `01..11`, `16`.
  - Feature-gated (6 examples): `12` (`aws,gcp,azure`), `13` (`mcp`), `14` (`server`), `15` (`policy`), `17` (`mcp,mcp-chatmodel`), `18` (`policy`).
- The failure mode shifts from "compile error" to "Cargo silently skips" — quieter, but Cargo logs `Skipping target ... required features ...` so the silence is intentional.
- This pattern locks in for any future example: if it imports a feature-gated facade item, declare `required-features` and update the doc-comment.

## References

- ADR-0064 — 1.0 release charter (facade is the canonical surface contract).
- `scripts/check-feature-matrix.sh` (ADR-0084) — feature isolation regression gate; covers the facade build, not examples.
- v3 plan slice 113 (examples migrate / cleanup).
