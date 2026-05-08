# entelix (facade)

Re-export crate. The 90% surface every consumer constructs: `entelix::ChatModel`, `entelix::Agent`, `entelix::StateGraph`, etc. Sub-crate paths (`entelix_core`, `entelix_agents`, …) stay accessible for the 10% that needs internals.

## Surface

- **Module re-exports** — `auth`, `cancellation`, `codecs`, `events`, `ir`, `service`, `skills`, `stream`, `tools`, `transports` (all from `entelix-core`).
- **Headline top-level types** — `ChatModel`, `ChatModelConfig`, `Error`, `ExecutionContext`, `Result`, `ThreadKey` (every consumer touches these on every call).
- **Sub-crate flat re-exports** — `Agent`, `Subagent`, `StateGraph`, `Annotated`, `BufferMemory`, `JsonOutputParser`, `Runnable`, `RunnableExt`, `SessionGraph`, `GraphEvent`, etc. The full re-export surface is the lib.rs body itself; `Cargo.toml` is the source of truth for what's gated.
- **Feature-gated re-exports** — each facade feature gates the corresponding sub-crate's full surface plus passes through the underlying crate's matching feature (e.g., `postgres = ["dep:entelix-persistence", "entelix-persistence/postgres"]`). Canonical list in `Cargo.toml`.
- **`prelude` module** — `use entelix::prelude::*` brings `ContentPart` / `Message` / `Role` / `ChatModel` / `Error` / `ExecutionContext` / `Result` / `ChatPromptPart` / `ChatPromptTemplate` / `PromptValue` / `PromptVars` / `JsonOutputParser` / `Runnable` / `RunnableExt`. Curated for "5-line agent" demos.

## Crate-local rules

- **Every feature-gated re-export carries `#[cfg_attr(docsrs, doc(cfg(feature = "X")))]` next to its `#[cfg(feature = "X")]`** — docs.rs surfaces "Available on **feature** `X` only" badges only when both pair.
- **Feature pass-through is mandatory** — facade `postgres` feature MUST enable underlying `entelix-persistence/postgres`. `cargo xtask feature-matrix` catches the regression where `dep:entelix-persistence` enabled but `entelix-persistence/postgres` forgotten.
- **Re-export ordering** — within each `pub use entelix_X::{.}` block, items are case-sensitive ASCII-sorted (rustfmt default). PascalCase types appear before snake_case functions because uppercase letters sort before lowercase in ASCII. Don't manually reorder — `cargo fmt` enforces.
- **Alias to disambiguate naming collisions** — `MCP_PROTOCOL_VERSION as MCP_PROTOCOL_VERSION`, `SERVER_DEFAULT_TENANT_HEADER as SERVER_DEFAULT_TENANT_HEADER`, `OPENAI_EMBEDDINGS_BASE_URL as OPENAI_EMBEDDINGS_BASE_URL`. Same-named constants from different crates carry a prefix at the facade level.
- **No baseline** — `entelix` is intentionally excluded from `cargo xtask public-api` (the underlying crates carry the surface contract; the facade just re-exports).

## Forbidden

- A `pub use` that drops the `#[cfg(feature = "X")]` gate for a re-export from a feature-gated dependency — facade compile breaks under `--no-default-features`.
- A new feature flag that doesn't pass through to the underlying crate's matching feature (and `check-feature-matrix.sh` doesn't catch the gap until CI runs).
- A re-export of an internal-only type (e.g. `entelix-graph::dispatch::scatter`) — the facade is the canonical 90% surface, advanced internals stay behind sub-crate paths.

## References

- `cargo xtask feature-matrix` — feature isolation regression gate.
- `cargo xtask dead-deps` — workspace dependency hygiene.
