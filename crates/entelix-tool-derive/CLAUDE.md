# entelix-tool-derive

Proc-macro crate. Holds the `#[tool]` attribute that generates typed-input tool boilerplate from an async fn signature.

## Surface

- **`#[tool]`** — function-attribute macro. Applied to an `async fn` whose signature is `async fn name(ctx: &AgentContext<()>,.args) -> Result<O>` (ctx optional). Generates: an `Input` struct (`Deserialize + JsonSchema` over the param list), a unit struct named after the function in `PascalCase`, and an `entelix_tools::SchemaTool` impl that deserialises, dispatches to the original fn, and returns the typed `O`. The original function stays callable in user code; the tool struct is the agent-side surface.

## Crate-local rules

- **Proc-macro only.** No runtime types live here; everything generated references `entelix_core` / `entelix_tools` / `serde` / `schemars` / `async_trait` from the *consumer* crate's dep graph. Adding runtime deps to this crate is a hygiene bug.
- **Doc-comment first paragraph → description.** Anything past the first blank line stays in the source as developer documentation; the `description()` accessor surfaces only the first paragraph (trimmed).
- **Function name (snake_case) → struct name (`PascalCase`).** Use `#[tool(name = "…")]` to override the LLM-facing tool name without renaming the Rust function. `effect`, `idempotent`, `version`, and `retry_hint` are also accepted as attribute args.
- **No generic functions / lifetime-parameterised functions.** Rejected at parse time. Operators with those requirements implement `SchemaTool` manually.
- **`Result<O>` return required.** Anything else is rejected at parse time.

## Forbidden

- A generated path that does not start with `::` (the leading-colons fully-qualified path is the proc-macro hygiene fix that prevents user-side shadowing of `entelix_core` / `entelix_tools`).
- A new attribute argument (effect, retry, version, …) without a corresponding `SchemaTool` accessor and an `entelix-tool-derive/tests/` test.
