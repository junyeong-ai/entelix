# entelix-prompt

LangChain-style prompt templating. `PromptTemplate` (string-output) + `ChatPromptTemplate` (message-list output) + few-shot variants. Both implement `Runnable`.

## Surface

- **`PromptTemplate`** — single-string template. `from_template("…{var}…")` → `Runnable<HashMap<&str, Value>, String>`. minijinja under the hood.
- **`ChatPromptTemplate`** — multi-role message list. `from_messages([("system", "…"), MessagesPlaceholder::new("history"), ("user", "…")])` → `Runnable<HashMap<_, _>, Vec<Message>>`.
- **`MessagesPlaceholder`** — splices a `Vec<Message>` from the input map into the rendered chat list (LangChain parity).
- **Few-shot** — `FewShotPromptTemplate` / `ChatFewShotPromptTemplate` + `ExampleSelector` trait + `FixedExampleSelector` / `LengthBasedExampleSelector` reference impls + `SharedExampleSelector = Arc<dyn ExampleSelector>` for shared dispatch. `Example = HashMap<String, String>` — variable bindings the selector returns for each pick.
- **`PromptValue` + `PromptVars`** — typed input + `prelude`-friendly aliases used by recipes.

## Crate-local rules

- **Templates are stateless** — `PromptTemplate` / `ChatPromptTemplate` are `Clone` + `Send + Sync`, no per-call mutable state. Per-render context lives in the input `HashMap`.
- **minijinja preserve_order + serde** — feature set fixed at `["builtins", "loader", "preserve_order", "serde"]` so iteration order over the input map is deterministic. New minijinja features go through review.
- **`MessagesPlaceholder` requires the slot to exist** — missing key in input map is an error, not silent empty. `ChatPromptTemplate::invoke` returns `Error::invalid_request` with the missing key name.
- **`Runnable` impl for `PromptTemplate` is ctx-last** — `invoke(input, ctx)` per ADR-0010 (computation/dispatch tier).

## Forbidden

- A template that mutates state across calls — breaks `Send + Sync` and the stateless contract.
- An `ExampleSelector` impl that returns mutable refs — examples are owned values, immutable post-construction.

## References

- ADR-0010 — naming taxonomy (`*Template`, `*Selector` suffixes).
- `docs/architecture/runnable-and-lcel.md` — `ChatPromptTemplate` as the canonical entry into an LCEL pipeline.
