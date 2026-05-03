# entelix-runnable

Composition contract (invariant 7). `Runnable<I, O>` + LCEL `.pipe()` connector + adapter combinators. Anything composable in entelix implements this trait.

## Surface

- **`Runnable<I, O>` trait** — `invoke(input, ctx) → Result<O>` + `stream(input, mode, ctx) → BoxStream<StreamChunk<O>>` + optional `name()`. ctx-last per naming taxonomy.
- **`RunnableExt` extension trait** — fluent adapters returning concrete `Runnable<I, O>` types so chains stay zero-cost in the steady state. `pipe(next)` / `with_retry(policy)` / `with_fallbacks(others)` / `map(fn)` / `with_config(fn)` / `with_timeout(duration)` / `stream_with(...)`.
- **Composition primitives** — `RunnableLambda` / `RunnableSequence` / `RunnableParallel` / `RunnableRouter` / `RunnablePassthrough` (LangChain LCEL parity).
- **`AnyRunnable` + `AnyRunnableHandle` + `erase`** — type-erased counterpart for dynamic dispatch (F12 mitigation). `invoke_any(Value, ctx) → Result<Value>`. Pays JSON ser/deser cost — typed `Runnable` for hot paths.
- **Parsers** — `JsonOutputParser<T: DeserializeOwned>` + `RetryParser<P>` + `FixingOutputParser<P, F>`. Parsers are also `Runnable<Message, T>` — chainable via `.pipe()`.
- **Streaming** — `StreamChunk<O>` + `RunnableEvent` + `StreamMode::{Values, Updates, Messages, Debug, Events}` (LangGraph parity). `BoxStream<'static, Result<StreamChunk<O>>>`.
- **Adapters** — `Configured<R, F>` (per-call ctx mutation), `Mapping<R, F>` (output transform), `Retrying<R>`, `Fallback<R>`, `Timed<R>`, `ToolToRunnableAdapter` (re-exported from `entelix-core::tools`), `DebugEvent` for observability tap.

## Crate-local rules

- **Adapter return type concrete** — `with_retry` returns `Retrying<Self>` not `Box<dyn Runnable>`. Boxing happens only at explicit `.erase()`. Keeps composition zero-cost.
- **`Runnable::stream` default impl** — single-shot wrap of `invoke` as a one-element stream. Implementors override only when streaming has true semantic value (token deltas, intermediate state events).
- **No `'static` requirement on the input/output beyond `Send + 'static`** — generic over any `I: Send + 'static, O: Send + 'static` so tuple-typed parallel outputs and trait-object inputs both compose.
- **`pipe` is the universal connector** — new combinators that don't compose through `.pipe()` are reviewer-rejected. Helpers like `Mapping` / `RunnablePassthrough` exist precisely to fit the `.pipe()` shape.
- **Cancellation surface** — every async path takes `&ExecutionContext`. Long retry loops poll `ctx.is_cancelled()` (CLAUDE.md §"Cancellation"). `RetryParser` + `FixingOutputParser` already poll between attempts.

## Forbidden

- An adapter that boxes its inner runnable when generic specialization would do — defeats the zero-cost composition contract.
- `unwrap()` / `expect()` in any adapter's `invoke` / `stream` — bubble through `Result<O, Error>`.
- Holding any lock across a user-supplied `Runnable::invoke` future (CLAUDE.md §"Lock ordering"). Lock guards drop before `.await`.

## References

- ADR-0006 — Runnable + StateGraph 1.0 spine.
- ADR-0010 — naming taxonomy (`Runnable<Verb>` composition prefix).
- ADR-0011 — `Tool` / `Runnable` adapter boundary (`ToolToRunnableAdapter`).
- ADR-0028 — retry / fallback policy externalisation.
