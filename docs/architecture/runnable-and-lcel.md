# Runnable + LCEL — composition layer

## What this layer provides

A single, composable contract that every "thing the agent can call" implements. Modeled on LangChain's `Runnable` + LCEL `|` pipe, but with Rust generics for compile-time I/O checking.

## The trait

```rust
// entelix-runnable/src/runnable.rs

#[async_trait::async_trait]
pub trait Runnable<I, O>: Send + Sync
where
    I: Send + 'static,
    O: Send + 'static,
{
    /// One-shot invocation.
    async fn invoke(&self, input: I, ctx: &ExecutionContext) -> Result<O, Error>;

    /// Batch invocation. Default = sequential `invoke`. Implementors may parallelize.
    async fn batch(&self, inputs: Vec<I>, ctx: &ExecutionContext) -> Result<Vec<O>, Error> {
        let mut out = Vec::with_capacity(inputs.len());
        for input in inputs {
            out.push(self.invoke(input, ctx).await?);
        }
        Ok(out)
    }

    /// Streamed output. Default = single-shot wrapping `invoke` as a one-element stream.
    async fn stream(
        &self,
        input: I,
        ctx: &ExecutionContext,
    ) -> Result<BoxStream<'static, Result<StreamEvent<O>, Error>>, Error>;

    /// Optional: human-readable name for tracing.
    fn name(&self) -> Cow<'_, str> { Cow::Borrowed(std::any::type_name::<Self>()) }
}
```

`StreamEvent<O>` carries one of: `Token(String)`, `Tool(ToolCall)`, `State(StateUpdate)`, `Final(O)`. See `docs/architecture/streaming.md` for the 5-mode model.

## LCEL — the `.pipe()` operator

```rust
let chain = prompt
    .pipe(model)
    .pipe(parser);

let result = chain.invoke(user_input, &ctx).await?;
```

Implementation:

```rust
// entelix-runnable/src/sequence.rs

pub struct RunnableSequence<I, M, O> {
    first: Arc<dyn Runnable<I, M>>,
    second: Arc<dyn Runnable<M, O>>,
}

impl<I, M, O> Runnable<I, O> for RunnableSequence<I, M, O>
where I: Send + 'static, M: Send + 'static, O: Send + 'static {
    async fn invoke(&self, input: I, ctx: &ExecutionContext) -> Result<O, Error> {
        let mid = self.first.invoke(input, ctx).await?;
        self.second.invoke(mid, ctx).await
    }
    // batch and stream similarly
}

// Extension trait for ergonomic .pipe()
pub trait RunnableExt<I, O>: Runnable<I, O> + Sized + 'static {
    fn pipe<P, O2>(self, next: P) -> RunnableSequence<I, O, O2>
    where P: Runnable<O, O2> + 'static;
}
```

## Built-in combinators

| Combinator | Purpose | Signature |
|---|---|---|
| `RunnableSequence<I, M, O>` | a → b → c | `Runnable<I, O>` from `Runnable<I, M>` + `Runnable<M, O>` |
| `RunnableParallel<I, (O1, O2, ...)>` | run multiple in parallel | takes tuple of Runnables, returns tuple of outputs |
| `RunnableRouter<I, O>` | conditional dispatch | `(predicate, runnable)*` + default |
| `RunnableLambda<I, O>` | wrap a closure | `RunnableLambda::new(\|x, ctx\| async { ... })` |
| `RunnablePassthrough` | identity, useful for fan-in | `Runnable<T, T>` |
| `RunnableMap` | rename / reshape state | `Runnable<I, O>` with `\|i\| → o` |

## What implements Runnable

| Type | I | O | Crate |
|---|---|---|---|
| `ChatModel<C, T>` (codec + transport bundle) | `Vec<Message>` | `Message` | `entelix-core` |
| `PromptTemplate` | `HashMap<&str, Value>` | `String` | `entelix-prompt` |
| `ChatPromptTemplate` | `HashMap<&str, Value>` | `Vec<Message>` | `entelix-prompt` |
| `JsonOutputParser<T: DeserializeOwned>` | `String` | `T` | `entelix-runnable::parser` |
| `RetryParser<P>` | `String` | `<P as Runnable>::O` | `entelix-runnable::parser` |
| `FixingOutputParser<P, F>` | `String` | `<P as Runnable>::O` | `entelix-runnable::parser` |
| `Tool` (via `ToolToRunnableAdapter`) | `serde_json::Value` | `serde_json::Value` | `entelix-core::tools` |
| `SchemaTool` (via `SchemaToolAdapter`) | `T::Input` | `T::Output` | `entelix-tools` |
| `Retriever` (via `RetrieverToRunnableAdapter`) | `String` | `Vec<Document>` | `entelix-memory` |
| `CompiledGraph<S>` | `S` | `S` | `entelix-graph` |
| `MergeNodeAdapter<S, U, F>` (delta-style node) | `S` | `S` | `entelix-graph` |
| `ContributingNodeAdapter<S>` (Contribution-style node) | `S` | `S` | `entelix-graph` |

## Type erasure — when generics get in the way

For dynamic dispatch (e.g., agent picking tools at runtime), we expose a Value-erased counterpart:

```rust
#[async_trait]
pub trait AnyRunnable: Send + Sync + 'static {
    fn name(&self) -> Cow<'_, str> { Cow::Borrowed("any-runnable") }
    async fn invoke_any(
        &self,
        input: serde_json::Value,
        ctx: &ExecutionContext,
    ) -> Result<serde_json::Value>;
}

pub type AnyRunnableHandle = Arc<dyn AnyRunnable>;

// `entelix-runnable::any_runnable::erase` wraps any `Runnable<I, O>`
// (with `I: DeserializeOwned`, `O: Serialize`) into a value-erased
// `AnyRunnable` by transcoding through JSON at the boundary.
let erased: AnyRunnableHandle = Arc::new(erase(typed_runnable));
```

The trait surface is intentionally narrow — `name` + `invoke_any`.
Streaming over the value-erased path stays on the typed
[`Runnable::stream`] surface; the erased boundary is for one-shot
dispatch tables (tool registries, plug-in routers).

## Tool-vs-Runnable relationship (ADR-0011 territory)

`Tool` does **not** extend `Runnable`. Reasons:
- `Tool` lives in `entelix-core` (DAG root); `Runnable` lives in `entelix-runnable` (depends on core).
- Trait inheritance across crates couples them in the wrong direction.
- `Tool` carries metadata (`name`, `description`, `input_schema`) that `Runnable<Value, Value>` doesn't need.

Instead, **`ToolToRunnableAdapter`** in `entelix-runnable::adapter` exposes any `Tool` as a `Runnable<Value, Value>`:

```rust
use entelix::{RunnableExt, ToolToRunnableAdapter};

let runnable_tool = ToolToRunnableAdapter::new(my_tool);
let chain = prompt.pipe(model).pipe(runnable_tool);   // works
```

This keeps both contracts pure. Decision recorded in ADR-0011.

## Streaming

`ChatModel::stream_deltas` returns a `BoxStream<Result<StreamDelta>>`
where the `StreamDelta` enum carries every increment that flows out
of a streaming model call:

```rust
pub enum StreamDelta {
    Start { id: String, model: String },
    TextDelta { text: String },
    ThinkingDelta { text: String, signature: Option<String> },
    ToolUseStart { id: String, name: String },
    ToolUseInputDelta { partial_json: String },
    ToolUseStop,
    Usage(Usage),
    RateLimit(RateLimitSnapshot),
    Warning(ModelWarning),                   // invariant 15 (no silent fallback)
    Stop { stop_reason: StopReason },
}
```

For `CompiledGraph<S>::stream`, the higher-level `RunnableEvent`
enum yields graph-level events (per-node `Update`, lifecycle
`Started` / `Complete`, etc.) selected by `StreamMode`
(`Values` / `Updates` / `Messages` / `Debug` / `Events`). See
`crates/entelix-runnable/src/stream.rs` for the full surface.

## Example — full LCEL chain

```rust
use std::sync::Arc;
use entelix::{ChatModel, JsonOutputParser, RunnableExt};
use entelix::auth::ApiKeyProvider;
use entelix::codecs::AnthropicMessagesCodec;
use entelix::transports::DirectTransport;
use entelix::{ChatPromptTemplate, MessagesPlaceholder};

let prompt = ChatPromptTemplate::from_messages([
    ("system", "You translate to {language}."),
    MessagesPlaceholder::new("history"),
    ("user", "{question}"),
]);

let creds = Arc::new(ApiKeyProvider::anthropic(std::env::var("ANTHROPIC_API_KEY")?));
let model = ChatModel::new(
    AnthropicMessagesCodec::new(),
    DirectTransport::anthropic(creds)?,
    "claude-opus-4-7",
);

let parser = JsonOutputParser::<TranslationResult>::new();
let chain = prompt.pipe(model).pipe(parser);

let result: TranslationResult = chain
    .invoke(input, &ctx)
    .await?;
```

## Performance notes

- `RunnableSequence` is zero-cost over hand-written sequential code (Arc clone on construction, normal calls thereafter).
- `RunnableParallel` uses `tokio::try_join_all` internally; no thread-pool overhead.
- `AnyRunnable::invoke_any` pays JSON ser/deser cost — use typed `Runnable` for hot paths.

## Resolved design questions

These were open at design time; 1.0 RC closes them:

- **Variadic `pipe` macro** — *not added*. The `.pipe(...).pipe(...)` chain is fluent enough; a macro would compete with the typed builder pattern without ergonomic gain. Reserved for a 1.x slice if real demand surfaces.
- **Async closures in `RunnableLambda`** — *closed by Rust 1.94*. Async closures are stable; `RunnableLambda::new(|x, ctx| async move { ... })` works without boxed-future workarounds for the typical case. `'static` captures still apply to the closure body itself, matching every other async closure in the workspace.

## Cross-references

- ADR-0006 — Runnable + StateGraph 1.0 spine.
- ADR-0010 — naming taxonomy (`Runnable<Verb>` composition prefix, `*Adapter` for trait converters).
- ADR-0011 — `Tool` / `Runnable` adapter boundary, `SchemaTool` typed-I/O.
- `docs/architecture/state-graph.md` — `CompiledGraph<S>` is a `Runnable<S, S>`; this is the "subgraph" composition.
