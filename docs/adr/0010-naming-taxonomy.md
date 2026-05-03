# ADR 0010 — Naming taxonomy

**Status**: Accepted
**Date**: 2026-04-26

## Context

A large unconstrained codebase accumulates inconsistent naming: the same role gets different suffixes (`*Manager` / `*Service` / `*Engine` for similar concepts), and trait names mix adjectives (`*able`) with capability nouns (`Codec`, `Tool`). Reviewers end up memorising per-module conventions.

For entelix's 15-crate workspace, naming consistency is a maintainability invariant. A reader should infer a type's role from its suffix without reading code.

## Decision

This taxonomy is **load-bearing**. CI checks via `clippy::similar_names` and reviewer enforcement.

### Type-suffix taxonomy

| Suffix | Semantics | Lifetime | Examples |
|---|---|---|---|
| `*Codec` | encoder/decoder, **stateless** | static | `AnthropicMessagesCodec`, `BedrockConverseCodec` |
| `*Transport` | HTTP transport, **stateful** (auth, conn pool) | runtime | `DirectTransport`, `BedrockTransport` |
| `*Provider` | async source-of-truth for ONE thing | runtime | `CredentialProvider`, `ModelProvider` |
| `*Registry` | init-time, append-only collection | runtime | `ToolRegistry`, `HookRegistry` |
| `*Manager` | runtime lifecycle (start/stop/refresh) | runtime | `McpManager` |
| `*Store` | persistent KV-style storage | durable | `InMemoryStore`, `PostgresStore` |
| `*Builder` | fluent construction | scoped | `AgentBuilder`, `StateGraphBuilder` |
| `*Config` | config data struct | static | `RetryConfig`, `ModelConfig` |
| `*Context` | request-scope state | request | `ExecutionContext`, `TenantContext` |
| `*Snapshot` | immutable point-in-time data | static | `StateSnapshot`, `CheckpointSnapshot` |
| `*Aggregator` | accumulator over time | scoped | `StreamAggregator`, `CostAggregator` |
| `*Adapter` | converts trait A → trait B | static | `ToolToRunnableAdapter` |
| `*Router` | dispatches based on input | runtime | `RunnableRouter` |
| `*Reducer` | state merge function | runtime | `MessagesReducer`, `MaxReducer` |
| `*Checkpointer` | StateGraph state persistence | runtime | `InMemoryCheckpointer`, `PostgresCheckpointer` |
| `*Limiter` | rate / quota enforcement | runtime | `RateLimiter`, `QuotaLimiter` |
| `*Redactor` | data scrubbing | runtime | `PiiRedactor` |
| `*Meter` | metric accumulation | runtime | `CostMeter` |
| `*Client` | HTTP / RPC remote client, **stateful** (auth, conn, retry) | runtime | `HttpMcpClient`, `OAuthClient` |
| `*Decoder` | binary / wire-format parser, **stateful buffer** | runtime | `EventStreamDecoder` |
| `*Signer` | request-signing (SigV4, JWT, …) | runtime | `SigV4Signer` (Phase 5) |
| `*Guard` | RAII-style scoped resource hold; warns on `Drop` if not released | scoped | `LockGuard` |
| `*SessionLog` | append-only event log per session/thread | durable | `PostgresSessionLog`, `RedisSessionLog` |
| `*Persistence` | aggregate persistence facade (checkpointer + store + lock) | runtime | `PostgresPersistence` (composite, when needed) |
| `*Layer` | `tower::Layer<S>` middleware factory; pairs 1:1 with a `*Service` impl | runtime | `PolicyLayer`, `OtelLayer` |
| `*Agent` | runtime entity wrapping `Runnable<S, S>` + event stream | runtime | `Agent<ReActState>` |
| `*Event` | enum of runtime events | static | `AgentEvent<S>` |
| `*Sink` | consumer trait + impls (events go here) | runtime | `AgentEventSink`, `BroadcastSink`, `CaptureSink` |
| `*Mode` | enum of behavior switches | static | `ExecutionMode`, `StreamMode` |
| `*Observer` | stateful agent-lifecycle observer (turn / complete level) | runtime | `AgentObserver` |
| `*Sandbox` | sandbox-agnostic isolated execution environment trait | runtime | `Sandbox`, `MockSandbox` |
| `*Policy` | declarative gate / decision data | static | `ShellPolicy`, `CodePolicy`, `TenantPolicy` |
| `*Spec` | declarative input contract — validated at construction, carries no behavior | static | `JsonSchemaSpec`, `ToolSpec`, `CommandSpec`, `CodeSpec` |
| `*Format` | structural rule for output shape — riding the IR to the wire | static | `ResponseFormat` |
| `*Memory` | stateful conversation/working-memory facade over a `Store<V>` (LangChain-style pattern) | runtime | `BufferMemory`, `EntityMemory`, `SummaryMemory`, `SemanticMemory` |
| `*Ext` | extension trait — provided methods (`fn …(self) -> X`) on top of a base trait, blanket-impl'd for every base impl. Lives next to the base trait it decorates. | static | `RunnableExt` (decorates `Runnable`), `SchemaToolExt` (decorates `SchemaTool`) |

### `Runnable<Verb>` composition prefix (Phase 9 amendment)

Composition primitives in `entelix-runnable` use the `Runnable<Verb>`
**prefix** instead of a suffix. The prefix signals "this is a
composition primitive, not a domain type" — and the verb names
the operation it performs.

| Pattern | Operation | Examples |
|---|---|---|
| `RunnableLambda` | wrap a closure as `Runnable<I, O>` | `RunnableLambda::new(|x| async move { ... })` |
| `RunnableSequence` | pipe two runnables (`a.pipe(b)`) | the type behind `RunnableExt::pipe` |
| `RunnableParallel` | fan-out one input to N runnables, collect | LangChain-parity primitive |
| `RunnableRouter` | dispatch on input — also matches `*Router` suffix | LangChain-parity primitive |
| `RunnablePassthrough` | identity — useful in graph slots | placeholder node |

Trait-to-trait converters keep the `*Adapter` suffix
(`ToolToRunnableAdapter`) — that is a different category: it
**translates** a non-Runnable into a Runnable rather than composing
existing Runnables. Output extractors keep the `*Parser` suffix
(`JsonOutputParser`).

### Forbidden suffixes (vague — be specific)

| Forbidden | Why | Use instead |
|---|---|---|
| `*Engine` | too vague, overpromises | `OrchestrationLoop`, `RetryStrategy` |
| `*Wrapper` | reveals nothing | `ToolToRunnableAdapter` |
| `*Handler` | overloaded term | `RequestProcessor`, `EventConsumer` |
| `*Helper` | "miscellaneous" smell | name the thing it helps with |
| `*Util` | same | put functions in well-named modules |
| `*Service` | overloaded with web "service" | `FooManager` if lifecycle, `FooClient` if HTTP — **permitted exception** when the type directly impls `tower::Service` (ecosystem standard); CI gates check the impl, not the name |
| `*Wrapper` | (repeated for emphasis) | actual purpose |

### Trait naming

Traits are **capability nouns**:

✅ Allowed: `Codec`, `Transport`, `Tool`, `Runnable`, `Reducer`, `Embedder`, `Retriever`, `Persistence`, `Checkpointer`, `Hook`

❌ Avoid: `*able` / `*ible` adjectives unless aligning with std (`Clone`, `Debug`)

Sealed traits use the explicit `Sealed` boundary pattern:

```rust
mod private {
    pub trait Sealed {}
}

pub trait Foo: private::Sealed { ... }
```

### Method naming

| Pattern | When to use | Example |
|---|---|---|
| `new(args) -> Self` | infallible default constructor | `Vec::new()` |
| `with_xxx(self, x) -> Self` | builder option setter | `RetryConfig::default().with_max_attempts(5)` |
| `Builder::build() -> Result<T>` | builder finalizer (always Result) | `Agent::builder().build()?` |
| `name(&self) -> &str` | accessor — **NOT** `get_name` | `tool.name()` |
| `set_name(&mut self, n: String)` | mutator | `config.set_timeout(Duration::from_secs(30))` |
| `try_xxx(...) -> Result<T>` | fallible variant of infallible op | `Vec::try_reserve(usize)` |
| `xxx_async(...)` | only when sync `xxx()` exists in same trait | `read` + `read_async` |
| `into_xxx(self) -> X` | consumes self | `String::into_bytes` |
| `as_xxx(&self) -> &X` | borrows | `String::as_str` |
| `to_xxx(&self) -> X` | clones / converts | `String::to_owned` |

**Async by default**: most entelix APIs are async. We do NOT suffix every async method with `_async`. Only use `_async` when a sync version of the same name exists alongside.

#### Builder verb-prefix exception — `add_*` / `set_*` for graph and collection construction

The `with_*` builder convention covers **configuration** builders — knobs that fine-tune a single, already-shaped value. It does **not** cover **graph and collection construction**, where the call describes a topology operation (insert a node, draw an edge, designate the entry, register a tool). Forcing those onto `with_*` reads worse than the domain verbs and obscures intent — `with_node("planner", planner)` looks like a setter, not a graph mutation.

Two consumed-`self` builder verb-prefix shapes coexist by deliberate intent:

| Prefix | Meaning | Examples |
|---|---|---|
| `with_<noun>(self, x) -> Self` | configuration setter — overrides one value on a single shaped target | `RetryConfig::default().with_max_attempts(5)`, `AgentBuilder::with_timeout(d)` |
| `add_<element>(self, …) -> Self` | append element to an internal collection | `StateGraph::add_node`, `StateGraph::add_edge` |
| `set_<role>(self, …) -> Self` | designate one element as occupying a named role | `StateGraph::set_entry_point`, `StateGraph::set_finish_point` |
| `register(self, …) -> Result<Self>` | append to a registry collection with validation | `ToolRegistry::register`, `SkillRegistry::register` |

When a builder is shaped as "graph / topology / collection," prefer the topology verb. When it is shaped as "configuration / single-target tuning," use `with_<noun>`. The signature shape (`self`-consuming, returning `Self` or `Result<Self>`) stays uniform — only the verb changes to fit the operation. Mixing both prefixes on one type is acceptable when the type genuinely does both (e.g. `StateGraph::add_node` *and* `StateGraph::set_entry_point` *and* a hypothetical `StateGraph::with_recursion_limit(n)`), since each prefix communicates a different operation class.

This is the only authorized exception to the global `with_*` rule. It is **not** a license to introduce new verb prefixes — `define_*`, `attach_*`, `mount_*`, etc. should still resolve to one of `with_` / `add_` / `set_` / `register`.

### Parameter ordering — `ctx` placement

`&ExecutionContext` flows through almost every async surface in the SDK. Two ordering conventions coexist by deliberate intent — neither is wrong, and the split tracks the *kind* of method, not the crate it lives in.

**ctx-first** — for **memory and persistence backends** that operate over a tenant- and thread-scoped store. The signature reads as "for this context, do X with Y":

| Trait / type | Method shape |
|---|---|
| `entelix_memory::Store<V>` | `get(&self, ctx, namespace) -> Result<Option<V>>` |
| `entelix_memory::VectorStore` | `add(&self, ctx, namespace, doc, vector) -> Result<()>` |
| `entelix_memory::SemanticMemory` / `SemanticMemoryBackend` | `search(&self, ctx, query, top_k) -> Result<Vec<Document>>` |
| `entelix_memory::BufferMemory` / `EntityMemory` / `SummaryMemory` | `append(&self, ctx, ...)` |
| `entelix_memory::GraphMemory` | `add_node(&self, ctx, node) -> Result<NodeId>` |
| `entelix_persistence::Checkpointer` / `SessionLog` | `read(&self, ctx, thread_id) -> Result<...>` |

The argument list is always `(&self, ctx, scope, payload...)`: scope (`namespace`, `thread_id`, `doc_id`) names *which* slice of state the operation touches, payload describes *what* changes.

**ctx-last** — for **computation and dispatch traits** where the call's primary input dominates the signature and `ctx` is a side-channel for cancellation, deadlines, and run-id propagation:

| Trait / type | Method shape |
|---|---|
| `entelix_core::tools::Tool` | `execute(&self, input, ctx) -> Result<Value>` |
| `entelix_memory::Embedder` | `embed(&self, text, ctx) -> Result<Vec<f32>>` |
| `entelix_memory::Retriever` | `retrieve(&self, query, ctx) -> Result<Vec<Document>>` |
| `entelix_memory::Reranker` | `rerank(&self, query, candidates, top_k, ctx) -> Result<Vec<Document>>` |
| `entelix_runnable::Runnable<I, O>` | `invoke(&self, input, ctx) -> Result<O>` |

The argument list is `(&self, input..., ctx)`: the IR / payload is the *contract*, and `ctx` is the ambient request envelope that the implementation reads but does not key the operation on.

**Why the split is intentional, not lazy.** Conflating into a single rule would break readability on one side or the other. Memory backends without ctx-first would lose the "scope-comes-first" reading that makes `store.get(ctx, namespace, key)` parse as one phrase. Computation traits with ctx-first would push the operation's actual input — the thing a reader cares about — to the end, hiding intent. Both shapes are stable: PRs that reorder a method's `ctx` parameter need an ADR amendment, not a drive-by change.

**Observer / decision traits — ctx-last by default.** Lifecycle observers, approval gates, and reducers also accept `&ExecutionContext`, and they follow the ctx-last shape because the *thing being observed or decided on* is the primary input:

| Trait | Method shape |
|---|---|
| `entelix_agents::AgentObserver<S>` | `pre_turn(state, ctx)`, `on_complete(state, ctx)`, `on_failure(state, error, ctx)` |
| `entelix_agents::Approver` | `decide(request, ctx) -> ApprovalDecision` |
| `entelix_graph::Reducer<V>` | `reduce(prev, next, ctx) -> V` (when ctx-bearing) |

**Event sinks — no ctx parameter.** `AgentEventSink::send(event)` deliberately omits `&ExecutionContext` because every `AgentEvent<S>` already carries the correlation fields a sink needs (`run_id`, `tenant_id`, payload state). Threading ctx through `send` would duplicate state that already rides the event, and sinks must remain trivially composable across runs — so they observe the event, not the ambient request.

**Policy contexts — distinct from `ExecutionContext`.** `ConsolidationPolicy::should_consolidate(&ConsolidationContext)` carries a *purpose-built* context (`message_count`, `last_consolidated_at`, token-budget snapshot) rather than `ExecutionContext` because the decision is about memory state, not about the request. Distinct context types are encouraged whenever a trait's deciding inputs would be lost inside the generic `ExecutionContext` envelope; they get their own `*Context` name and don't fall under the ctx-first/ctx-last split.

### Module naming

- **Plural** for collections: `tools/`, `codecs/`, `transports/`, `agents/`
- **Singular** for concepts: `error.rs`, `ir.rs`, `auth.rs`, `runnable.rs`

### Feature flag naming

- Single lowercase word: `mcp`, `postgres`, `otel`, `aws`, `gcp`, `azure`, `policy`, `server`, `agents`, `memory`, `redis`
- No dashes, no underscores
- `full` aggregates everything

### Constant naming

- `SCREAMING_SNAKE_CASE`
- Type-namespace when ambiguous: `Self::DEFAULT_TIMEOUT_MS`, never bare `DEFAULT_TIMEOUT_MS` at crate root unless globally meaningful
- Avoid `MAX_*` / `MIN_*` without unit suffix: prefer `MAX_REQUEST_BYTES` over `MAX_REQUEST_SIZE`

### Error variants

- Variants are `PascalCase` nouns — `Invalid`, `Provider`, `Network`, `Timeout`
- Each variant carries actionable message
- Provider failures discriminate transport-class vs HTTP-class via the typed `kind: ProviderErrorKind` (`Network` / `Tls` / `Dns` / `Http(u16)`); retry classifiers branch on the typed signal
- `#[error(transparent)]` for pass-through wrappers
- Top-level enum is `Error`; module-internal enums are `FooError`

### Crate naming

- Workspace member crates: `entelix-{role}` — kebab-case, single role word
- Future companion crates outside workspace: `entelix-{role}-{detail}` — `entelix-embedder-openai`

## Consequences

✅ Reviewer doesn't memorize per-module conventions.
✅ New contributor inferns role from suffix.
✅ rustdoc surface is self-documenting.
✅ `cargo public-api` diff is human-readable (no naming churn noise).
❌ Some renames during early development as we converge — accepted, no backwards-compat (invariant 12 / ADR-0003 spirit).
❌ Slightly verbose names in some cases (`PostgresCheckpointer` over `PgSaver`) — accepted for clarity.

## Enforcement

1. **Code review** — primary gate.
2. **Clippy** — `similar_names`, `module_name_repetitions = "allow"` (intentional in workspace), `struct_excessive_bools = "warn"`.
3. **Custom lint** (Phase 5) — `dylint` rule for forbidden suffixes (`*Engine`, `*Wrapper`, `*Handler`, `*Helper`, `*Util`).
4. **Naming audit script** — `scripts/check-naming.sh` greps public types matching forbidden patterns; CI fails on hit.

## References

- Rust API guidelines — std naming conventions
- CLAUDE.md — full taxonomy table inline
