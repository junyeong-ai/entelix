---
paths:
  - "**/*.rs"
---

# Naming taxonomy

Canonical source: `docs/adr/0010-naming-taxonomy.md`.

## Type suffix — fixed semantics

| Suffix | Meaning | Example |
|---|---|---|
| `*Codec` | encoder/decoder, **stateless** | `AnthropicMessagesCodec` |
| `*Transport` | HTTP transport, **stateful** (auth, conn pool) | `BedrockTransport` |
| `*Provider` | async source-of-truth for ONE thing | `CredentialProvider` |
| `*Registry` | runtime collection of named entries (`register` / `unregister` / lookup); usually init-time but runtime mutation is allowed when the entries are operator-scoped (tenants, plugins) rather than wire-protocol-scoped | `ToolRegistry`, `SkillRegistry`, `PolicyRegistry` |
| `*Manager` | runtime lifecycle (start/stop/refresh) | `McpManager` |
| `*Store` | persistent KV-style storage | `InMemoryStore`, `PostgresStore` |
| `*Builder` | fluent construction | `AgentBuilder` |
| `*Config` | config data struct | `RetryConfig` |
| `*Context` | request-scope state | `ExecutionContext`, `ConsolidationContext` |
| `*Snapshot` | immutable point-in-time | `RateLimitSnapshot`, `TokenSnapshot` |
| `*Aggregator` | accumulator over time | `StreamAggregator` |
| `*Adapter` | converts trait A → trait B | `ToolToRunnableAdapter` |
| `*Router` | dispatches | `RunnableRouter` |
| `*Reducer` | state merge function (trait + built-in impls) | `Reducer<T>`, `Append<T>`, `MergeMap<K,V>` |
| `*Checkpointer` | StateGraph state persistence | `PostgresCheckpointer` |
| `*Limiter` | rate / quota | `RateLimiter` |
| `*Redactor` | data scrubbing | `PiiRedactor` |
| `*Meter` | metric accumulation | `CostMeter` |
| `*Client` | HTTP / RPC remote client, **stateful** | `HttpMcpClient` |
| `*Decoder` | binary / wire-format parser, **stateful buffer** | `EventStreamDecoder` |
| `*Signer` | request signing (SigV4, JWT, …) | `SigV4Signer` |
| `*Guard` | RAII scoped resource hold | `LockGuard` |
| `*SessionLog` | append-only event log per session/thread | `PostgresSessionLog` |
| `*Persistence` | aggregate facade (checkpointer + store + lock) | `PostgresPersistence` |
| `*Layer` | `tower::Layer<S>` middleware factory (pairs 1:1 with `*Service`) | `PolicyLayer`, `OtelLayer` |
| `*Agent` | runtime entity wrapping `Runnable<S, S>` + event stream | `Agent<ReActState>` |
| `*Event` | enum of runtime events | `AgentEvent<S>` |
| `*Sink` | consumer trait + impls (events go here) | `AgentEventSink`, `BroadcastSink` |
| `*Handle` | refcounted access wrapper around an `Arc<dyn ...>` trait object (type-erased, rides cross-cutting carriers like `ExecutionContext` extensions) **or** an `Arc<concrete>` with RAII drop semantics (e.g. flush-on-drop for OTel exporters) | `AuditSinkHandle`, `ToolApprovalEventSinkHandle`, `OtlpHandle` |
| `*Mode` | enum of behavior switches | `ExecutionMode`, `StreamMode`, `TenantMode` |
| `*Observer` | stateful agent-lifecycle observer | `AgentObserver` |
| `*Sandbox` | sandbox-agnostic isolated execution environment trait | `Sandbox`, `MockSandbox` |
| `*Policy` | declarative gate / decision data | `ShellPolicy`, `TenantPolicy` |
| `*Spec` | declarative input contract — validated at construction, behavior-free | `JsonSchemaSpec`, `ToolSpec`, `CommandSpec` |
| `*Format` | structural rule for output shape — rides the IR to the wire | `ResponseFormat` |
| `*Memory` | stateful conversation/working-memory facade over a `Store<V>` | `BufferMemory`, `EntityMemory`, `SummaryMemory`, `SemanticMemory` |
| `*Ext` | extension trait — provided methods on a base trait, blanket-impl'd for every base impl | `RunnableExt`, `SchemaToolExt` |

## `Runnable<Verb>` composition prefix

Composition primitives in `entelix-runnable` use a `Runnable<Verb>` **prefix**:

| Pattern | Operation | Example |
|---|---|---|
| `RunnableLambda` | wrap a closure as `Runnable<I,O>` | `RunnableLambda::new(|x| async move {…})` |
| `RunnableSequence` | pipe two runnables (`a.pipe(b)`) | returned by `RunnableExt::pipe` |
| `RunnableParallel` | fan-out one input to N runnables, collect | LangChain parity |
| `RunnableRouter` | dispatch on input (also matches `*Router`) | LangChain parity |
| `RunnablePassthrough` | identity — placeholder in graph slots | placeholder |

Trait-to-trait converters keep `*Adapter` (`ToolToRunnableAdapter`); output extractors keep `*Parser` (`JsonOutputParser`).

## Forbidden suffixes (too vague)

| Forbidden | Use instead |
|---|---|
| `*Engine` | name the operation (`OrchestrationLoop`) |
| `*Wrapper` | `*Adapter` for conversion, otherwise be specific |
| `*Handler` | `*Processor`, `*Consumer`, or `*Receiver` |
| `*Helper`, `*Util` | name the module instead |
| `*Service` (in core libs) | `*Manager` (lifecycle) or `*Client` (HTTP) |

## Trait names

- **Capability nouns**: `Codec`, `Transport`, `Tool`, `Runnable`, `Reducer`, `Embedder`, `Retriever`, `Persistence`, `Checkpointer`
- Avoid `*able`/`*ible` adjective traits unless aligning with std (`Clone`, `Send`)
- Sealed: `pub(crate) mod private { pub trait Sealed {} }` + `pub trait Foo: private::Sealed`

## Method names

| Pattern | Use for |
|---|---|
| `new(args) -> Self` | infallible default constructor |
| `with_xxx(self, x) -> Self` | builder option setter |
| `Builder::build() -> Result<T>` | builder finalizer — always `Result` even when validation cannot fail today, so adding validation later is non-breaking |
| `name(&self) -> &str` | accessor — **never `get_name`** |
| `set_name(&mut self, n: String)` | mutator |
| `try_xxx(...) -> Result<T>` | fallible variant |
| `xxx_async(...)` | only if a sync `xxx()` also exists |
| `xxx_dyn(...)` | object-safe sibling of a generic `xxx<T>(...)` method on the same type — present only when the generic version exists and an `Arc<dyn …>` consumer needs to dispatch dynamically |
| `into_xxx(self) -> X` | consumes self |
| `as_xxx(&self) -> &X` | borrows |
| `to_xxx(&self) -> X` | clones / converts |

Async-by-default: most entelix APIs are `async`. Use `_async` suffix only when a sync sibling exists.

### Builder verb-prefix exception — `add_*` / `set_*` / `register` for graph and collection construction

`with_*` covers configuration builders (knobs on a single shaped target). Topology / collection construction uses domain verbs — they read more naturally and signal intent (insert vs. configure) at the call site. Both shapes consume `self` and return `Self` (or `Result<Self>` for fallible registry inserts).

| Prefix | Use for | Examples |
|---|---|---|
| `with_<noun>(self, x)` | configuration setter on a single target | `RetryConfig::with_max_attempts`, `AgentBuilder::with_timeout` |
| `add_<element>(self, …)` | append to an internal collection | `StateGraph::add_node`, `StateGraph::add_edge` |
| `set_<role>(self, …)` | designate one element as a named role | `StateGraph::set_entry_point`, `StateGraph::set_finish_point` |
| `register(self, …) -> Result<Self>` | append to a registry collection with validation, init-time | `ToolRegistry::register`, `SkillRegistry::register` |
| `register(&self, …)` (+ `unregister(&self, …)`) | runtime-mutable registry of operator-scoped entries (tenants, plugins) | `PolicyRegistry::register` |
| `restrict_to(self, …)` / `filter(self, F)` | selection / narrowing — *which subset* the builder targets, not a configuration value | `SubagentBuilder::restrict_to`, `SubagentBuilder::filter` |

These are the **only** authorized exceptions to `with_*`. Do not introduce new verb prefixes (`define_*`, `attach_*`, `mount_*`). If unsure, reach for `with_` / `add_` / `set_` / `register` in that order. Selection verbs (`restrict_to` / `filter`) are reserved for builders that produce *narrowed views* of an existing parent set.

Canonical: ADR-0010 §"Builder verb-prefix exception".

## `ctx` parameter ordering — split convention

Two conventions, both intentional. Don't reorder a method's `ctx` without an ADR amendment.

- **ctx-first** — memory and persistence backends. Reads as "for this context, do X with Y": `Store`, `VectorStore`, `SemanticMemory(Backend)`, `BufferMemory`, `EntityMemory`, `SummaryMemory`, `GraphMemory`, `Checkpointer`, `SessionLog`. Shape: `(&self, ctx, scope, payload...)` where scope is `namespace` / `thread_id` / `doc_id`.
- **ctx-last** — computation and dispatch. Primary input dominates; ctx is the ambient request envelope: `Tool::execute(input, ctx)`, `Embedder::embed(text, ctx)`, `Retriever::retrieve(query, ctx)`, `Reranker::rerank(query, candidates, top_k, ctx)`, `Runnable::invoke(input, ctx)`. Observer/decision traits also follow this shape: `Approver::decide(request, ctx)`, `AgentObserver::pre_turn(state, ctx)`.
- **No ctx, sync** — `Reducer::reduce(&self, current: T, update: T) -> T` is pure and synchronous (entelix-graph CLAUDE.md: "Reducers must be pure — no IO, no `Send` futures"). The dispatch loop calls it inside the super-step lock; ctx would invite IO that the contract forbids.
- **No ctx** — `AgentEventSink::send(event)`. Events carry their own correlation fields, so sinks observe the event without the ambient request.
- **Purpose-built context** — `ConsolidationPolicy::should_consolidate(&ConsolidationContext)`. When a trait's decision inputs don't fit `ExecutionContext`, give it its own `*Context` type instead of forcing the split.

Canonical: ADR-0010 §"Parameter ordering — `ctx` placement".

## Module names

- **Plural** for collections: `tools/`, `codecs/`, `transports/`, `agents/`
- **Singular** for concepts: `error.rs`, `ir.rs`, `auth.rs`, `runnable.rs`

## Feature flags

- One lowercase word: `mcp`, `postgres`, `otel`, `aws`, `gcp`, `azure`, `policy`, `server`
- No dashes/underscores
- `full` aggregates everything

## Constants

- `SCREAMING_SNAKE_CASE`
- Type-namespace when ambiguous: `Self::DEFAULT_TIMEOUT_MS`

## Error variants

- `PascalCase` nouns: `Invalid`, `Provider`, `Network`, `Timeout`
- Each variant carries an actionable message
- Provider failures include `hint: Option<String>`
- `#[error(transparent)]` for pass-through wrappers
- Top-level type: `Error`. Module-internal: `FooError`.

## Crate names

- Workspace: `entelix-{role}` (kebab-case, single role word)
- External companions (1.1+): `entelix-{role}-{detail}` — e.g. `entelix-embedder-openai`

## Self-check before commit (`cargo xtask naming` runs this in CI)

```bash
# forbidden suffixes
rg 'pub (struct|enum|trait) \w+(Engine|Wrapper|Handler|Helper|Util)\b' crates/

# get_X accessor
rg 'fn get_\w+\(&self' crates/
```

Both must return zero hits.
