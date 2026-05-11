---
paths:
  - "**/*.rs"
---

# Naming taxonomy

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
| `*Gate` | check / predicate that fails closed on detection (xtask invariants, future runtime policy gates). A `CadenceStep` typedef is the orthogonal "one step in a sequenced cadence" shape — sequencing is not gating, do not conflate. | `FileGate`, `WorkspaceGate`, `NamingGate` |

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
| `set_name(&mut self, n: String)` | mutator (uniquely-owned `&mut self`) |
| `replace_name(&self, n: T)` | interior-mutable hot-swap of a single field on a share-cloneable type (`Arc<Self>` clones can replace from any holder). Pairs with a `RwLock` / `ArcSwap` field. Distinct from `set_*` so the call site signals share-mutation, not exclusive ownership. |
| `try_xxx(...) -> Result<T>` | fallible variant |
| `xxx_async(...)` | only if a sync `xxx()` also exists |
| `xxx_dyn(...)` | object-safe sibling of a generic `xxx<T>(...)` method on the same type — present only when the generic version exists and an `Arc<dyn …>` consumer needs to dispatch dynamically |
| `into_xxx(self) -> X` | consumes self |
| `as_xxx(&self) -> &X` | borrows |
| `to_xxx(&self) -> X` | clones / converts |

Async-by-default: most entelix APIs are `async`. Use `_async` suffix only when a sync sibling exists.

### Persistence read verb-family taxonomy

Persistence-trait read methods (`Store`, `VectorStore`, `GraphMemory`, `Checkpointer`, `SessionLog`, …) follow a four-verb taxonomy. The shape — single-item lookup, paginated enumeration, cursor-based stream, or query-shaped multi-result — picks the verb; pick-by-noun (`.node()`, `.edge()`, `.latest()`) is reviewer-rejected because the noun's shape is not self-evident at the call site.

| Verb-family | Use for | Example |
|---|---|---|
| `get_xxx(.) -> Result<Option<T>>` | single-item primary-key lookup. Always returns `Option<T>` — absence is a normal outcome, not an error. | `Checkpointer::get_latest`, `Checkpointer::get_by_id`, `GraphMemory::get_node`, `GraphMemory::get_edge`, `Store::get` |
| `list_xxx(.)` | paginated enumeration. Operator-time admin shape — pages of ids or full records, no cursor semantics implied. | `GraphMemory::list_nodes`, `PgGraphMemory::list_node_records` |
| `load_since(.)` / `load_until(.)` | cursor-based stream. The `since/until` suffix carries the temporal bound. | `SessionLog::load_since` |
| `find_xxx(.)` / `search_xxx(.)` | query-shaped multi-result. `find` for typed predicates, `search` for ranked retrieval. | `VectorStore::search`, `GraphMemory::find_path` |

`get_xxx` is *only* forbidden for **field accessors** (`get_name(&self) -> &str` is reviewer-rejected — use `name(&self)`). Method names that take parameters and return `Option<T>` are not field accessors and the `get_xxx` form is the canonical persistence-read shape.

### Batch placement

Batch variants of CRUD operations use the **suffix** form (`add_batch`, never `batch_add`) so the operation name reads as a noun-verb chunk modified by `_batch`. Mirrors the `*_batch` taxonomy in `Embedder::embed_batch` and `GraphMemory::add_edges_batch`.

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

### IR state-transition verbs

Builder taxonomy above covers configuration setters and collection inserts on construction-time targets. IR value types (`ModelRequest`, `ModelResponse`, …) additionally expose **state-transition** methods that advance the value to a derived value reflecting a conceptual step. These are not builders — they transform existing semantically-meaningful state — and use a domain verb that names the transition.

| Verb | Use for | Example |
|---|---|---|
| `continue_<noun>(self, …)` | advance an IR value to its post-transition state — round-trip conversational state, chain opaque vendor echoes, etc. Consumes `self`, returns `Self`. | `ModelRequest::continue_turn(prior_response, next_message)` |

State-transition verbs are reserved for IR-level value advancement where (a) the operation has a single clearly-named conceptual unit (`turn`, …) the caller already understands and (b) the transformation touches more than one field, so the builder taxonomy's "single shaped target" framing is misleading. Reviewer-rejected outside that narrow case — recipe-side or service-side state changes use `tower::Service` dispatch or graph nodes, not IR helper methods.


## `ctx` parameter ordering — split convention

Two conventions, both intentional. Don't reorder a method's `ctx` without amending this rule and the affected trait deliberately.

- **ctx-first** — memory and persistence backends. Reads as "for this context, do X with Y": `Store`, `VectorStore`, `SemanticMemory(Backend)`, `BufferMemory`, `EntityMemory`, `SummaryMemory`, `GraphMemory`, `Checkpointer`, `SessionLog`. Shape: `(&self, ctx, scope, payload...)` where scope is `namespace` / `thread_id` / `doc_id`.
- **ctx-last** — computation and dispatch. Primary input dominates; ctx is the ambient request envelope: `Tool::execute(input, ctx)`, `Embedder::embed(text, ctx)`, `Retriever::retrieve(query, ctx)`, `Reranker::rerank(query, candidates, top_k, ctx)`, `Runnable::invoke(input, ctx)`. Observer/decision traits also follow this shape: `Approver::decide(request, ctx)`, `AgentObserver::pre_turn(state, ctx)`.
- **No ctx, sync** — `Reducer::reduce(&self, current: T, update: T) -> T` is pure and synchronous (entelix-graph CLAUDE.md: "Reducers must be pure — no IO, no `Send` futures"). The dispatch loop calls it inside the super-step lock; ctx would invite IO that the contract forbids.
- **No ctx** — `AgentEventSink::send(event)`. Events carry their own correlation fields, so sinks observe the event without the ambient request.
- **Purpose-built context** — `ConsolidationPolicy::should_consolidate(&ConsolidationContext)`. When a trait's decision inputs don't fit `ExecutionContext`, give it its own `*Context` type instead of forcing the split.


## Module names

- **Plural** for collections: `tools/`, `codecs/`, `transports/`, `agents/`
- **Singular** for concepts: `error.rs`, `ir.rs`, `auth.rs`, `runnable.rs`

## Feature flags

- Lowercase. Compound names use **hyphens** (matches Cargo's TOML-key convention); no underscores.
- Sub-namespacing reads `<role>-<detail>` (`embedders-<vendor>`, `vectorstores-<backend>`); single-word features describe the role directly.
- `full` aggregates everything.
- Canonical list lives in the facade `crates/entelix/Cargo.toml` — that is the single source of truth.

## Constants

- `SCREAMING_SNAKE_CASE`
- Type-namespace when ambiguous: `Self::DEFAULT_TIMEOUT_MS`

## Error variants

- `PascalCase` nouns: `Invalid`, `Provider`, `Network`, `Timeout`
- Each variant carries an actionable message
- `#[error(transparent)]` for pass-through wrappers
- Top-level type: `Error`. Module-internal: `FooError`.

## Crate names

- Workspace: `entelix-{role}` (kebab-case, single role word)
- External companions (1.1+): `entelix-{role}-{detail}` — e.g. `entelix-embedder-openai`

## Enforcement

`cargo xtask naming` runs the typed-AST visitor in CI. Forbidden suffixes, `get_*` accessors, builder verb-prefix violations, and `ctx`-position drift all fail the gate.
