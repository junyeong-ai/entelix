# Architecture Principles

This document is the living design contract for entelix. It captures the
current rules future agents should preserve while changing the codebase. It is
not a history log.

## Agent SDK Position

entelix is a general-purpose Rust SDK for agents that need model calls,
tooling, memory, durable state, observability, policy, and provider routing.
It must stay domain-neutral: service-specific workflows, vertical policies,
and deployment opinions belong behind traits, layers, feature flags, or
companion crates.

## Core Boundaries

- `entelix-core` is the DAG root. It owns provider-neutral IR, codecs,
  transports, auth, tools, layers, model settings, and model catalogs.
- User-facing docs prefer the facade path (`entelix::Type`) unless a crate-local
  rule is being documented.
- Provider JSON stays inside codecs. Public APIs carry `ModelRequest`,
  `ModelResponse`, typed warnings, typed stop reasons, and typed settings.
- Concrete persistence, cloud, vector, graph-memory, and embedding backends live
  in companion crates so the default surface stays small.

## State Model

- `StateGraph` state is per-thread working memory.
- `SessionGraph` events are the append-only audit source of truth.
- `Memory Store`, `VectorStore`, and `GraphMemory` are cross-thread persistent
  knowledge surfaces.
- `AuditSink` is a typed channel into the event log, not a fourth state store.
  It records managed-agent lifecycle events without making tool, graph, or
  memory crates depend on session storage.

## Model And Provider Surface

- `ChatModel<C, T>` remains the dispatch primitive: codec plus transport plus
  `tower::Layer` stack.
- `ModelCatalog` is a named endpoint table, not a client pool. Operators build
  fully configured `ChatModel` values and register them under stable names.
  `RunOverrides::with_model_endpoint` is the only per-call catalog routing
  knob; unknown endpoint keys fail as config errors instead of falling back.
- `RunOverrides` carries per-call overrides through `ExecutionContext` for
  model endpoint selection, provider model-id selection on an already-selected
  endpoint, system prompt, structured-output response format, model tuning
  settings, prompt-cache routing, reasoning effort, and max-iteration lowering.
  Catalog endpoint routing and provider model-id rewriting stay separate.
- Agent run defaults merge by explicit specificity: reusable capability
  defaults first, concrete builder defaults second, call-scoped
  `RunOverrides` last. Provider or model names must not trigger hidden default
  selection in recipe code.
- `ModelSettings` carries reusable model tuning settings. Portable fields stay
  first-class; provider-specific settings live in typed extension structs under
  `ProviderExtensions`. Codecs reject unsupported mandatory combinations rather
  than silently approximating.
- Structured output supports native and tool-call strategies. Unsupported
  prompt-parser fallback should not be added unless it has a concrete provider
  gap and a typed validation story.

## Tools And Sub-Agents

- `Tool<D = ()>::execute(input, ctx)` is the single behavioural hand contract.
  Metadata is inspectable; execution stays side-effect explicit.
- `AgentContext<D>` carries operator-supplied typed dependencies to leaf tools.
  The layer stack consumes a `D`-free `ToolInvocation` so policy, approval,
  retry, OTel, scoped ambient state, and tool-progress layers compose for every
  dependency type.
- `Toolset<D = ()>` is the reusable declaration surface for tool bundles.
  It installs into a `ToolRegistry<D>` and never dispatches directly. This keeps
  reusable capabilities composable without creating a second tool execution
  path. Provider-neutral run defaults (catalog endpoint, system prompt, model
  settings, prompt-cache routing, reasoning effort, graph iteration caps) ride
  on `RunOverrides` attached to `ExecutionContext` so caller-supplied overrides
  remain the final authority.
- `ToolHook` is the typed control surface for tool lifecycle policy. Pre-hooks
  may continue, replace JSON input, or reject; success hooks may enforce
  post-conditions; error hooks observe without masking the original failure.
  Hooks are middleware on the single `ToolRegistry` dispatch path, not a second
  tool execution mechanism.
- `SchemaTool` and `#[tool]` are the default authoring surface for typed I/O.
  Manual `Tool` implementations are for state-rich or highly custom tools.
- Sub-agents receive a narrowed view of the parent registry and inherit the
  parent layer stack. They must not create a fresh registry or bypass policy.
- Skills use progressive disclosure: list, activate, then read resources.
  Bulk-injecting every skill body into the model context is a token and
  reliability bug.

## Composition And Middleware Boundary

- `Runnable<I, O>` is the graph IR — what an agent *computes* (codecs, prompts,
  parsers, sub-agents, compiled state graphs all implement it; `.pipe()` is the
  universal connector). `tower::Service<Request>` is the middleware wiring —
  *how* a model or tool dispatch is mediated (`PolicyLayer`, `OtelLayer`,
  `RetryService`, `ApprovalLayer`, `ToolHookLayer`, `ToolEventLayer`,
  `ScopedToolLayer` plug onto `ModelInvocation` and `ToolInvocation`).
  The two domains are intentionally disjoint — adapters that convert one to
  the other are reviewer-rejected because they would smuggle middleware
  concerns into composition logic, or vice versa.

## Observability And Policy

- One-shot and streaming model calls use the same `tower::Service` spine so
  retry, OTel, cost, policy, and audit behaviour stay symmetric.
- Cost is emitted only after successful calls. Failed model/tool/embedding
  operations must not generate spend.
- Lossy codec decisions emit typed warnings. Unknown vendor stop reasons stay
  visible as typed `Other` values.
- Heuristics belong on typed `*Policy`, `*Decision`, or settings surfaces.
  Dispatch hot paths must not hide routing, fallback, retry, or probability
  literals that operators cannot override.
- LLM-facing content passes through explicit rendering surfaces. Operator
  diagnostics, raw provider errors, timestamps, internal type names, and
  unfiltered metadata do not enter model context.

## Security

- First-party crates do not perform filesystem or shell syscalls. Sandboxed
  tools delegate through `Sandbox`; concrete OS sandboxes are external
  deployment concerns or companion crates.
- Credentials live in auth providers and transports. Tool input, tool context,
  memory namespaces, and public event logs never carry provider tokens.
- Tenant identity is mandatory at every persistent boundary. Postgres-backed
  stores use tenant-scoped transactions and row-level security as defense in
  depth.
- Lock ordering is `tenant > session > checkpoint > memory > tool_registry >
  orchestrator`; never await user code while holding a lock.

## Naming

- Trait implementation types use precise suffixes: `*Codec`, `*Transport`,
  `*Provider`, `*Store`, `*Retriever`, `*Checkpointer`, `*Reducer`, `*Layer`,
  `*Service`, `*Adapter`, `*Registry`, `*Catalog`, `*Policy`, `*Decision`,
  `*Sink`, `*Tool`, and `*Builder`.
- Avoid vague suffixes such as `*Engine`, `*Wrapper`, `*Handler`, `*Helper`, and
  `*Util`.
- Accessors use the bare noun, not `get_*`.
- Builders use `with_*` for setters, `add_*` for accumulating entries,
  `register` for registry insertion, and `build` for fallible finalization.

## Documentation Policy

- Keep this file as the living design contract. Anything that is not a current
  rule does not belong here.
- Per-crate `crates/<name>/CLAUDE.md` files carry the local surface, crate-local
  rules, and forbidden patterns for each sub-crate.
- Keep `docs/public-api` because it is a CI baseline, not prose.
- Do not reintroduce per-slice retrospectives, roadmap fragments, or deferred
  historical plans as source-of-truth documentation. Current behaviour belongs
  in `CLAUDE.md`, this file, sub-crate `CLAUDE.md` files, tests, invariants,
  and public API baselines.
