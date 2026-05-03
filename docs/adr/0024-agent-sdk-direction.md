# ADR 0024 — Agent SDK direction + Sandbox extension point

**Status**: Accepted
**Date**: 2026-04-27
**Effect**: commits entelix to the production agent-SDK shape.
Re-interprets Invariant 9 to clarify what is permanent (first-party
crates touch no fs/shell) vs what is extensible (the `Sandbox`
trait is a first-class delegation point).

## Context

After the 1.0 candidate hardening pass, an audit measured entelix's
surface against the production agent-SDK shape consumers expect.
The answer, by inventory, was that entelix shipped StateGraph +
Runnable + tower::Layer middleware but lacked the four primitives
every production agent SDK in 2026 had converged on:

| Production SDK | Agent | Event stream | Per-tool gating | Lifecycle hook |
|---|---|---|---|---|
| Anthropic Agents SDK | `Agent` | yes | tool_choice + permission | `on_*` callbacks |
| OpenAI Agents SDK | `Agent` | RunStep events | `Approval` callback | tool/run handlers |
| LangChain v1 | `AgentExecutor` | `AgentAction`/`AgentFinish` | `human_approval` | callbacks |
| LangGraph | (compiled graph) | streaming chunks | `interrupt_before` | listeners |
| CrewAI | `Crew` + `Agent` | task events | `human_input` | task callbacks |
| AutoGen | `ConversableAgent` | conversation events | `human_input_mode` | reply hooks |

Six independent SDKs converging on the same four-element
abstraction is the operational definition of a validated best
practice. entelix's current shape sits one level below at
"compile a graph and stream chunks" — academically correct, but
not what any 2026 production agent runtime exposes to its users.

A parallel audit raised the question: should Invariant 9
("no filesystem, no shell") forbid coding-style tools forever, or
open a clean delegation point so sandbox-mediated execution becomes
a first-class extension? The Anthropic managed-agents engineering
post (the founding reference for ADR-0005) is explicit: the
**Hand contract is sandbox-agnostic**. Tools talk to whatever
isolation the operator picks — local subprocess, E2B, Modal,
Fly.io machines, AWS Lambda, Cloudflare Workers, Kubernetes Job.
The harness does not care. That is the design that lets the same
agent run identically in three deployment environments by
swapping one component.

This ADR makes two commitments in one document because they are
the same architectural move: declare the agent-SDK direction, and
declare the sandbox delegation point that lets it carry coding-
agent workloads without compromising Invariant 9.

## Decisions

### 1. entelix becomes a production agent SDK

This direction adds an `Agent<S>` runtime that wraps any `Runnable<S, S>`
(typically a `CompiledGraph<S>`) plus an event sink, an execution
mode, and lifecycle observers. The existing `create_react_agent`
/ `create_supervisor_agent` / `create_hierarchical_agent` /
`create_chat_agent` recipes change their return type from
`Result<CompiledGraph<S>>` to `Result<Agent<S>>`. The graph layer
is preserved as the lower-level composition contract; `Agent<S>`
itself implements `Runnable<S, S>` so an agent is also a node in
a larger graph (recursive composition).

Rejected alternatives:
- *Stay graph-only.* entelix would be permanently academic — no
  production agent consumer can wire it as-is. Validated best
  practice across six SDKs disagrees.
- *Make `Agent` an opaque type with no `Runnable` impl.*
  Composability collapses; sub-agent dispatch loses the natural
  graph-node treatment.
- *Adopt a different agent-loop primitive (e.g., free
  functions).* Lose the lifecycle event surface that production
  observability needs.

### 2. Sandbox-mediated tool execution — security model

Tools that need shell or filesystem go through a `Sandbox` trait
whose concrete impls live in 1.x companion crates (E2B, Modal,
Fly.io, Lambda, Kubernetes Job, …). Host-direct execution (running
shell commands in the same process as the SDK, optionally inside
Landlock / Seatbelt) is **out of scope** — operators that need that
shape pick a coding-agent harness with built-in host primitives
instead. entelix targets web-service / data / API agent deployment
where every fs / shell op crosses an isolation boundary.

### 3. Invariant 9 reinterpretation — permanent first-party
   abstinence + first-class `Sandbox` delegation

The current Invariant 9 forbids first-party crates from importing
`std::fs`, `std::process`, `tokio::fs`, `tokio::process`,
`landlock`, `seatbelt`, `tree-sitter*`, `nix`. That rule **stays
unchanged** and `scripts/check-no-fs.sh` continues to enforce it
across every workspace crate.

What this ADR makes explicit is what the rule was always implicitly
about: **the rule applies to the SDK's own implementation, not to
its tool surface**. The `Tool` trait has always been sandbox-
agnostic. This direction adds a `Sandbox` trait to `entelix-core`:

```rust
#[async_trait]
pub trait Sandbox: Send + Sync {
    fn backend(&self) -> &str;
    async fn run_command(&self, spec: CommandSpec, ctx: &ExecutionContext)
        -> Result<CommandOutput>;
    async fn run_code(&self, spec: CodeSpec, ctx: &ExecutionContext)
        -> Result<CommandOutput>;
    async fn read_file(&self, path: &str, ctx: &ExecutionContext)
        -> Result<Vec<u8>>;
    async fn write_file(&self, path: &str, bytes: &[u8], ctx: &ExecutionContext)
        -> Result<()>;
    async fn list_dir(&self, path: &str, ctx: &ExecutionContext)
        -> Result<Vec<DirEntry>>;
}
```

Concrete implementations are deferred to 1.x companion crates
(`entelix-sandbox-e2b`, `entelix-sandbox-modal`,
`entelix-sandbox-fly`, …). For test usage and coding-agent
adapters, `entelix-tools::sandboxed` provides
`SandboxedShellTool`, `SandboxedCodeTool`, `SandboxedReadFileTool`,
etc. — these are `Tool` impls that hold an `Arc<dyn Sandbox>` and
forward execution. **None of them imports `std::fs` or
`std::process`**; `check-no-fs.sh` still passes.

The same trait-only-in-core / concrete-in-companion pattern is
already in use for `Embedder` (ADR-0008), `VectorStore`,
`Retriever`, `SearchProvider`. `Sandbox` joins that cohort.

Rejected alternatives:
- *Forbid all sandbox-related types in entelix-core.* Every future
  agent that needs even mocked code execution would be forced to
  invent its own abstraction. Surface fragmentation.
- *Ship a default local-subprocess sandbox impl in entelix-core.*
  Imports `std::process` directly, violates Invariant 9, and
  forces every entelix user to inherit the local-shell threat
  model whether they want it or not.

### 4. `AgentObserver` is a different abstraction layer than `tower::Layer`

An earlier hardening pass standardised on `tower::Layer` for
cross-cutting concerns over single model / tool invocations. That
decision stands.

This direction introduces `AgentObserver`, which is **not** a Hook
revival. It operates at a different abstraction layer:

- **`tower::Layer`** wraps `Service<ModelInvocation>` /
  `Service<ToolInvocation>` — single call boundary, stateless
  cross-cutting (PII redaction, quota, OTel events).
- **`AgentObserver`** wraps the agent's lifecycle — turn
  boundaries (`pre_turn`) and terminal completion (`on_complete`).
  Stateful (e.g. write the assembled conversation to a vector
  store on `on_complete`). Cannot be expressed as `tower::Layer`
  because Layer sees one invocation, not a turn.

The two abstractions stack: a request flows through one or more
Layers (per call) inside one or more Observer windows (per turn).
Naming reflects the difference: `*Layer` vs `*Observer`. Both
suffixes get rows in ADR-0010.

### 5. IR vendor-knob 2+ rule

This direction extends `ModelRequest` with `cache_control` (on
`SystemBlock`) and `response_format` (on the request root). The
risk is uncontrolled IR growth: every codec has vendor-specific
knobs, and putting all of them in IR balloons the surface.

Rule: **a vendor knob enters IR only when 2 or more shipping
codecs natively support it**. Single-vendor knobs stay in
`ProviderOptions` (the codec-specific bag).

| Knob | Native codecs | Decision |
|---|---|---|
| `cache_control` | Anthropic Messages, Bedrock Converse | IR (≥2) |
| `response_format` | OpenAI Chat, OpenAI Responses, Gemini schema | IR (≥3) |
| `thinking_budget` | Anthropic only | ProviderOptions |
| `seed` | OpenAI Chat, Gemini | IR-eligible (deferred to next slice) |

Codecs without native support emit `ModelWarning::LossyEncode`
per ADR-0006 — silent drop is forbidden. `OtelLayer` exports
warnings to `gen_ai.warnings`, surfacing capability mismatches at
observability time.

### 6. `interrupt()` unifies HITL pause/resume and supervised tool gating

The graph crate ships `Command::{Resume, Update, GoTo, ApproveTool}`
and `interrupt(payload)` for human-in-the-loop pause/resume. This
direction adds `ExecutionMode::Supervised` with an `Approver` trait whose
`AwaitExternal` decision must block agent execution until an
external decision arrives.

These are the same problem. Implementation: `Approver::decide`
returning `AwaitExternal` causes the recipe's tool-dispatch path
to call the underlying graph's `interrupt(payload)`. External
approval arrives through the approver's reply channel —
`PendingApproval.reply.send(ApprovalDecision::Approve)` for
`ChannelApprover` — which the agent's loop translates to
`graph.resume_with(Command::Update(approved_state))`.

Single mechanism, two presentation surfaces. No parallel
state machine; no separate `Agent::approve_tool` API — the
approver itself owns the resolution channel.

### 7. LLM-facing vs observability-facing surface separation

Production agents fail in two ways tied to information flow:
- **Token waste**: every byte of metadata that round-trips through
  the LLM is paid in input tokens. Verbose error messages,
  observability IDs, duration counters — none belongs in the next
  turn's prompt.
- **Hidden context loss**: dropping useful signal because the
  observability sink does not see what the agent sent.

Discipline (ADR enforcement; tested in slice integration tests):

| Surface | LLM next turn? | Observability sink? |
|---|---|---|
| `Tool::execute` returned `Value` (success) | yes | yes |
| `Tool::execute` returned `Value` (error: lean prose) | yes | yes |
| Tool wall-clock duration | no | yes |
| Tool argv echo (caller already knows it) | no | yes |
| `ModelResponse.warnings` | no | yes |
| `OtelLayer` span attributes | no | yes |
| MCP raw protocol envelope | no | yes (debug only) |
| Sandbox raw `stderr` (success path) | no | yes |
| Sandbox `exit_code` + `stderr` (failure path) | yes (lean) | yes |

Implementation discipline applied in this slice:

- `SandboxedShellTool` / `SandboxedCodeTool` return `{stdout}` on
  success, `{exit_code, stdout, stderr}` on failure. `duration_ms`
  is observability-only.
- `SandboxedReadFileTool` returns `{content}` only — caller
  already knows the path it asked about.
- `SandboxedWriteFileTool` returns `{ok: true}` — confirmation,
  no echo.
- `SandboxedListDirTool` returns `{entries: [...]}` — no path
  echo.

The duration / id / span metadata flows to the
[`AgentEventSink`] / `OtelLayer` separately, never through the
tool's returned `Value`.

### 8. End-to-end agent-stream coverage

A single integration test stands as the binary proof: mock model +
tool + one `AgentObserver` + `Agent::execute_stream` — the
observed sequence of `AgentEvent` matches what an SSE renderer
downstream consumes.

## Naming taxonomy additions (ADR-0010 mirror)

| Suffix | Semantics | Lifetime | Examples |
|---|---|---|---|
| `*Agent` | runtime entity wrapping `Runnable<S, S>` + event stream | runtime | `Agent<ReActState>` |
| `*Event` | enum of runtime events | static | `AgentEvent<S>` |
| `*Sink` | consumer trait + impls (events go here) | runtime | `AgentEventSink`, `BroadcastSink` |
| `*Mode` | enum of behavior switches | static | `ExecutionMode` |
| `*Observer` | stateful agent-lifecycle observer (turn / complete) | runtime | `AgentObserver` |
| `*Sandbox` | sandbox-agnostic isolated execution environment trait | runtime | `Sandbox`, `MockSandbox` |
| `*Policy` | declarative gate / decision data | static | `ShellPolicy`, `CodePolicy`, `TenantPolicy` |

`Approver` (verb-er trait name) follows the
`Serializer`/`Handler`/`Visitor` ecosystem-standard pattern; no
new suffix row needed.

The pre-existing `*Layer` row in the naming taxonomy is preserved.

## Implementation plan

| Slice | Deliverable | Depends on |
|---|---|---|
| 0 | This ADR | — |
| A | `Agent<S>` runtime + `AgentEvent` (Started / Complete book-ends) + `AgentEventSink` (4 concrete: Dropping/Channel/Broadcast/Capture) | 0 |
| B | `ExecutionMode` + `Approver` trait + `interrupt()`-based supervised gating | A |
| C | `AgentObserver` lifecycle hooks (`pre_turn` / `on_complete`) | A |
| D | IR `CacheControl` + `ResponseFormat` + 5 codec adaptation with LossyEncode warnings | (parallel) |
| E | `Sandbox` trait in core + `SandboxedShellTool`/`SandboxedCodeTool`/`SandboxedReadFileTool`/`SandboxedWriteFileTool`/`SandboxedListDirTool` + `MockSandbox` for tests | (parallel) |
| F | `Auth` helper module + recipes return `Result<Agent<S>>` | A, B, C |
| G | end-to-end agent-stream regression test + 1.0 release-ready declaration | A-F |

Critical path A → B/C → F → G ≈ 3-4 working days; D and E parallelize.

## Done criteria

1. 9-gate suite green across all (now 14-15) workspace crates.
2. Public-api baselines refrozen for every crate touched
   — drift ≠ change-without-ADR.
3. `check-no-fs.sh` 0 violations — Invariant 9's first-party
   abstinence preserved.
4. `check-naming.sh` 0 violations — new suffixes (`*Agent`,
   `*Event`, `*Sink`, `*Mode`, `*Observer`, `*Sandbox`, `*Policy`)
   recognized.
5. ADR-0010, CLAUDE.md §"Naming Taxonomy", `.claude/rules/naming.md`
   all carry the new rows in lockstep.
6. End-to-end agent-stream regression test passes.
7. README + CLAUDE.md surface updated — 1.0 release-ready (live-API verification only).

## References

- ADR-0005 — managed-agent shape (Session/Harness/Hand)
- ADR-0006 — capability honesty + LossyEncode
- ADR-0008 — Embedder trait-only (the trait-in-core / concrete-
  in-companion pattern this ADR generalizes to Sandbox)
- ADR-0010 — naming taxonomy (extended here)
- Anthropic engineering blog — Managed Agents
  (https://www.anthropic.com/engineering/managed-agents)
  consumer driving the ≥85% replaceability bar
  peer SDK with host-direct security model
