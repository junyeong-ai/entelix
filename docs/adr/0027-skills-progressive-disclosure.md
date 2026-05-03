# ADR-0027 — Skills with progressive disclosure

* **Status**: Accepted
* **Date**: 2026-04-27
* **Drivers**: Phase 8B
* **Supersedes**: nothing — Skills are a new abstraction layer

## Context

`Tool` (single-call invocation) and `create_*_agent` recipes (one-shot
factory functions) cover the bare-metal end of the agentic surface.
Industry-leading agent SDKs (Anthropic Claude Skills, Claude Code's
filesystem skills, the OpenAI Agents SDK's parameterised assistants)
expose a third unit between them — a **packaged, composable, lifecycle-
aware capability** the agent can opt into per turn.

Two forces drive the abstraction:

1. **Token discipline**. As skill catalogues grow, putting every skill's
   full instructions into every system prompt scales the request body
   linearly with skill count. Anthropic's progressive-disclosure model
   (description loaded by default; body loaded only when the skill is
   activated; resources fetched on demand) keeps the per-turn token cost
   proportional to *what the model actually needs*, not what the operator
   has registered.
2. **Backend-agnosticism**. Skill content can live in many places: an
   in-memory struct embedded by the SDK consumer, a sandbox-internal
   file tree (Anthropic Skills shape — `SKILL.md` + `reference/` +
   `examples/`), an MCP server, an HTTP store. The lifecycle abstraction
   must let any of these back the same `Skill` instance without the
   agent runtime caring.

The constraints are also two:

- **Invariant 9** (no fs / shell in core libs). `Skill` trait and the
  registry stay backend-agnostic; concrete fs-backed implementations
  reach the filesystem only via `Sandbox`.
- **Invariant 11** (multi-tenant `Namespace` mandatory). Every async
  call carries `ExecutionContext`, propagating `tenant_id` through
  whatever backend the skill talks to.

## Decision

Introduce a `Skill` trait + `SkillRegistry` (in `entelix-core::skills`),
three LLM-facing built-in tools (in `entelix-tools::skills`), and two
concrete `Skill` implementations (`InMemorySkill`, `SandboxSkill`).
Sub-agents extend with a `with_skill_filter` mirroring the existing
tool-filter pattern (F7 parity).

### `Skill` trait

```rust
#[async_trait]
pub trait Skill: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &str;                 // T1 — always loaded
    fn description(&self) -> &str;          // T1 — always loaded
    fn version(&self) -> Option<&str> { None }
    async fn load(&self, ctx: &ExecutionContext)
        -> Result<LoadedSkill>;             // T2 — on activation
}
```

**No `triggers()` method.** Keyword/pattern auto-activation is an
"어설픈 휴리스틱" risk — false positives spam unrelated turns with
skill content; false negatives swallow correctly-shaped queries. The
right discipline is *model-driven activation* (the model reads
descriptions and chooses), with optional rule-based wrappers a layer
above the core trait for users who want them. The trait stays minimal.

### `LoadedSkill` and `SkillResource`

```rust
pub struct LoadedSkill {
    pub instructions: String,
    pub resources: BTreeMap<String, Arc<dyn SkillResource>>,
}

#[async_trait]
pub trait SkillResource: Send + Sync + std::fmt::Debug {
    async fn read(&self, ctx: &ExecutionContext) -> Result<SkillResourceContent>;
}

pub enum SkillResourceContent {
    Text(String),
    Binary { mime_type: String, bytes: Vec<u8> },
}
```

Resources are **lazily** read; the activated skill body lists only the
keys, not the contents. The model reads a resource by issuing
`read_skill_resource(skill, key)` — the third tier in the disclosure
ladder.

### `SkillRegistry`

```rust
pub struct SkillRegistry { ... }

impl SkillRegistry {
    pub fn new() -> Self;
    pub fn register(self, skill: Arc<dyn Skill>) -> Result<Self>;
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Skill>>;
    pub fn summaries(&self) -> Vec<SkillSummary<'_>>;
    pub fn filter(&self, allowed: &[&str]) -> Self;     // F7 parity
}
```

Append-only at init time, cloneable for sub-agent scope, exact-name
lookup. `summaries()` is the T1 view consumed by `ListSkillsTool`.

### Three built-in tools

The agent reaches the registry through three small, well-named `Tool`
impls. They live in `entelix-tools::skills` so the LLM-facing wire shape
(input schema + output JSON) is auditable, but their authority is
strictly delegated to the registry.

| Tool | Input | Output | Tier transition |
|---|---|---|---|
| `ListSkillsTool` | `{}` | `[{name, description, version}, ...]` | T1 → T1 (metadata only) |
| `ActivateSkillTool` | `{name}` | `{instructions, resources: [keys]}` | T1 → T2 (load body) |
| `ReadSkillResourceTool` | `{skill, key}` | text → `{text}`, binary → `{mime_type, size_bytes, sha256}` | T2 → T3 (load resource) |

**Binary resources are not embedded** in the LLM-facing tool result —
only metadata. The bytes are accessible via the `SkillRegistry` API for
out-of-band use (e.g., the agent's host application uploading the
resource to a vendor file API and then handing the resulting `FileId`
back to the model). Embedding base64 of a 5MB PDF directly in the
conversation would silently consume tens of thousands of tokens —
exactly the failure mode progressive disclosure exists to prevent.

### Concrete impls shipped in 1.0

- `InMemorySkill` — struct holding name, description, version,
  instructions, and a resource map. Construction via
  `InMemorySkillBuilder` for ergonomic chaining.
- `SandboxSkill` — backed by a sandbox-internal directory tree mirroring
  the Anthropic Claude Skills layout (`SKILL.md` for instructions,
  any other relative files become resources). All filesystem access
  flows through `Arc<dyn Sandbox>` — invariant 9 preserved.

`McpSkill` is **not shipped** in 1.0. MCP and Skills are at different
layers (MCP = wire protocol; Skill = packaging/lifecycle); pinning a
default mapping would force a particular usage shape on operators. Any
user can implement `Skill` directly against `McpManager` — the trait is
public.

### No automatic system-prompt injection

The recipe / agent runtime does **not** automatically render skill
summaries into the system prompt. Two reasons:

1. The system prompt is the operator's contract surface — runtime
   injection is invisible to the operator and bypasses their control of
   the prompt's exact bytes.
2. Even at T1, summaries cost tokens. Operators who want them inline
   can call `SkillRegistry::summaries()` and concatenate them into the
   system prompt themselves; operators who prefer model-driven discovery
   simply register `ListSkillsTool` and let the model invoke it on
   demand.

### Sub-agent filtering (F7 parity)

```rust
impl<S> Subagent<S> {
    pub fn from_whitelist(parent: &Agent<S>, tools: &[&str], skills: &[&str]) -> Self;
}
```

A sub-agent receives the parent's `SkillRegistry::filter(skills)` —
explicitly-named subset, no inheritance of unnamed skills. F7's "권한
widening 회피" applies: there is no default that hands the parent's
full skill set to a sub-agent.

### Sandbox-native usage (no Agent required)

The `Skill` trait carries no dependency on `Agent`. A consumer using
`entelix` purely as a sandbox tool dispatcher:

```rust
let registry = SkillRegistry::new()
    .register(Arc::new(SandboxSkill::new(sandbox.clone(), "/skills/code-review")))?
    .register(Arc::new(SandboxSkill::new(sandbox.clone(), "/skills/sql-expert")))?;

let loaded = registry.get("code-review").unwrap().load(&ctx).await?;
println!("{}", loaded.instructions);
```

`ListSkillsTool` / `ActivateSkillTool` / `ReadSkillResourceTool` are
ordinary `Tool` impls — they work in any dispatcher (LLM-driven or
deterministic-workflow), so the SDK is first-class for both agentic
and sandbox-native callers.

## Consequences

- New crate-level module: `entelix_core::skills`. Re-exported from the
  `entelix` facade alongside existing primitives.
- `entelix-tools` gains `skills::{ListSkillsTool, ActivateSkillTool,
  ReadSkillResourceTool, InMemorySkill, InMemorySkillBuilder, SandboxSkill}`.
- `Subagent` API gains a `skills` parameter on whitelist constructors.
- Public-api baselines refrozen for `entelix-core`, `entelix-tools`,
  `entelix-agents`, `entelix`.
- New `.claude/rules/skill.md` quick-reference.
- Tests:
  - `skill_registry_*.rs` — append-only, exact lookup, filter behaviour.
  - `skill_in_memory_*.rs` — round-trip activation, resource read.
  - `skill_sandbox_*.rs` — `MockSandbox`-backed read paths.
  - `skill_tools_*.rs` — three LLM-facing tools, including binary
    placeholder behaviour.
  - `skill_progressive_disclosure_token_budget.rs` — asserts T1 / T2 /
    T3 boundaries (description-only listing, instructions-on-activate,
    resource-on-read).
  - `skill_subagent_filter.rs` — F7 enforcement.

## Alternatives considered

1. **Trigger-based auto-activation as a first-class trait method** —
   rejected. Heuristic routing is exactly the failure mode the user
   explicitly flagged ("어설픈 휴리스틱 회피"). Keep it out of core;
   ship as an optional layer above.
2. **Eagerly inject all skill instructions into the system prompt** —
   rejected. Linear-in-skill-count token cost defeats the entire
   progressive-disclosure premise.
3. **Embed binary resource bytes (base64) in the
   `read_skill_resource` tool result** — rejected. A 5 MB PDF becomes
   ~7 M base64 characters → ~1.7 M tokens. The metadata-only return
   shape preserves the LLM's cost model.
4. **Bundle a default `McpSkill` adapter in `entelix-tools`** —
   rejected for 1.0. Anthropic's own SDK keeps Skills and MCP separate;
   pinning a particular mapping is premature. The trait is public —
   any consumer can implement.
5. **Auto-wire the three built-in tools via an explicit helper** —
   adopted (Phase 9C). The opt-in surface is
   [`entelix_tools::skills::install(tool_registry, skill_registry)
   -> Result<ToolRegistry>`]: it returns a fresh `ToolRegistry`
   with `list_skills` / `activate_skill` / `read_skill_resource`
   appended to the caller's existing tools. Sub-agents call this
   automatically inside [`Subagent::into_react_agent`] when
   [`Subagent::with_skills`] has been used to scope the parent's
   skill subset. This satisfies the "drop a `SKILL.md` and the
   model sees it" promise without the silent-magic concern that
   motivated the original rejection: the wiring trigger is one
   explicit function call, not implicit inference from
   `sink(...)` or similar. Operators that want skills as data
   only (no LLM-facing tools) skip the `install` call.

## References

- ADR-0024 — Agent SDK direction (managed-agents shape)
- Invariant 9 (no fs / shell in core libs)
- Invariant 11 (multi-tenant `Namespace` mandatory)
- F7 (sub-agent permission widening avoidance)
