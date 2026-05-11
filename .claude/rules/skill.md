---
paths:
  - "crates/entelix-core/src/skills/**"
  - "crates/entelix-tools/src/skills/**"
---

# Skills with progressive disclosure

## 3-tier disclosure model

| Tier | Loaded when | Content | Token cost |
|---|---|---|---|
| T1 | Available every turn | `name + description (+ version)` | tiny (one line) |
| T2 | `activate_skill` invoked | full `instructions` + resource key list | medium (~hundreds–thousands) |
| T3 | `read_skill_resource` invoked | one resource body (text) or metadata only (binary) | variable |

Binary resource bodies are **never** placed in LLM-facing responses. Image/PDF base64 in context defeats progressive disclosure — the host application handles binary delivery on a separate channel.

## Trait

```rust
#[async_trait]
pub trait Skill: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &str;                 // T1
    fn description(&self) -> &str;          // T1
    fn version(&self) -> Option<&str> { None }
    async fn load(&self, ctx: &ExecutionContext)
        -> Result<LoadedSkill>;             // T2
}
```

No `triggers()` method. Keyword/pattern auto-activation is brittle (false positives + false negatives both bad). Model-driven activation is the only first-class path. Wrap in a separate layer if auto-activation is desired.

## Registry

- `SkillRegistry::register(skill)` — duplicate name → `Error::Config`
- `SkillRegistry::get(name)` — exact lookup
- `SkillRegistry::summaries()` — stable-sorted by name
- `SkillRegistry::filter(allowed)` — narrowing only, never widening

## Sub-agents

```rust
let sub = Subagent::builder(model, &parent_tools, "research_assistant", "...")
    .restrict_to(&["read_file"])
    .with_skills(&parent_skills, &["code-review", "sql-expert"])?
    .build()?;
```

Default skill set is empty — `with_skills` must be called explicitly to inherit a parent subset.

## Three LLM-facing tools

```
list_skills           → [{name, description, version}, ...]                  (T1 → T1 metadata)
activate_skill(name)  → {instructions, resources: [keys]}                    (T1 → T2 body)
read_skill_resource(skill, key) → {text} | {mime_type, size_bytes, sha256}   (T2 → T3)
```

### Auto-wire helper — `entelix_tools::skills::install`

Auto-wire is **opt-in**. The default is still explicit `ToolRegistry::register`. Recipes/operators that want to skip the boilerplate use:

```rust
use entelix_tools::skills::install;
let tools = install(tool_registry, skill_registry)?;
```

The three LLM-facing tools are appended to a clone of `tool_registry`, which is returned. Without this call there is no wiring — operators retain explicit control over which tools enter the registry. `Subagent::with_skills` calls this helper inside `into_react_agent`.

## No automatic system-prompt injection

`SkillRegistry::summaries()` is **not** auto-concatenated into the system prompt:

1. The system prompt is operator contract — runtime injection is invisible.
2. Summaries cost tokens. Model-driven discovery via `list_skills` is cleaner.

Operators that want inline summaries call `summaries()` and concat themselves.

## Concrete impls

- `InMemorySkill` + `InMemorySkillBuilder` — embedded/test
- `SandboxSkill` — Anthropic Skills layout (`SKILL.md` + sub-files); takes `Arc<dyn Sandbox>` to preserve invariant 9
- `StaticResource` (text/binary), `SandboxResource` (lazy)

`McpSkill` is intentionally not shipped — MCP and Skill are orthogonal abstractions. Consumers implement `Skill` directly when they need an MCP-backed skill.
