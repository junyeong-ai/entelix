# ADR 0078 — `ReasoningEffort` cross-vendor IR + 5 codec wire

**Status**: Accepted
**Date**: 2026-05-05
**Decision**: `ReasoningEffort` is a 7-variant cross-vendor IR knob (`Off / Minimal / Low / Medium / High / Auto / VendorSpecific(String)`) on `ModelRequest::reasoning_effort`. Five codecs translate onto their native wire shapes through a documented mapping table; lossy approximations emit `ModelWarning::LossyEncode` per invariant 6. Vendor-specific reasoning surfaces (`AnthropicExt::thinking`, `OpenAiResponsesExt::reasoning`'s effort + summary) are removed in the same commit (invariant 14 — no shim). OpenAI Responses' summary verbosity stays vendor-specific on `OpenAiResponsesExt::reasoning_summary`.

## Context

Q2 2026 reasoning-capable model families expose the "how hard should the model think" knob in incompatible wire shapes:

- **Anthropic Messages** — `thinking: { type: "enabled" | "adaptive" | "disabled", budget_tokens: N }`. Token budget. Opus 4.7 is **adaptive-only** (manual `budget_tokens` rejected with vendor 4xx).
- **OpenAI Responses** — `reasoning: { effort: "none" | "minimal" | "low" | "medium" | "high" | "xhigh" }`. Discrete bucket; no token budget.
- **OpenAI Chat Completions** — no reasoning knob; the surface lives on Responses only.
- **Gemini 2.5** — `thinkingConfig: { thinkingBudget: N | -1 | 0 }`. Token budget; `-1` is auto, `0` disables (Flash only — Pro cannot disable).
- **Gemini 3** — `thinkingConfig: { thinkingLevel: "minimal" | "low" | "medium" | "high" }`. Discrete bucket.
- **Bedrock Converse** — Anthropic family routes the Anthropic `thinking` shape through `additionalModelRequestFields` passthrough; non-Anthropic Bedrock models (Nova, Mistral, Llama) have no thinking surface.

Up to and including 1.0-RC.1 entelix exposed two narrow vendor-specific surfaces:

- `ProviderExtensions::anthropic.thinking: Option<ThinkingConfig { budget_tokens: u32 }>` — Anthropic-only, raw budget.
- `ProviderExtensions::openai_responses.reasoning: Option<ReasoningConfig { effort: ReasoningEffort, summary: Option<ReasoningSummary> }>` — OpenAI-only, 4-variant `ReasoningEffort`.
- Gemini and OpenAI Chat had no reasoning surface (Gemini missed thinking entirely; OpenAI Chat never had one).

That asymmetry forced operators routing the same logical request across model tiers ("light reasoning for the routing turn, deep reasoning for the planning turn") to manually branch on which vendor each model belonged to. It also missed Anthropic Opus 4.7's adaptive-only constraint — manual `budget_tokens` with that model produced a vendor 4xx that the codec did not catch at encode time.

Fork 4 of the 6-fork audit (`project_entelix_2026_05_05_master_plan.md`) confirmed the cross-vendor mapping is **lossy by vendor design, not by SDK design** — the discrete buckets do not align across vendors, the budget integers map to fundamentally different scales, and Anthropic's adaptive mode has no OpenAI / Gemini-3 analogue. The right SDK shape exposes one canonical enum, snaps each variant to the nearest vendor bucket, and surfaces the snap as `LossyEncode` so operators see exactly which mappings degraded.

The 6 industry SDKs surveyed (pydantic-ai 1.90, LangGraph 1.0 GA, OpenAI Agents SDK, Claude Agent SDK, rig, Vercel AI SDK 5) share LiteLLM's `reasoning_effort: "minimal"|"low"|"medium"|"high"` taxonomy as the cross-vendor *enum* axis, with no consensus on `Off` / `Auto` / `xhigh` handling and no consensus on token budgets. entelix's 7-variant enum + escape-hatch + emit-on-snap design extends the industry baseline to cover the missing variants without forcing operators to drop into vendor-specific codepaths for the corner cases.

## Decision

### `ReasoningEffort` enum

`entelix-core/src/ir/reasoning.rs`:

```rust
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ReasoningEffort {
    Off,
    Minimal,
    Low,
    #[default]
    Medium,
    High,
    Auto,
    VendorSpecific(String),
}
```

Six discrete cross-vendor buckets ordered `Off < Minimal < Low < Medium < High` (Auto sits orthogonal — vendor decides). The seventh variant `VendorSpecific(String)` is the escape-hatch — operators reach for it when they need exact `budget_tokens: 9000` on Anthropic, exact `thinkingBudget: 16384` on Gemini, or `effort: "xhigh"` on OpenAI. Each codec parses the literal per its own contract; codecs that cannot interpret the literal emit `LossyEncode` and fall through to `Medium`.

The default is `Medium` — matches every vendor's "reasonable balance" tier. `Capabilities::thinking: bool` continues to advertise whether a codec/model accepts the knob at all; `ReasoningEffort` documents the *level* once support exists.

### `ModelRequest::reasoning_effort: Option<ReasoningEffort>`

The IR field is `Option` — `None` means "vendor default" (codec emits no thinking / reasoning field). Operators that explicitly want the model not to think set `Some(ReasoningEffort::Off)`.

`ChatModel::with_reasoning_effort(effort)` is the builder; `ChatModelConfig::reasoning_effort()` is the accessor.

### Per-codec mapping (encode-time)

| `ReasoningEffort` | Anthropic                                    | OpenAI Responses        | Gemini 2.5 (Flash)        | Gemini 2.5 (Pro)          | Gemini 3                  |
|-------------------|----------------------------------------------|-------------------------|---------------------------|---------------------------|---------------------------|
| `Off`             | `{type:"disabled"}`                          | `effort:"none"`         | `thinkingBudget:0`        | LossyEncode → `512`       | LossyEncode → `"minimal"` |
| `Minimal`         | LossyEncode → `{type:"adaptive", effort:"low"}` | `effort:"minimal"`      | `thinkingBudget:512`      | `thinkingBudget:512`      | `thinkingLevel:"minimal"` |
| `Low`             | `{type:"enabled", budget_tokens:1024}` ¹     | `effort:"low"`          | `thinkingBudget:1024`     | `thinkingBudget:1024`     | `thinkingLevel:"low"`     |
| `Medium`          | `{type:"enabled", budget_tokens:4096}` ¹     | `effort:"medium"`       | `thinkingBudget:8192`     | `thinkingBudget:8192`     | `thinkingLevel:"medium"`  |
| `High`            | `{type:"enabled", budget_tokens:16384}` ¹    | `effort:"high"`         | `thinkingBudget:24576`    | `thinkingBudget:24576`    | `thinkingLevel:"high"`    |
| `Auto`            | `{type:"adaptive"}`                          | LossyEncode → `medium`  | `thinkingBudget:-1`       | `thinkingBudget:-1`       | LossyEncode → `"high"`    |
| `VendorSpecific`  | parse as numeric `budget_tokens` ²           | passthrough literal     | parse as numeric ³        | parse as numeric ³        | passthrough literal       |

¹ Anthropic Opus 4.7 emits `{type:"adaptive", effort:"low|medium|high"}` for `Low/Medium/High` — manual budget is rejected.

² Anthropic VendorSpecific: `s.parse::<u32>()` produces `budget_tokens`; non-numeric `s` emits LossyEncode and falls through to `Medium`. Opus 4.7 rejects any VendorSpecific with LossyEncode → `{type:"adaptive"}`.

³ Gemini 2.5 VendorSpecific: `s.parse::<i32>()` produces `thinkingBudget`; non-numeric falls through to `Medium`.

OpenAI Chat Completions has no reasoning knob — it emits LossyEncode `"OpenAI Chat Completions has no reasoning / thinking knob — drop the field; use OpenAiResponsesCodec for o-series reasoning models"` whenever `reasoning_effort` is set.

Bedrock Converse routes Anthropic-family models (`anthropic.claude-…` ID prefix, including cross-region inference prefixes like `us.anthropic.claude-…`) through `additionalModelRequestFields.thinking` with the same Anthropic mapping. Non-Anthropic Bedrock models emit LossyEncode and drop the field.

### Removed surfaces (invariant 14, no shim)

- `ir::AnthropicExt::thinking: Option<ThinkingConfig>` — gone. The Anthropic codec sources thinking from `ModelRequest::reasoning_effort` instead. `ThinkingConfig` struct removed.
- `ir::OpenAiResponsesExt::reasoning: Option<ReasoningConfig>` — gone. The OpenAI Responses codec sources effort from `ModelRequest::reasoning_effort`.
- `ReasoningConfig` struct (effort + summary pair) — gone. Summary verbosity is OpenAI-Responses-specific (Anthropic / Gemini / Bedrock have no equivalent), so it stays as `OpenAiResponsesExt::reasoning_summary: Option<ReasoningSummary>`.
- The previous 4-variant `ReasoningEffort` enum (`Minimal / Low / Medium / High`) at `provider_extensions::ReasoningEffort` — replaced by the 7-variant cross-vendor enum at `ir::reasoning::ReasoningEffort`. Re-exported via `entelix_core::ir::ReasoningEffort` so call sites import from the same path they always have.

### Why `VendorSpecific(String)` not `BudgetTokens(u32)` + `Effort(&'static str)`

The escape-hatch could have split into two variants — `BudgetTokens(u32)` for budget-based vendors and `EffortLiteral(&'static str)` for enum-based vendors. The single-string form is cleaner because:

- Anthropic and Gemini 2.5 both take a *number*; OpenAI and Gemini 3 both take a *literal*. A two-variant escape splits the operator's intent ("I want Anthropic budget 9000 OR Gemini 2.5 budget 9000 OR OpenAI xhigh") across both forms — operators routing across vendors would have to maintain two parallel call sites.
- `String` lets each codec parse the literal per its own contract. Numeric parsing in Anthropic / Gemini 2.5; literal pass-through on OpenAI / Gemini 3. The Rust type system surfaces the constraint via per-codec encode behaviour, not via a top-level enum split that the operator would have to remember.
- The `&'static str` alternative would force operators to hard-code `"9000"` literals at compile time, blocking runtime-driven budget selection (e.g. config-file-sourced budget tuning). `String` keeps the path open.

### Why per-codec helpers, not a generic dispatcher

Each codec carries its own `encode_*_thinking` private helper. The alternative — a generic `dispatch_reasoning_effort(vendor, model, effort) -> Value` — was rejected:

- The shape of the produced `Value` differs per vendor (root-level `thinking` field for Anthropic, root-level `reasoning` for OpenAI Responses, nested `generationConfig.thinkingConfig` for Gemini, nested `additionalModelRequestFields.thinking` for Bedrock). The dispatcher would need a per-vendor "where does this go" mapping anyway.
- LossyEncode emit is per-codec — the helper's caller needs to know which vendor's wire shape failed to round-trip. A central dispatcher hides which codec emitted the warning.
- Per-vendor model-family detection (Opus 4.7 adaptive-only, Gemini 2.5 Flash vs Pro, Gemini 3 vs 2.5, Bedrock Anthropic vs Nova) is codec-internal — it does not generalise.

### Capability bit consistency

`Capabilities::thinking: bool` flags whether the codec/model pair accepts the knob. The bit is independent of `ReasoningEffort` — it lets agent recipes branch on "this model can think" before committing to a level. Codecs with `thinking: false` (none today, but future codecs that ship without reasoning support) emit LossyEncode on every variant.

## Consequences

**Positive**:

- One IR field captures the full cross-vendor knob — operators write `with_reasoning_effort(ReasoningEffort::Medium)` and the SDK produces the right wire shape per codec.
- The `Off` and `Auto` variants close the previous gap — entelix matches LiteLLM's `none` / vendor-decides handling, plus exposes `Auto` cleanly on Anthropic 4.6+ adaptive and Gemini 2.5 `-1`.
- `VendorSpecific(String)` escape-hatch covers `xhigh` (OpenAI), exact `budget_tokens: 9000` (Anthropic non-Opus-4.7), exact `thinkingBudget: 6000` (Gemini 2.5) — operators with vendor-specific tuning needs do not lose access to the raw wire value.
- Anthropic Opus 4.7's adaptive-only constraint is enforced at encode time — manual budget rejected before the request hits the vendor 4xx path.
- Lossy approximations are typed `ModelWarning::LossyEncode` events the operator's `entelix-otel` layer surfaces — silent loss is impossible (invariant 6).

**Negative**:

- `ReasoningEffort` carries a `String` payload on `VendorSpecific` (cannot be `Copy`). The `Clone` cost is one `Arc`-or-bytes copy per request — negligible against the model call's network latency.
- Per-codec helpers add ~80 lines per codec for the dispatch table. The verbosity pays for itself in per-variant LossyEncode precision (operators see "Gemini Pro cannot disable thinking — snapped to 512", not a generic "lossy reasoning effort").
- One additional public type (`ReasoningEffort`) re-located from `provider_extensions::` to `ir::` and expanded from 4 to 7 variants; one removed type (`ThinkingConfig`); one removed type (`ReasoningConfig`); one removed field (`AnthropicExt::thinking`); one removed/renamed field (`OpenAiResponsesExt::reasoning` → `OpenAiResponsesExt::reasoning_summary`). Per ADR-0064 §"Public-API baseline contract" the typed-strengthening drift is authorised at the 1.0 RC contract boundary.

**Migration outcome (one-shot, no shim)**: All vendor-specific surfaces (`AnthropicExt::thinking`, `OpenAiResponsesExt::reasoning`) and their supporting types (`ThinkingConfig`, `ReasoningConfig`) are deleted, not deprecated. Operators that previously wrote `AnthropicExt::default().with_thinking_budget(2048)` rewrite to `ChatModel::with_reasoning_effort(ReasoningEffort::VendorSpecific("2048".into()))` (or `ReasoningEffort::Low` for the canonical bucket). The OpenAI Responses summary verbosity continues to live on `OpenAiResponsesExt::reasoning_summary` — operators that paired `reasoning.effort = Medium` with `reasoning.summary = Concise` now write `with_reasoning_effort(ReasoningEffort::Medium)` plus `OpenAiResponsesExt::default().with_reasoning_summary(ReasoningSummary::Concise)`. There is no `#[deprecated]`, no `pub use OldName as NewName` — invariant 14 forbids the shim.

## References

- CLAUDE.md invariant 6 — lossy encoding emits warnings (the LossyEncode emit sites in this ADR's per-codec helpers)
- CLAUDE.md invariant 14 — no backwards-compatibility shims
- ADR-0010 — naming taxonomy (`ReasoningEffort` is `Effort` not `Level` per the established `*Effort` precedent on the previous 4-variant enum; the 7-variant expansion preserves the existing public name)
- ADR-0032 — silent-fallback bug class (lossy approximations route through `LossyEncode`, not `unwrap_or_else`)
- ADR-0064 — 1.0 release charter (post-RC typed-strengthening authorised)
- Master plan May-05 (`project_entelix_2026_05_05_master_plan.md`) §"B-3 — ReasoningEffort 6-variant" — Fork 4's vendor wire-format research expanded the 6-variant proposal to 7 with the `VendorSpecific(String)` escape
- `crates/entelix-core/src/ir/reasoning.rs` — enum definition, doc table, round-trip serde tests
- `crates/entelix-core/src/codecs/anthropic.rs` `encode_anthropic_thinking` — Anthropic Messages API mapping including Opus 4.7 adaptive-only enforcement
- `crates/entelix-core/src/codecs/openai_responses.rs` `encode_openai_responses_reasoning` — OpenAI Responses `reasoning` mapping with `Auto`-snap-to-medium LossyEncode
- `crates/entelix-core/src/codecs/openai_chat.rs` — LossyEncode emit (Chat Completions has no reasoning knob)
- `crates/entelix-core/src/codecs/gemini.rs` `encode_gemini_thinking` — Gemini 2.5 (`thinkingBudget`) and Gemini 3 (`thinkingLevel`) split, Flash vs Pro `Off` differential
- `crates/entelix-core/src/codecs/bedrock_converse.rs` `encode_bedrock_thinking` — Anthropic-family routing through `additionalModelRequestFields`, non-Anthropic LossyEncode
