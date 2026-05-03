# ADR-0026 — IR modernization for 2025+ model capabilities

* **Status**: Accepted
* **Date**: 2026-04-27
* **Drivers**: Phase 8A
* **Replaces**: §"Provider IR" sections of ADR-0015 (Phase 4 codecs) where IR
  shape is described. Concrete IR types in `entelix_core::ir` are now
  defined as below.

## Context

The IR added in ADR-0015 (Phase 4) modeled the model APIs as they shipped in
mid-2024: text + image input, tool use, structured output, prompt caching on
the system prompt. By 2026 the deployed model fleet has moved beyond that:

- **Reasoning blocks** — Claude 3.7 (`thinking` content), OpenAI o-series
  (`reasoning_content` on response), Gemini (`thoughtSignature` on parts).
  The agent surface needs to carry these *out of band of regular text* so
  recipes can show / suppress / cache them independently of the user-facing
  reply.
- **Multimodal beyond image** — Audio, Video, Document inputs are first-class
  in OpenAI Chat (`input_audio`), Gemini (`inline_data` / `file_data` for any
  MIME), Anthropic (`document` blocks). The previous `ContentPart::Image` was
  the only modality variant; everything else was silently dropped — a direct
  invariant 6 violation.
- **Built-in vendor tools** — Anthropic `web_search_20250305`, Anthropic
  `computer_20250124`, OpenAI `web_search_preview` / `computer_use_preview`,
  Gemini `google_search`. The previous `ToolSpec` modeled only function
  tools, so vendor built-ins escaped the IR and round-tripped only through
  `ProviderOptions` — not portable.
- **Citations / grounding** — every major vendor returns provenance for
  retrieval-augmented or web-search outputs. Previously discarded on decode.
- **Cache + reasoning accounting** — `Usage` distinguished only
  `cache_read_tokens` / `cache_write_tokens` as `Option<u32>`. Reasoning
  tokens were folded into `output_tokens`, hiding cost. Caching is now broad
  enough across vendors that the `Option` distinction (`None` = "vendor
  doesn't report") is no longer useful — every shipping codec reports cache
  numbers; absence is just zero.
- **Safety ratings** — Gemini returns category-level safety scores on every
  response. Previously dropped silently — invariant 6 violation.

The IR cannot grow per-vendor knobs (the 2-codec rule from ADR-0024 §5
applies), but it must cover the *common shape* of these capabilities.
Anything more specific stays in `ProviderOptions`.

## Decision

The IR adopts the following shape. **No backwards-compatibility shims** —
the previous fields are removed in the same change set; downstream code is
updated atomically.

### `ContentPart` variants

```rust
#[non_exhaustive]
pub enum ContentPart {
    Text     { text: String },
    Image    { source: MediaSource },
    Audio    { source: MediaSource },
    Video    { source: MediaSource },
    Document { source: MediaSource, name: Option<String> },
    Thinking { text: String, signature: Option<String> },
    Citation { snippet: String, source: CitationSource },
    ToolUse     { id: String, name: String, input: serde_json::Value },
    ToolResult  { tool_use_id: String, content: ToolResultContent, is_error: bool },
}
```

`#[non_exhaustive]` is added at the enum level so future variants do not
break exhaustive `match` in user code. (Per-variant `#[non_exhaustive]` is
deliberately *not* applied — IR construction is the typical user pattern
and field-level non-exhaustiveness blocks struct-literal construction
outside the crate.)

### `MediaSource` (new)

```rust
pub enum MediaSource {
    Url     { url: String, media_type: Option<String> },
    Base64  { media_type: String, data: String },
    FileId  { id: String, media_type: Option<String> },
}
```

Shared across `Image / Audio / Video / Document`. `media_type` is required
on `Base64` (no other context to infer from), optional on `Url` (URL
extension or HTTP `Content-Type` may suffice) and `FileId` (vendor metadata
typically supplies it).

`FileId` is the new variant — covers OpenAI Files API, Gemini File API,
Anthropic file inputs.

### `CitationSource` (new)

```rust
pub enum CitationSource {
    Url      { url: String, title: Option<String> },
    Document { document_index: u32, title: Option<String> },
}
```

Two-variant lean union covering the *common subset* every vendor returns.
Vendor-specific positioning (start/end byte offsets, chunk indices) does
not enter the IR — codecs that receive offsets in the response emit a
`LossyEncode` warning. The 2-codec rule applies.

### `Thinking` reasoning blocks

Modeled as a `ContentPart` variant alongside `Text`, not a separate `Vec`
on `Message`, so that the *order* of `Text` / `Thinking` blocks (which
Claude relies on for chain-of-thought integrity) round-trips losslessly.

`signature` is optional — Anthropic supplies a redaction signature so that
re-presenting the thinking block on follow-up turns is verifiable; other
vendors leave it `None`.

### `ToolKind` (new) on `ToolSpec`

```rust
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub kind: ToolKind,
}

#[non_exhaustive]
pub enum ToolKind {
    Function   { input_schema: serde_json::Value },
    WebSearch  { max_uses: Option<u32>, allowed_domains: Vec<String> },
    Computer   { display_width: u32, display_height: u32 },
}
```

`input_schema` moves *inside* the `Function` variant — it never made sense
on built-in tools (`WebSearch` and `Computer` have vendor-defined schemas).
The struct stays flat for `name` / `description` because every kind
advertises both.

### `Usage` shape

```rust
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cached_input_tokens: u32,
    pub cache_creation_input_tokens: u32,
    pub reasoning_tokens: u32,
    pub safety_ratings: Vec<SafetyRating>,
}
```

`Option<u32>` collapses to `u32` with default `0` — every shipping codec
populates these, and `0` is the natural "no cache hit" / "no reasoning"
value. Capability advertisement (`Capabilities::prompt_caching`) governs
whether the field is meaningful, not its presence.

`reasoning_tokens` is new — captures o-series `reasoning_tokens`,
Anthropic thinking accounting (where vendors expose it), Gemini thinking
budget consumption.

`safety_ratings` is a `Vec<SafetyRating>` (default empty) — Gemini
populates per request; other codecs leave empty unless they map a refusal.

### `SafetyRating` (new)

```rust
pub struct SafetyRating {
    pub category: SafetyCategory,
    pub level: SafetyLevel,
}

#[non_exhaustive]
pub enum SafetyCategory {
    Harassment, HateSpeech, SexuallyExplicit, DangerousContent,
    Other(String),
}

#[non_exhaustive]
pub enum SafetyLevel {
    Negligible, Low, Medium, High,
}
```

Levels follow the four-bucket convention common to Gemini and equivalent
moderation APIs. `Other(String)` carries vendor categories the IR doesn't
yet model.

### `Capabilities` reshape

Per-modality flags replace the single `vision: bool`:

```rust
pub struct Capabilities {
    pub streaming: bool,
    pub tools: bool,

    pub multimodal_image:    bool,
    pub multimodal_audio:    bool,
    pub multimodal_video:    bool,
    pub multimodal_document: bool,

    pub system_prompt:     bool,
    pub structured_output: bool,
    pub prompt_caching:    bool,

    pub thinking:     bool,
    pub citations:    bool,
    pub web_search:   bool,
    pub computer_use: bool,

    pub max_context_tokens: u32,
}
```

Every capability that a `ContentPart` / `ToolKind` / `Usage` field
represents has a corresponding `bool` so codec preflight can warn before
the wire encode.

### `ModelWarning::LossyEncode` semantics

Stays the universal "codec dropped or coerced an IR field" marker — the
direction (encode vs decode) is inferable from the `field` path. A second
variant for decode-side losses was considered and rejected — doubles the
test surface without strong gain.

A `lossy_warning_completeness.rs` integration test asserts: for every IR
variant × every codec, the codec either produces a wire shape that
preserves the variant (native), or emits exactly one `LossyEncode`
warning naming the variant's IR path. Silent loss fails the test.

## Codec coverage matrix

| IR feature              | Anthropic | OpenAI Chat | OpenAI Resp. | Gemini   | Bedrock   |
|-------------------------|-----------|-------------|--------------|----------|-----------|
| `Thinking`              | native    | LossyEncode (request); native (response decode for o-series) | same | native | native (Anthropic-on-Bedrock) |
| `Image`                 | native    | native      | native       | native   | native    |
| `Audio`                 | LossyEncode | native    | native       | native   | LossyEncode |
| `Video`                 | LossyEncode | LossyEncode | LossyEncode | native   | LossyEncode |
| `Document`              | native    | native (FileId) | native (FileId) | native | native    |
| `Citation`              | native    | native      | native       | native   | native (Anthropic-on-Bedrock) |
| `ToolKind::WebSearch`   | native    | LossyEncode | native       | native   | native (Anthropic-on-Bedrock) |
| `ToolKind::Computer`    | native    | LossyEncode | LossyEncode  | LossyEncode | native (Anthropic-on-Bedrock) |
| `cached_input_tokens`   | native    | native      | native       | native   | native    |
| `cache_creation_input_tokens` | native | LossyEncode (no field) | LossyEncode | LossyEncode | native |
| `reasoning_tokens`      | LossyEncode (folded into output) | native | native | native | LossyEncode |
| `safety_ratings`        | LossyEncode (refusal mapping only) | LossyEncode | LossyEncode | native | LossyEncode |

## Consequences

- **Breaks**: every consumer of `ContentPart::Image { source: ImageSource }`
  must migrate to `MediaSource`. `ToolSpec { input_schema }` callers must
  wrap into `ToolKind::Function { input_schema }`. `Usage::cache_*_tokens`
  callers drop the `Option`. `Capabilities { vision }` callers move to
  `multimodal_image`.
- **Public-api baseline drift**: every crate exposing IR types refreezes
  (`entelix-core`, `entelix`, plus derivative crates).
- **Test surface**: `lossy_warning_completeness.rs` (new) becomes the
  forcing function for invariant 6.

## Alternatives considered

1. **Add `Multimodal` variant carrying any modality** — rejected. The
   model's `match` logic differs per modality (encoders, token counters),
   so a single variant would force everyone to write a sub-match — a
   pyrrhic compaction.
2. **Vendor-shaped `RawContent(Value)` escape hatch** — rejected. Violates
   invariant 5 (provider IR before wire format); reduces portability to
   zero.
3. **Per-vendor `VendorOptions` on `ToolSpec`** — rejected for built-in
   tools. The 2-codec rule for `WebSearch` and `Computer` already crosses
   the threshold (Anthropic + OpenAI Responses for WebSearch; Anthropic +
   Bedrock for Computer).

## References

- ADR-0015 §"Provider IR" (replaced)
- ADR-0024 §5 (vendor knob 2+ rule)
- Invariant 5 (provider IR before wire) and Invariant 6 (lossy-encode warnings)
