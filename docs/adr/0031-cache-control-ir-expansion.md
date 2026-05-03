# ADR-0031 — Cache control IR expansion

* **Status**: Accepted
* **Date**: 2026-04-28
* **Drivers**: Phase 9D
* **Refines**: ADR-0026 (IR modernization). Extends `CacheControl` from
  system-only to a uniform per-block + per-tool cache surface.

## Context

`CacheControl` shipped in Phase 7 as a `SystemBlock` field — operators
could mark a system block as cached, and codecs that supported it
(Anthropic, Bedrock for Claude) emitted the directive natively while
others emitted `LossyEncode`.

Two production patterns surfaced where this is too narrow:

1. **RAG context caching** — Anthropic and Bedrock support per-block
   cache markers on user/assistant content blocks (long retrieval
   chunks, large image blocks). The IR cannot express this; operators
   cannot cache a 50 KB retrieval block they re-send across turns.
2. **Tool-definition caching** — Anthropic supports `cache_control` on
   `tools[i]` declarations. Stable tool catalogs amortise cleanly; the
   IR has no path.

Two further vendor capabilities the IR does not address:

3. **OpenAI `prompt_cache_key`** — Chat Completions and Responses APIs
   accept a `prompt_cache_key` body field that routes the request to
   the same cache-bucket as a previous one. No per-block annotation
   needed; opaque key.
4. **Gemini `cachedContents` reference** — Gemini's separate
   `cachedContents` API mints a server-side cached-content ID; the
   subsequent `generateContent` call references it via `cachedContent:
   "<id>"` at the request root.

Both (3) and (4) live at the **request** scope, not the block scope —
they are cache *routing keys*, not block-level annotations. They
deserve their own request-root fields.

## Decision

### Per-block `cache_control`

`ContentPart` gains a uniform `cache_control: Option<CacheControl>`
field on every variant that carries content (`Text`, `Image`, `Audio`,
`Video`, `Document`, `Thinking`, `Citation`, `ToolResult`). The
`ToolUse` variant — the assistant's outbound call — does not carry
caching (the model emits it; nothing to cache).

```rust
pub enum ContentPart {
    Text { text: String, cache_control: Option<CacheControl> },
    Image { source: MediaSource, cache_control: Option<CacheControl> },
    Audio { source: MediaSource, cache_control: Option<CacheControl> },
    Video { source: MediaSource, cache_control: Option<CacheControl> },
    Document { source: MediaSource, name: Option<String>, cache_control: Option<CacheControl> },
    Thinking { text: String, signature: Option<String>, cache_control: Option<CacheControl> },
    Citation { snippet: String, source: CitationSource, cache_control: Option<CacheControl> },
    ToolUse { id: String, name: String, input: serde_json::Value },
    ToolResult { tool_use_id: String, content: ToolResultContent, is_error: bool, cache_control: Option<CacheControl> },
}
```

Rationale for the uniform field rather than a wrapper enum:
- A wrapper (`Cached<ContentPart>`) doubles the variant count for
  consumers and complicates pattern matching.
- The field is `Option<CacheControl>` — zero size cost when unset,
  no allocation impact.
- Constructors (`ContentPart::text` etc.) leave it `None` by default.

### `ToolSpec::cache_control`

```rust
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub kind: ToolKind,
    pub cache_control: Option<CacheControl>,
}
```

### Request-root cache routing keys

```rust
pub struct ModelRequest {
    …
    /// OpenAI prompt-cache routing key. When set, the codec emits
    /// `prompt_cache_key: <value>` in the request body so OpenAI's
    /// automatic cache routes consistently across calls. Other
    /// codecs emit a `LossyEncode` warning (the field is not
    /// expressible in their wire format).
    pub cache_key: Option<String>,

    /// Gemini server-side cached-content reference. When set, the
    /// codec emits `cachedContent: <value>` at the request root
    /// (the value is a `cachedContents/<id>` resource name minted
    /// by Gemini's separate `cachedContents` API). Other codecs
    /// emit a `LossyEncode` warning.
    pub cached_content: Option<String>,
}
```

The two fields are kept distinct rather than merged into a single
`cache_routing_key: Option<String>` because the semantics differ:
OpenAI's value is operator-chosen and opaque; Gemini's is a vendor-
minted resource name. A merged field would force codecs to decode
which kind of value they got — clean separation is cheaper.

### Codec coverage matrix (encode)

| IR position | Anthropic | OpenAI Chat | OpenAI Resp. | Gemini | Bedrock |
|---|---|---|---|---|---|
| `SystemBlock.cache_control` | native | LossyEncode | LossyEncode | LossyEncode | native |
| `ContentPart::*.cache_control` | native | LossyEncode | LossyEncode | LossyEncode | native |
| `ToolSpec.cache_control` | native | LossyEncode | LossyEncode | LossyEncode | native |
| `ModelRequest.cache_key` | LossyEncode | native (`prompt_cache_key`) | native (`prompt_cache_key`) | LossyEncode | LossyEncode |
| `ModelRequest.cached_content` | LossyEncode | LossyEncode | LossyEncode | native (`cachedContent`) | LossyEncode |

Three categories — every cell is either native or `LossyEncode`. No
silent loss.

## Consequences

- Every `ContentPart` constructor (`text`, `image`, `audio`, `video`,
  `document`, `thinking`) gains a `cache_control: None` default. The
  IR struct literals across the workspace gain `..Default::default()`
  patterns or explicit `cache_control: None`.
- `ToolSpec::function` keeps the existing convenience constructor and
  defaults `cache_control: None`.
- `ModelRequest::default` defaults the two new fields to `None`.
- 5 codec encode paths read the new positions; the existing
  `system.cache_control` paths are unchanged.
- `lossy_warning_completeness.rs` matrix gains three new rows.
- Public-api baselines refrozen for `entelix-core`, `entelix`.

## Alternatives considered

1. **Wrap content blocks** — `Cached(Box<ContentPart>)` variant.
   Rejected: doubles match-arm count, hurts pattern ergonomics,
   makes `iter_mut()` harder, no functional gain.
2. **Single `cache_routing_key` request field** — rejected (semantic
   conflation, see above).
3. **Generalise `prompt_cache_key` to all vendors** — rejected.
   Vendors with auto-caching (OpenAI) take a routing key; vendors
   with explicit annotations (Anthropic) do not. Forcing one shape
   would mis-model both.
4. **Carry vendor-specific cache options in `ProviderOptions`** —
   rejected for the canonical surface. ADR-0024 §5 (2-codec rule)
   admits per-block caching to IR (Anthropic + Bedrock); auto-cache
   routing admits to IR (OpenAI Chat + OpenAI Responses are the
   2-codec coverage); cached-content references would qualify only
   if a second vendor adopts the same pattern — kept on the IR root
   anyway because the alternative (provider_options) leaks vendor
   shape into recipes.

## References

- ADR-0026 — IR modernization (introduced `MediaSource`, `ToolKind`,
  `Usage` extensions)
- ADR-0024 §5 — vendor-knob 2+ rule
- Anthropic Messages prompt caching docs (per-block `cache_control`)
- OpenAI Chat / Responses `prompt_cache_key`
- Gemini `cachedContents` API + `cachedContent` request field
