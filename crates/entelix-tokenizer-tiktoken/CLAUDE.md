# entelix-tokenizer-tiktoken

Vendor-accurate `entelix_core::TokenCounter` for OpenAI BPE encodings. Wraps `tiktoken-rs`.

## Surface

- **`TiktokenCounter`** — implements `TokenCounter`. Construct via `TiktokenCounter::for_encoding(TiktokenEncoding)` — eager BPE preload, `Arc<CoreBPE>` cached for cheap clone.
- **`TiktokenEncoding`** — closed enum: `Cl100kBase` / `O200kBase` / `P50kBase` / `R50kBase`. `name()` returns the canonical `&'static str` (`"cl100k_base"`, …) that surfaces on `TokenCounter::encoding_name` and the OTel `gen_ai.tokenizer.name` attribute.
- **`TiktokenError::Load { encoding_name, message }`** — thiserror-backed, struct variant aligned with `HfTokenizerError::Load` for cross-companion consistency.

## Crate-local rules

- **Eager preload at construction.** `TokenCounter::count` is sync per the trait contract; `for_encoding` loads the BPE tables synchronously so per-call counting is zero-IO.
- **Encoding-to-model mapping is operator-driven** — `TiktokenEncoding::Cl100kBase` for GPT-3.5/4-turbo + text-embedding-3-*, `O200kBase` for GPT-4o / o1 / o3, etc. The wrapper does not auto-route from model strings (vendors change mappings; auto-routing silently miscounts on stale tables).
- **`encode_ordinary` only** — special-token handling is vendor-and-version-specific; chat-message overhead (3 tokens per message + 3 priming tokens for cl100k/o200k) is operator-supplied via a wrapper counter that overrides `count_messages`.

## Forbidden

- Default-injecting a count on encode failure (silent fallback — invariant 15). The upstream loader signature returns `Result`; surface via `TiktokenError::Load`.
- Adding a model-name → encoding routing table to the wrapper. That ages out and stops matching vendor reality.
