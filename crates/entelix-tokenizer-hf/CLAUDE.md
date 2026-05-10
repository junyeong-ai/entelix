# entelix-tokenizer-hf

Vendor-accurate `entelix_core::TokenCounter` wrapping the HuggingFace `tokenizers` crate. Llama / Qwen / Mistral / DeepSeek / Gemma / Phi or any tokenizer.json source.

## Surface

- **`HfTokenCounter`** — implements `TokenCounter`. Construct via `HfTokenCounter::from_bytes(bytes, encoding_name)` — bytes-only (no `from_file` / `from_pretrained`). `Arc<Tokenizer>` cached for cheap clone.
- **`HfTokenizerError::Load { encoding_name, message }`** — variant shape mirrors `TiktokenError::Load` for cross-companion consistency.

## Crate-local rules

- **Bytes-only construction.** No `from_file` / `from_pretrained` constructors. Two reasons:
  1. **Invariant 9** — entelix first-party crates do not import `std::fs`. Operators read tokenizer files in their own application code (or `include_bytes!` at compile time) and pass the byte payload in.
  2. **No silent network IO** — `tokenizers::Tokenizer::from_pretrained` does HTTP downloads + disk caching as a side effect. Operators wanting hub integration wire it explicitly in their boot path.
- **`Box::leak` on `encoding_name`.** `TokenCounter::encoding_name` returns `&'static str`; HF tokenizers do not embed a canonical name, so the operator-supplied `String` is leaked once at construction. One allocation per `from_bytes` call — expects the canonical "construct once at app boot" pattern. Constructing in a hot loop is anti-pattern.
- **Encode failure → `u64::MAX`.** `tokenizers::Tokenizer::encode` is fallible; on `Err`, `count` returns `u64::MAX` so `RunBudget` pre-flight checks fail closed (refuse the call rather than silently under-count). `tracing::warn!` records the underlying error.
- **`fancy-regex` only** — workspace pins `tokenizers` with `default-features = false, features = ["fancy-regex"]` to keep the build graph C-free (no `onig` / `esaxx_fast` native deps).

## Forbidden

- Adding a `from_file` / `from_pretrained` constructor (would require `std::fs` import and/or network IO inside the wrapper).
- Default-injecting a count on encode failure with a value other than `u64::MAX` (silent fallback — invariant 15).
