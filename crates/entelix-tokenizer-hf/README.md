# entelix-tokenizer-hf

Vendor-accurate [`TokenCounter`](https://docs.rs/entelix-core/latest/entelix_core/trait.TokenCounter.html)
wrapping the [HuggingFace `tokenizers`](https://crates.io/crates/tokenizers)
crate. Construct from any `tokenizer.json` byte payload — Llama 3, Qwen 2.5,
Mistral, DeepSeek, Gemma, Phi, and every other model whose tokenizer is
published in HF's standard format.

```toml
[dependencies]
entelix-core = "0.4"
entelix-tokenizer-hf = "0.4"
```

```rust,ignore
use std::sync::Arc;
use entelix_core::{ChatModelConfig, TokenCounter};
use entelix_tokenizer_hf::HfTokenCounter;

let bytes = std::fs::read("./llama-3-tokenizer.json")?;
let counter: Arc<dyn TokenCounter> =
    Arc::new(HfTokenCounter::from_bytes(&bytes, "llama-3")?);

let config = ChatModelConfig::default().with_token_counter(Arc::clone(&counter));
```

## Why bytes-only construction

The wrapper exposes [`HfTokenCounter::from_bytes`] but no `from_file` or
`from_pretrained` constructor. Two reasons:

- **Invariant 9 alignment** — entelix first-party crates do not import
  `std::fs`. Operators read tokenizer files in their own application code
  (or at compile time via `include_bytes!`) and pass the byte payload in.
- **No silent network IO** — `tokenizers::Tokenizer::from_pretrained` does
  HTTP downloads and disk caching as a side effect. SDK consumers that
  need hub integration wire it explicitly in their boot path; the wrapper
  stays pure.

## Pairing with `TokenCountSplitter`

```rust,ignore
use std::sync::Arc;
use entelix_core::TokenCounter;
use entelix_rag::TokenCountSplitter;
use entelix_tokenizer_hf::HfTokenCounter;

let bytes = std::fs::read("./qwen2.5-tokenizer.json")?;
let counter: Arc<dyn TokenCounter> =
    Arc::new(HfTokenCounter::from_bytes(&bytes, "qwen-2.5")?);

let splitter = TokenCountSplitter::new(counter)
    .with_chunk_size(512)
    .with_chunk_overlap(64);
```

## Position in the workspace

Sister crate to `entelix-tokenizer-tiktoken` — both implement the same
[`TokenCounter`](https://docs.rs/entelix-core/latest/entelix_core/trait.TokenCounter.html)
trait. Pick the wrapper whose tokenizer family matches the target model:

| Vendor | Wrapper |
|---|---|
| OpenAI (GPT-3.5, GPT-4, GPT-4o, o1, o3, embeddings) | `entelix-tokenizer-tiktoken` |
| Llama, Qwen, Mistral, DeepSeek, Gemma, Phi | `entelix-tokenizer-hf` |
| Korean / Japanese morphological accuracy | future: `entelix-tokenizer-ko-mecab`, `entelix-tokenizer-lindera` |

## License

MIT
