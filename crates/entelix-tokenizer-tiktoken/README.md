# entelix-tokenizer-tiktoken

Vendor-accurate [`TokenCounter`](https://docs.rs/entelix-core/latest/entelix_core/trait.TokenCounter.html)
for OpenAI's BPE tokenizer family — `cl100k_base`, `o200k_base`,
`p50k_base`, `r50k_base`. Wraps [`tiktoken-rs`](https://crates.io/crates/tiktoken-rs)
with eager BPE preload at construction so the per-call `count` stays
synchronous per the `TokenCounter` contract.

```toml
[dependencies]
entelix-core = "0.3"
entelix-tokenizer-tiktoken = "0.3"
```

```rust,ignore
use std::sync::Arc;
use entelix_core::{ChatModelConfig, TokenCounter};
use entelix_tokenizer_tiktoken::{TiktokenCounter, TiktokenEncoding};

let counter: Arc<dyn TokenCounter> =
    Arc::new(TiktokenCounter::for_encoding(TiktokenEncoding::O200kBase)?);

let config = ChatModelConfig::default().with_token_counter(Arc::clone(&counter));

assert_eq!(counter.count("Hello world"), 2);
```

## Encoding to model mapping

| Encoding | Models |
|---|---|
| `Cl100kBase` | GPT-3.5-turbo, GPT-4, GPT-4-turbo, text-embedding-3-* |
| `O200kBase` | GPT-4o, GPT-4o-mini, o1, o3, o3-mini, o4 |
| `P50kBase` | GPT-3 davinci, codex |
| `R50kBase` | GPT-3 ada / babbage / curie, GPT-2 |

The mapping is left to operators by design — OpenAI changes it over
time, and accidentally pinning to a stale mapping silently miscounts
without surfacing a build failure. Pick the encoding for your target
model from the table above and the wrapper preloads the matching BPE
tables.

## Position in the workspace

Companion crate to `entelix-core` — `TokenCounter` is the trait
surface; this crate provides one of several concrete impls. Sister
crates ship the HuggingFace Tokenizers backend (`entelix-tokenizer-hf`)
and locale-aware morphological counters (Korean / Japanese).

The zero-dependency `entelix_core::ByteCountTokenCounter` ships in
core for development scaffolding; production multilingual workloads
should swap in this crate (or a sibling) at
[`ChatModelConfig::with_token_counter`](https://docs.rs/entelix-core/latest/entelix_core/struct.ChatModelConfig.html#method.with_token_counter).

## License

MIT
