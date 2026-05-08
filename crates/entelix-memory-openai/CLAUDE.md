# entelix-memory-openai

Companion crate. Concrete `Embedder` impl for OpenAI's Embeddings API (`text-embedding-3-{small,large}`).

## Surface

- **`OpenAiEmbedder`** + **`OpenAiEmbedderBuilder`** — `with_api_key(.)` / `with_model(.)` / `with_base_url(.)` / `build() -> Result<Self>`. Pool-shared via `Arc<Self>` — never per-call client construction.
- **Constants** — `TEXT_EMBEDDING_3_SMALL` / `TEXT_EMBEDDING_3_LARGE` model identifiers + their `*_DIMENSION` companion constants. `DEFAULT_BASE_URL` for OpenAI's hosted endpoint.
- **`OpenAiEmbedderError`** — typed error wrapping HTTP status, vendor message, and serde failures. `#[non_exhaustive]`.

## Crate-local rules

- **`Arc<Self>` constraint** — every `Embedder` impl must be cheap to clone. `OpenAiEmbedder` holds `Arc<reqwest::Client>` internally so `clone()` doesn't reconstruct the connection pool.
- **`embed_batch` calls the vendor batch endpoint** — single HTTP request per `Vec<&str>` input, not N parallel singletons. The vendor charges per-token regardless, but batch saves request overhead.
- **Cancellation polled before HTTP send** + **between batch chunks** — `ctx.is_cancelled()` checked at the top of `embed` / `embed_batch`. Regression-tested in `tests/openai_e2e.rs::cancelled_context_short_circuits_before_http_call`.
- **No PII in error messages** — vendor errors are normalized to drop request-body content (which may contain user PII) before populating `OpenAiEmbedderError::Malformed`.

## Forbidden

- Per-call `reqwest::Client::new()` construction — defeats the connection pool.
- Bypassing `gen_ai.embedding.cost` emission on the `Ok` branch (invariant 12 — cost is transactional, only on success).
