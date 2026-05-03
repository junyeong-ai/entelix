//! Live-API smoke for [`OpenAiEmbedder`].
//!
//! `#[ignore]`-gated. Run with:
//!
//! ```text
//! OPENAI_API_KEY=sk-... \
//!     cargo test -p entelix-memory-openai --test live_openai_embedder -- --ignored
//! ```
//!
//! Optional override: `ENTELIX_LIVE_OPENAI_EMBEDDING_MODEL`
//! (default `text-embedding-3-small`).
//!
//! ## Cost discipline
//!
//! One embedding call on a single short string against the
//! `text-embedding-3-small` tier — fractions of a cent per run.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::doc_markdown,
    clippy::option_if_let_else
)]

use std::sync::Arc;

use entelix_core::auth::{BearerProvider, CredentialProvider};
use entelix_core::context::ExecutionContext;
use entelix_memory::Embedder;
use entelix_memory_openai::{OpenAiEmbedder, TEXT_EMBEDDING_3_SMALL_DIMENSION};
use secrecy::SecretString;

#[tokio::test]
#[ignore = "live-API: requires OPENAI_API_KEY"]
async fn openai_embedder_minimal_round_trip() {
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("set OPENAI_API_KEY to run this live-API smoke");
    let model_override = std::env::var("ENTELIX_LIVE_OPENAI_EMBEDDING_MODEL").ok();

    // OpenAI authenticates with `Authorization: Bearer …` — the
    // `BearerProvider` helper wraps the token + canonical header
    // name in one call, matching the chat smokes.
    let credentials: Arc<dyn CredentialProvider> =
        Arc::new(BearerProvider::new(SecretString::from(api_key)));

    let mut builder = match model_override {
        Some(model) => OpenAiEmbedder::custom(model, TEXT_EMBEDDING_3_SMALL_DIMENSION),
        None => OpenAiEmbedder::small(),
    };
    builder = builder.with_credentials(credentials);
    let embedder = builder.build().expect("OpenAiEmbedder built");

    let ctx = ExecutionContext::new();
    let embedding = embedder
        .embed("entelix live-API smoke", &ctx)
        .await
        .expect("openai embedder live call");

    assert_eq!(
        embedding.vector.len(),
        TEXT_EMBEDDING_3_SMALL_DIMENSION,
        "vector dimension must equal the configured one"
    );
    let nonzero = embedding.vector.iter().any(|x| *x != 0.0);
    assert!(nonzero, "live embedding should not be all-zero");

    let usage = embedding.usage.expect("usage must be populated by OpenAI");
    assert!(
        usage.input_tokens > 0,
        "usage.input_tokens must be > 0 on a real call"
    );
}
