//! # entelix-memory-openai
//!
//! Concrete [`entelix_memory::Embedder`] implementation backed by
//! OpenAI's `/v1/embeddings` endpoint. Ships the
//! `text-embedding-3-{small,large}` models plus operator-supplied
//! custom model identifiers for forward compatibility.
//!
//! Companion to the trait-only [`entelix_memory`] crate (invariant
//! 13's "trait-only `entelix-memory`" principle): concrete vendor
//! impls live in their own crate so the core memory trait surface
//! ships without pulling `reqwest` + `secrecy` for users who
//! provide their own embedder.
//!
//! ## One-call setup
//!
//! ```ignore
//! use std::sync::Arc;
//! use entelix_core::auth::ApiKeyProvider;
//! use entelix_memory_openai::OpenAiEmbedder;
//!
//! let credentials = Arc::new(ApiKeyProvider::new(
//!     "authorization",
//!     format!("Bearer {}", std::env::var("OPENAI_API_KEY")?),
//! )?);
//! let embedder = OpenAiEmbedder::small().with_credentials(credentials).build()?;
//! ```
//!
//! ## Invariant alignment
//!
//! - **Invariant 10** — credentials never reach `Tool::execute`. The
//!   embedder holds an `Arc<dyn CredentialProvider>` and resolves
//!   per-call; no token enters request scope state.
//! - **F10** — `Embedder` is wrapped in `Arc` at the call boundary.
//!   Cloning the embedder is cheap (`Arc::clone` of the inner
//!   `reqwest::Client` plus a credential handle).
//! - **F4** — `EmbeddingUsage` is populated only on the `Ok` branch
//!   of `embed`/`embed_batch`; failed calls produce no phantom
//!   token charge in downstream meters.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_root_url = "https://docs.rs/entelix-memory-openai/0.5.0")]
#![deny(missing_docs)]
// `OpenAI` ride through doc strings as the vendor name (not as code
// identifier) frequently — backticking each occurrence noisily.
#![allow(clippy::doc_markdown)]

mod embedder;
mod error;

pub use embedder::{
    DEFAULT_BASE_URL, OpenAiEmbedder, OpenAiEmbedderBuilder, TEXT_EMBEDDING_3_LARGE,
    TEXT_EMBEDDING_3_LARGE_DIMENSION, TEXT_EMBEDDING_3_SMALL, TEXT_EMBEDDING_3_SMALL_DIMENSION,
};
pub use error::{OpenAiEmbedderError, OpenAiEmbedderResult};
