//! # entelix-cloud
//!
//! Cloud-routed [`entelix_core::transports::Transport`] impls —
//! `BedrockTransport` (AWS), `VertexTransport` (GCP), `FoundryTransport`
//! (Azure) — plus the OAuth / SigV4 / AAD refresh logic shared across
//! them. The provider IR (`entelix-core`) is unchanged across cloud
//! routes; only credential resolution, request signing, and
//! transport-specific framing (e.g. AWS event-stream) live here.
//!
//! Cargo features:
//! - `aws` — pulls `aws-sigv4`, `aws-config`, `aws-credential-types`,
//!   enables the `bedrock` module.
//! - `gcp` — pulls `gcp_auth`, enables the `vertex` module (Slice C).
//! - `azure` — pulls `azure_identity`, enables the `foundry` module
//!   (Slice C).

#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_root_url = "https://docs.rs/entelix-cloud/0.3.0")]
#![deny(missing_docs)]
// Cloud transports translate between AWS / GCP / Azure SDK error
// shapes many times; the indexing / cast lints below are exercised
// by binary frame parsing whose offsets are bounded by upstream
// length-prefix checks.
#![allow(
    clippy::indexing_slicing,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::checked_conversions,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::redundant_closure_for_method_calls,
    clippy::needless_pass_by_value,
    clippy::missing_const_for_fn,
    clippy::needless_collect,
    clippy::needless_lifetimes,
    clippy::doc_markdown
)]

mod error;
pub mod refresh;

#[cfg(feature = "aws")]
#[cfg_attr(docsrs, doc(cfg(feature = "aws")))]
pub mod bedrock;

#[cfg(feature = "gcp")]
#[cfg_attr(docsrs, doc(cfg(feature = "gcp")))]
pub mod vertex;

#[cfg(feature = "azure")]
#[cfg_attr(docsrs, doc(cfg(feature = "azure")))]
pub mod foundry;

pub use error::CloudError;
