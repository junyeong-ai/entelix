//! # entelix-memory-qdrant
//!
//! Concrete [`entelix_memory::VectorStore`] implementation backed by
//! the qdrant 1.5+ gRPC API.
//!
//! Companion to the trait-only [`entelix_memory`] crate:
//! vendor-SDK dependencies (`qdrant-client` plus its tonic / prost
//! transitive tree) live here so users who plug their own
//! `VectorStore` pay nothing in compile time.
//!
//! ## One-call setup
//!
//! ```ignore
//! use entelix_memory_qdrant::{DistanceMetric, QdrantVectorStore};
//!
//! let store = QdrantVectorStore::builder("memories", 1536)
//!     .with_url("http://localhost:6334")
//!     .with_distance(DistanceMetric::Cosine)
//!     .build()
//!     .await?;
//! ```
//!
//! ## Multi-tenancy
//!
//! Single-collection design (qdrant official multi-tenancy
//! guidance): every read / write / count / list rides a `must`
//! anchor on the rendered [`entelix_memory::Namespace`]. Two
//! tenants sharing the same operator-facing `doc_id` are stored as
//! distinct points without coordination — the qdrant `PointId` is
//! a deterministic UUID v5 of `(namespace_key, doc_id)`. The
//! original `doc_id` survives in the payload for round-trip.
//!
//! ## Schema-as-code escape hatch
//!
//! Operators that provision the qdrant collection externally
//! (helm chart, Terraform, qdrant Cloud console) call
//! [`QdrantVectorStoreBuilder::with_existing_collection`] — the
//! builder skips both collection creation and payload-index
//! provisioning and trusts the operator to have stamped them.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_root_url = "https://docs.rs/entelix-memory-qdrant/0.3.0")]
#![deny(missing_docs)]
// `qdrant`, `Postgres`, etc. ride through doc strings as vendor names
// (not code identifiers); backticking each occurrence is noise.
// `Err`-variant size is large because `QdrantStoreError::Sqlx` boxes
// the upstream chain — that's the point of `#[from]` and we accept
// the heap-shaped Result on these cold paths.
// `pub(crate)` items inside private modules are the canonical
// crate-internal helper pattern.
#![allow(
    clippy::doc_markdown,
    clippy::too_long_first_doc_paragraph,
    clippy::result_large_err,
    clippy::pub_with_shorthand,
    clippy::redundant_pub_crate,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::redundant_closure_for_method_calls,
    clippy::needless_pass_by_value,
    clippy::map_unwrap_or,
    clippy::indexing_slicing,
    clippy::expect_used
)]

mod error;
mod filter;
mod store;

pub use error::{QdrantStoreError, QdrantStoreResult};
pub use filter::{CONTENT_KEY, DOC_ID_KEY, METADATA_KEY, NAMESPACE_KEY};
pub use store::{DistanceMetric, QdrantVectorStore, QdrantVectorStoreBuilder};
