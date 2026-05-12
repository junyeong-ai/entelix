//! # entelix-memory-pgvector
//!
//! Concrete [`entelix_memory::VectorStore`] implementation backed by
//! Postgres + the pgvector extension.
//!
//! Companion to the trait-only [`entelix_memory`] crate:
//! sqlx + pgvector specifics live here so users who plug their own
//! `VectorStore` pay nothing in compile time.
//!
//! ## One-call setup
//!
//! ```ignore
//! use entelix_memory_pgvector::{DistanceMetric, IndexKind, PgVectorStore};
//!
//! let store = PgVectorStore::builder(1536)
//!     .with_connection_string("postgres://localhost/entelix")
//!     .with_distance(DistanceMetric::Cosine)
//!     .with_index_kind(IndexKind::Hnsw)
//!     .build()
//!     .await?;
//! ```
//!
//! ## Multi-tenancy
//!
//! Single-table design with a composite `(namespace_key, doc_id)`
//! primary key. Every read / write / count / list rides a
//! `WHERE namespace_key = $1` anchor — invariant 11 / F2 demand
//! structural tenant isolation, and the composite PK doubles as
//! the B-tree index that anchor relies on.
//!
//! ## Schema-as-code escape hatch
//!
//! Operators that own the schema externally (DBA-managed, IaC,
//! migration pipeline) call [`PgVectorStoreBuilder::with_auto_migrate`]
//! with `false` — the builder
//! skips extension creation, table creation, and index
//! provisioning, trusting the operator to have stamped them.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_root_url = "https://docs.rs/entelix-memory-pgvector/0.5.0")]
#![deny(missing_docs)]
// `Postgres`, `pgvector` ride through doc strings as vendor names.
// `pub(crate)` items inside private modules are the canonical
// crate-internal helper pattern.
#![allow(
    clippy::doc_markdown,
    clippy::redundant_pub_crate,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::redundant_closure_for_method_calls,
    clippy::missing_const_for_fn,
    clippy::expect_used
)]

mod error;
mod filter;
mod migration;
mod store;
mod tenant;

pub use error::{PgVectorStoreError, PgVectorStoreResult};
pub use store::{DistanceMetric, IndexKind, PgVectorStore, PgVectorStoreBuilder};
