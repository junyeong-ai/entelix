//! # entelix-graphmemory-pg
//!
//! Concrete [`entelix_memory::GraphMemory`] implementation backed by
//! Postgres (no extension required) — typed nodes + typed,
//! timestamped edges keyed by `entelix_memory::Namespace`.
//!
//! Companion to the trait-only [`entelix_memory`] crate:
//! sqlx specifics live here so users who plug their own
//! `GraphMemory` pay nothing in compile time.
//!
//! ## One-call setup
//!
//! ```ignore
//! use entelix_graphmemory_pg::PgGraphMemory;
//!
//! let graph = PgGraphMemory::<String, String>::builder()
//!     .with_connection_string("postgres://localhost/entelix")
//!     .build()
//!     .await?;
//! ```
//!
//! ## Multi-tenancy
//!
//! Two-table design (`graph_nodes` + `graph_edges`) with composite
//! `(namespace_key, id)` primary keys. Every read / write rides a
//! `WHERE namespace_key = $1` anchor — invariant 11 / F2 demand
//! structural tenant isolation, and the composite PK doubles as
//! the B-tree index that anchor relies on.
//!
//! Defense-in-depth via Postgres row-level security (analogous to
//!'s treatment of the `entelix-persistence` tables) is
//! reserved for a follow-up slice — operators that need it today
//! enable it externally with a policy that consults
//! `current_setting('entelix.tenant_id', true)` against the
//! `tenant_id` segment encoded in `namespace_key`.
//!
//! ## Schema-as-code escape hatch
//!
//! Operators that own the schema externally (DBA-managed, IaC,
//! migration pipeline) call [`PgGraphMemoryBuilder::with_auto_migrate`]
//! with `false` — the builder skips table / index creation,
//! trusting the operator to have stamped them.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_root_url = "https://docs.rs/entelix-graphmemory-pg/0.2.0")]
#![deny(missing_docs)]
// `Postgres` rides through doc strings as a vendor name. Crate-
// internal `pub(crate)` items inside private modules are the
// canonical helper pattern.
#![allow(
    clippy::doc_markdown,
    clippy::redundant_pub_crate,
    clippy::missing_const_for_fn,
    clippy::expect_used,
    // Three CREATE INDEX statements bind to `create_from_idx` /
    // `create_to_idx` / `create_ts_idx` — mirrors the SQL pattern;
    // the lint flags the deliberate parallel naming as confusable.
    clippy::similar_names
)]

mod error;
mod migration;
mod store;
mod tenant;

pub use error::{PgGraphMemoryError, PgGraphMemoryResult};
pub use store::{PgGraphMemory, PgGraphMemoryBuilder};
