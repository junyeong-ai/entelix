//! # entelix-persistence
//!
//! Postgres + Redis backends for the three storage traits owned by
//! other crates: [`entelix_graph::Checkpointer`],
//! [`entelix_memory::Store`], and the session event log
//! ([`entelix_session::SessionGraph`]). Each backend is a
//! `*Persistence` bundle — a single builder produces a
//! `(Checkpointer, Store, SessionLog)` triplet sharing one connection
//! pool / lock backend.
//!
//! Distributed session locking lives behind the [`DistributedLock`]
//! trait; [`with_session_lock`] composes lock acquisition with a
//! caller closure (invariant 8 — durable session writes happen under
//! a lock).
//!
//! Cargo features:
//! - `postgres` — pulls `sqlx`, enables the `postgres` module.
//! - `redis` — pulls `redis`, enables the `redis` module.
//! - `test-containers` — pulls `testcontainers` for ephemeral docker
//!   integration tests; implies both backends.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_root_url = "https://docs.rs/entelix-persistence/0.4.0")]
#![deny(missing_docs)]
// Backend modules cross-translate `sqlx`/`redis` errors many times; the
// `&Error` form fights with the implicit `Into` conversions sqlx
// returns. The cast lints below are exercised by integer-space
// arithmetic in serialisation; we audit at the call sites.
#![allow(
    clippy::needless_pass_by_value,
    clippy::missing_const_for_fn,
    clippy::redundant_closure,
    clippy::redundant_closure_for_method_calls,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::indexing_slicing,
    clippy::expect_used,
    clippy::single_match_else,
    clippy::manual_let_else,
    clippy::needless_pass_by_ref_mut
)]

pub mod advisory_key;
pub mod error;
pub mod lock;
pub mod schema_version;

#[cfg(feature = "postgres")]
#[cfg_attr(docsrs, doc(cfg(feature = "postgres")))]
pub mod postgres;

#[cfg(feature = "redis")]
#[cfg_attr(docsrs, doc(cfg(feature = "redis")))]
pub mod redis;

pub use advisory_key::AdvisoryKey;
pub use error::{PersistenceError, PersistenceResult};
pub use lock::{DistributedLock, LockGuard, with_session_lock};
pub use schema_version::SessionSchemaVersion;
