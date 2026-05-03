//! Postgres-backed persistence — `Checkpointer<S>`, `Store<V>`,
//! `SessionLog`, and a `DistributedLock` over `pg_advisory_xact_lock`.
//!
//! All four live behind a single [`PostgresPersistence`] handle that
//! owns a `sqlx::PgPool`. The bundle is the public surface; callers
//! pull individual traits out of it via accessor methods.

mod checkpointer;
mod lock;
mod persistence;
mod session_log;
mod store;
mod tenant;

pub use checkpointer::PostgresCheckpointer;
pub use lock::PostgresLock;
pub use persistence::{PostgresPersistence, PostgresPersistenceBuilder};
pub use session_log::PostgresSessionLog;
pub use store::PostgresStore;
