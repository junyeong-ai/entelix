//! Redis-backed persistence — `Checkpointer<S>`, `Store<V>`,
//! `SessionLog`, and a `DistributedLock` over `SET NX PX` plus a Lua
//! release script.
//!
//! Schema:
//! - Checkpoints: `entelix:cp:{thread_id}:{step}` → JSON blob,
//!   plus `entelix:cp:{thread_id}:latest` pointer
//! - Memory: `entelix:mem:{tenant_id}:{namespace}:{key}` → JSON blob
//! - Session events: `entelix:session:{tenant_id}:{thread_id}:events`
//!   → list (LPUSH for monotonic ordinal via list length)

mod checkpointer;
mod lock;
mod persistence;
mod session_log;
mod store;

pub use checkpointer::RedisCheckpointer;
pub use lock::RedisLock;
pub use persistence::{RedisPersistence, RedisPersistenceBuilder};
pub use session_log::RedisSessionLog;
pub use store::RedisStore;
