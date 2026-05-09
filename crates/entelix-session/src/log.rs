// Mutex guards in this module are short-lived and never held across an
// `await`; the `tightening` lint's recommendation to early-drop them
// would force opaque closures over every method body.
#![allow(clippy::significant_drop_tightening)]

//! `SessionLog` trait — the persistent companion to [`SessionGraph`].
//!
//! `SessionGraph` is in-memory; `SessionLog` is the durable audit
//! store backing it. Concrete impls live elsewhere:
//! - [`InMemorySessionLog`] (this crate) — in-process default for tests
//!   and single-binary deployments
//! - `entelix_persistence::postgres::PostgresSessionLog` — Postgres
//! - `entelix_persistence::redis::RedisSessionLog` — Redis
//!
//! All implementations share these guarantees:
//!
//! - **Append-only**: appended events never mutate, never reorder
//!   (invariant 1).
//! - **Per-thread monotonic ordinal**: `append` returns the next
//!   integer; `load_since(cursor)` returns events with ordinal >
//!   `cursor`.
//! - **Tenant-scoped**: every method takes `tenant_id` (invariant 11)
//!   so cross-tenant reads / writes are structurally impossible.
//! - **Archival watermark monotonic**: once `archive_before(w)` is
//!   called, `w` only ever moves forward.

use std::collections::HashMap;

use async_trait::async_trait;
use entelix_core::{Result, ThreadKey};
use parking_lot::Mutex;

use crate::event::GraphEvent;

/// Persistent durable session-event log.
///
/// `SessionGraph::events` is the in-memory shape; this trait is the
/// durable companion that survives process restarts. A `SessionGraph`
/// can be hydrated from a `SessionLog` by replaying every event the
/// log returns for `key`.
///
/// Every method is keyed by [`ThreadKey`], the canonical
/// `(tenant_id, thread_id)` tuple — a backend cannot accidentally
/// drop the tenant scope because the tenant component is
/// syntactically required by the type signature. Same isolation
/// pattern that `entelix_graph::Checkpointer` uses (Invariant 11 /
/// F2).
#[async_trait]
pub trait SessionLog: Send + Sync + 'static {
    /// Append `events` for `key`. Returns the highest ordinal
    /// assigned (1-based, so an empty log becomes ordinal 1 after
    /// appending the first event).
    async fn append(&self, key: &ThreadKey, events: &[GraphEvent]) -> Result<u64>;

    /// Load every event with ordinal `> cursor`. Pass `0` for "from
    /// the beginning". Returns events in ordinal-ascending order.
    async fn load_since(&self, key: &ThreadKey, cursor: u64) -> Result<Vec<GraphEvent>>;

    /// Advance the archival watermark to `watermark`. Events with
    /// ordinal `<= watermark` may be moved to cold storage at the
    /// implementation's discretion. Returns the number of events
    /// archived. The watermark is monotonic per `key`; calls with
    /// a value `<=` the current watermark are no-ops.
    async fn archive_before(&self, key: &ThreadKey, watermark: u64) -> Result<usize>;
}

/// In-process [`SessionLog`] for single-binary deployments and tests.
///
/// Backed by a per-[`ThreadKey`] `Vec<GraphEvent>`. Production
/// multi-pod deployments use the Postgres or Redis impls in
/// `entelix-persistence`.
#[derive(Default)]
pub struct InMemorySessionLog {
    inner: Mutex<HashMap<ThreadKey, ThreadLog>>,
}

#[derive(Default)]
struct ThreadLog {
    events: Vec<GraphEvent>,
    archival_watermark: u64,
}

impl InMemorySessionLog {
    /// Build an empty in-memory log.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl SessionLog for InMemorySessionLog {
    async fn append(&self, key: &ThreadKey, events: &[GraphEvent]) -> Result<u64> {
        let len = {
            let mut guard = self.inner.lock();
            let log = guard.entry(key.clone()).or_default();
            log.events.extend(events.iter().cloned());
            log.events.len()
        };
        Ok(len as u64)
    }

    async fn load_since(&self, key: &ThreadKey, cursor: u64) -> Result<Vec<GraphEvent>> {
        let snapshot = {
            let guard = self.inner.lock();
            let Some(log) = guard.get(key) else {
                return Ok(Vec::new());
            };
            // Honor archival semantics: events with ordinal
            // `<= archival_watermark` are conceptually gone, even
            // though this in-memory impl retains them in the Vec
            // for replay tooling. Postgres `archive_before` DELETEs
            // those rows, and Redis `LTRIM`s them, so both backends
            // skip ordinals at or below the watermark regardless of
            // the cursor. Clamping `start` here keeps the in-memory
            // backend's observable behaviour consistent with the
            // durable backends — otherwise a caller archiving and
            // then reading from cursor `0` would see archived
            // events on the in-memory backend but not on Postgres
            // or Redis, silently breaking cross-backend tests.
            let effective_start = cursor.max(log.archival_watermark);
            let start = usize::try_from(effective_start).unwrap_or(usize::MAX);
            log.events.get(start..).unwrap_or(&[]).to_vec()
        };
        Ok(snapshot)
    }

    async fn archive_before(&self, key: &ThreadKey, watermark: u64) -> Result<usize> {
        let archived = {
            let mut guard = self.inner.lock();
            let Some(log) = guard.get_mut(key) else {
                return Ok(0);
            };
            if watermark <= log.archival_watermark {
                return Ok(0);
            }
            let prior = log.archival_watermark;
            log.archival_watermark = watermark;
            watermark.saturating_sub(prior)
        };
        Ok(usize::try_from(archived).unwrap_or(usize::MAX))
    }
}
