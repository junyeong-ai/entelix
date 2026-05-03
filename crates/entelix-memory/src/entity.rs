//! `EntityMemory` — `entity_name → EntityRecord` map keyed by namespace.
//!
//! Each record carries the entity's current fact plus the wall-clock
//! time it was last confirmed (`last_seen`) and originally created
//! (`created_at`). Long-running agents call [`EntityMemory::prune_older_than`]
//! periodically to drop facts that have not been re-confirmed within
//! the configured TTL — without that, entity stores grow without
//! bound and stale facts pollute every retrieval.
//!
//! The entire map lives under a single store key so reads and writes
//! are atomic per-thread. Persistent backends that prefer one row per
//! entity can implement a dedicated `Store<EntityRecord>` variant
//! later — the trait surface stays the same.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use entelix_core::{ExecutionContext, Result};
use serde::{Deserialize, Serialize};

use crate::namespace::Namespace;
use crate::store::Store;

const DEFAULT_KEY: &str = "entities";

/// One entity's recorded fact plus provenance metadata.
///
/// `last_seen` is refreshed every time [`EntityMemory::set_entity`]
/// or [`EntityMemory::touch`] runs; reads do not advance it.
/// `created_at` is set once on first insertion and preserved across
/// subsequent updates so the audit trail of "when did we first
/// learn this entity?" stays intact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntityRecord {
    /// The current fact recorded for this entity.
    pub fact: String,
    /// Wall-clock time the fact was last confirmed (set or touched).
    pub last_seen: DateTime<Utc>,
    /// Wall-clock time the entity was first observed.
    pub created_at: DateTime<Utc>,
}

/// Map of `entity_name → EntityRecord` keyed by namespace.
pub struct EntityMemory {
    store: Arc<dyn Store<HashMap<String, EntityRecord>>>,
    namespace: Namespace,
}

impl EntityMemory {
    /// Build an entity memory over `store` scoped to `namespace`.
    pub fn new(store: Arc<dyn Store<HashMap<String, EntityRecord>>>, namespace: Namespace) -> Self {
        Self { store, namespace }
    }

    /// Borrow the bound namespace.
    pub const fn namespace(&self) -> &Namespace {
        &self.namespace
    }

    /// Insert or replace the fact for `entity`. `last_seen` is set
    /// to `Utc::now()`; `created_at` is preserved on update or set
    /// to `now` on first insertion.
    pub async fn set_entity(
        &self,
        ctx: &ExecutionContext,
        entity: &str,
        fact: impl Into<String>,
    ) -> Result<()> {
        let mut all = self
            .store
            .get(ctx, &self.namespace, DEFAULT_KEY)
            .await?
            .unwrap_or_default();
        let now = Utc::now();
        let fact = fact.into();
        match all.entry(entity.to_owned()) {
            std::collections::hash_map::Entry::Occupied(mut occ) => {
                let existing = occ.get_mut();
                existing.fact = fact;
                existing.last_seen = now;
            }
            std::collections::hash_map::Entry::Vacant(vac) => {
                vac.insert(EntityRecord {
                    fact,
                    last_seen: now,
                    created_at: now,
                });
            }
        }
        self.store.put(ctx, &self.namespace, DEFAULT_KEY, all).await
    }

    /// Refresh `last_seen` for `entity` without changing the fact.
    /// Use when the agent re-encounters an entity in a way that
    /// re-confirms relevance (the entity was mentioned again, even
    /// if no new fact was learned). Returns `Ok(false)` when the
    /// entity is not present so callers can distinguish absent vs
    /// touched.
    pub async fn touch(&self, ctx: &ExecutionContext, entity: &str) -> Result<bool> {
        let Some(mut all) = self.store.get(ctx, &self.namespace, DEFAULT_KEY).await? else {
            return Ok(false);
        };
        let Some(record) = all.get_mut(entity) else {
            return Ok(false);
        };
        record.last_seen = Utc::now();
        self.store
            .put(ctx, &self.namespace, DEFAULT_KEY, all)
            .await?;
        Ok(true)
    }

    /// Look up a single entity's fact. The lightweight ergonomic
    /// accessor — callers needing provenance use
    /// [`Self::entity_record`].
    pub async fn entity(&self, ctx: &ExecutionContext, entity: &str) -> Result<Option<String>> {
        Ok(self
            .entity_record(ctx, entity)
            .await?
            .map(|record| record.fact))
    }

    /// Look up a single entity's full record (fact + timestamps).
    pub async fn entity_record(
        &self,
        ctx: &ExecutionContext,
        entity: &str,
    ) -> Result<Option<EntityRecord>> {
        Ok(self
            .store
            .get(ctx, &self.namespace, DEFAULT_KEY)
            .await?
            .and_then(|all| all.get(entity).cloned()))
    }

    /// Read the `entity → fact` projection over every recorded
    /// record. Use [`Self::all_records`] to retain timestamps.
    pub async fn all(&self, ctx: &ExecutionContext) -> Result<HashMap<String, String>> {
        let records = self.all_records(ctx).await?;
        Ok(records
            .into_iter()
            .map(|(name, record)| (name, record.fact))
            .collect())
    }

    /// Read every recorded entity's full record.
    pub async fn all_records(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<HashMap<String, EntityRecord>> {
        Ok(self
            .store
            .get(ctx, &self.namespace, DEFAULT_KEY)
            .await?
            .unwrap_or_default())
    }

    /// Drop every record whose `last_seen` is older than `ttl` ago.
    /// Returns the number of records removed so callers can log
    /// or expose pruning metrics.
    ///
    /// Runs as a single read-modify-write under the namespace's
    /// store key, so the prune is atomic per-thread.
    pub async fn prune_older_than(&self, ctx: &ExecutionContext, ttl: Duration) -> Result<usize> {
        let Some(mut all) = self.store.get(ctx, &self.namespace, DEFAULT_KEY).await? else {
            return Ok(0);
        };
        // chrono::Duration is signed and uses i64 nanoseconds; for
        // pathological ttls (above i64::MAX seconds) saturate to
        // chrono::Duration::MAX so the cutoff stays in the past.
        let cutoff = Utc::now() - chrono::Duration::from_std(ttl).unwrap_or(chrono::Duration::MAX);
        let before = all.len();
        all.retain(|_, record| record.last_seen >= cutoff);
        let removed = before - all.len();
        if removed > 0 {
            self.store
                .put(ctx, &self.namespace, DEFAULT_KEY, all)
                .await?;
        }
        Ok(removed)
    }

    /// Remove a single entity. Idempotent — removing an absent
    /// entity is a no-op.
    pub async fn remove(&self, ctx: &ExecutionContext, entity: &str) -> Result<()> {
        let Some(mut all) = self.store.get(ctx, &self.namespace, DEFAULT_KEY).await? else {
            return Ok(());
        };
        all.remove(entity);
        self.store.put(ctx, &self.namespace, DEFAULT_KEY, all).await
    }

    /// Clear every entity in this namespace.
    pub async fn clear(&self, ctx: &ExecutionContext) -> Result<()> {
        self.store.delete(ctx, &self.namespace, DEFAULT_KEY).await
    }
}
