//! `EpisodicMemory<V>` — append-only, time-ordered store of
//! domain-shaped `Episode<V>` records keyed by [`Namespace`].
//!
//! Where [`crate::EntityMemory`] answers "what is the current fact
//! about X?" and [`crate::SemanticMemory`] answers "what stored
//! content resembles this query?", `EpisodicMemory` answers
//! questions about *time* — "what happened in this thread between
//! Tuesday and Friday?", "what were the last five things this
//! agent did?". The payload `V` stays operator-domain-shaped so
//! the same memory pattern serves conversation episodes,
//! task-completion records, decision logs, or any other
//! time-stamped event the agent wants to revisit.
//!
//! ## Storage shape
//!
//! Every namespace holds a single `Vec<Episode<V>>` under one
//! store key. The vector is maintained in non-decreasing
//! `timestamp` order — fresh appends use `Utc::now()` (always
//! ≥ the prior tail), and [`EpisodicMemory::append_at`]'s
//! caller-supplied timestamp is binary-inserted to preserve the
//! invariant. The single-key design mirrors [`crate::EntityMemory`]
//! so any [`crate::Store`] backend works unchanged. Companion
//! crates that need per-row indexing for very long histories
//! ship a dedicated backend without changing the surface here.
//!
//! ## Episode identity
//!
//! Each [`Episode`] carries an [`EpisodeId`] (UUID v7 — time
//! ordered) the operator can quote in audit trails or use to
//! correlate with external systems. The id is generated at
//! `append` time; callers backfilling from an external source can
//! call [`EpisodicMemory::append_record`] with a pre-built
//! [`Episode`] instead.

use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use entelix_core::{ExecutionContext, Result};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::namespace::Namespace;
use crate::store::Store;

/// Stable identifier for one episode. Backed by UUID v7 so two ids
/// minted in order compare in the same order — the audit trail and
/// external correlation paths stay consistent without a separate
/// sequence column.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EpisodeId(uuid::Uuid);

impl EpisodeId {
    /// Generate a fresh time-ordered id.
    #[must_use]
    pub fn new() -> Self {
        Self(uuid::Uuid::now_v7())
    }

    /// Reconstruct an id from a `uuid::Uuid` — used by persistence
    /// backends decoding stored rows.
    #[must_use]
    pub const fn from_uuid(uuid: uuid::Uuid) -> Self {
        Self(uuid)
    }

    /// Borrow the underlying UUID.
    #[must_use]
    pub const fn as_uuid(&self) -> &uuid::Uuid {
        &self.0
    }

    /// Render as a hyphenated string. Mirrors
    /// `CheckpointId::to_hyphenated_string` (entelix-graph) so id
    /// surfaces line up across audit channels.
    #[must_use]
    pub fn to_hyphenated_string(&self) -> String {
        self.0.to_string()
    }
}

impl Default for EpisodeId {
    fn default() -> Self {
        Self::new()
    }
}

/// One time-stamped episode of operator-shaped payload.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Episode<V> {
    /// Unique identifier (UUID v7).
    pub id: EpisodeId,
    /// Wall-clock time the episode was recorded.
    pub timestamp: DateTime<Utc>,
    /// Operator-supplied payload.
    pub payload: V,
}

const DEFAULT_KEY: &str = "episodes";

/// Time-ordered append-only episode store keyed by [`Namespace`].
pub struct EpisodicMemory<V>
where
    V: Clone + Serialize + DeserializeOwned + Send + Sync + 'static,
{
    store: Arc<dyn Store<Vec<Episode<V>>>>,
    namespace: Namespace,
    _marker: PhantomData<fn() -> V>,
}

impl<V> EpisodicMemory<V>
where
    V: Clone + Serialize + DeserializeOwned + Send + Sync + 'static,
{
    /// Build an episodic memory over `store` scoped to `namespace`.
    pub fn new(store: Arc<dyn Store<Vec<Episode<V>>>>, namespace: Namespace) -> Self {
        Self {
            store,
            namespace,
            _marker: PhantomData,
        }
    }

    /// Borrow the bound namespace.
    #[must_use]
    pub const fn namespace(&self) -> &Namespace {
        &self.namespace
    }

    /// Append a fresh episode timestamped at `Utc::now()`. Returns
    /// the id so the caller can correlate with audit / external
    /// systems.
    pub async fn append(&self, ctx: &ExecutionContext, payload: V) -> Result<EpisodeId> {
        let episode = Episode {
            id: EpisodeId::new(),
            timestamp: Utc::now(),
            payload,
        };
        let id = episode.id.clone();
        self.append_record(ctx, episode).await?;
        Ok(id)
    }

    /// Append at a caller-supplied timestamp. Use when backfilling
    /// from an external ledger or replaying historical events. The
    /// new entry is binary-inserted so the stored vector stays in
    /// non-decreasing `timestamp` order.
    pub async fn append_at(
        &self,
        ctx: &ExecutionContext,
        payload: V,
        timestamp: DateTime<Utc>,
    ) -> Result<EpisodeId> {
        let episode = Episode {
            id: EpisodeId::new(),
            timestamp,
            payload,
        };
        let id = episode.id.clone();
        self.append_record(ctx, episode).await?;
        Ok(id)
    }

    /// Append a fully-formed [`Episode`]. Use when the caller is
    /// migrating records minted elsewhere (a UUID + timestamp pair
    /// already exists). The entry is inserted at the correct
    /// position to preserve chronological order.
    pub async fn append_record(&self, ctx: &ExecutionContext, episode: Episode<V>) -> Result<()> {
        let mut all = self
            .store
            .get(ctx, &self.namespace, DEFAULT_KEY)
            .await?
            .unwrap_or_default();
        let pos = all.partition_point(|e| e.timestamp <= episode.timestamp);
        all.insert(pos, episode);
        self.store.put(ctx, &self.namespace, DEFAULT_KEY, all).await
    }

    /// Read every episode in chronological order. Empty namespaces
    /// return `Ok(vec![])`.
    pub async fn all(&self, ctx: &ExecutionContext) -> Result<Vec<Episode<V>>> {
        Ok(self
            .store
            .get(ctx, &self.namespace, DEFAULT_KEY)
            .await?
            .unwrap_or_default())
    }

    /// Most-recent-first slice of up to `n` episodes. `n = 0`
    /// returns an empty vector.
    pub async fn recent(&self, ctx: &ExecutionContext, n: usize) -> Result<Vec<Episode<V>>> {
        let mut all = self.all(ctx).await?;
        all.reverse();
        all.truncate(n);
        Ok(all)
    }

    /// Episodes whose `timestamp` falls in the inclusive range
    /// `[start, end]`. Order is chronological. `start > end`
    /// returns an empty vector rather than erroring — the question
    /// "what happened between two timestamps?" is well-defined
    /// even when the answer is empty.
    pub async fn range(
        &self,
        ctx: &ExecutionContext,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<Episode<V>>> {
        if start > end {
            return Ok(Vec::new());
        }
        let all = self.all(ctx).await?;
        let lo = all.partition_point(|e| e.timestamp < start);
        let hi = all.partition_point(|e| e.timestamp <= end);
        // partition_point returns 0..=len; `lo <= hi` holds because the
        // first predicate (`< start`) implies the second (`<= end`).
        // `into_iter().skip(lo).take(hi - lo)` avoids the indexing-
        // slicing lint without sacrificing clarity.
        Ok(all
            .into_iter()
            .skip(lo)
            .take(hi.saturating_sub(lo))
            .collect())
    }

    /// Episodes whose `timestamp` is greater than or equal to
    /// `start`. Order is chronological.
    pub async fn since(
        &self,
        ctx: &ExecutionContext,
        start: DateTime<Utc>,
    ) -> Result<Vec<Episode<V>>> {
        let all = self.all(ctx).await?;
        let lo = all.partition_point(|e| e.timestamp < start);
        Ok(all.into_iter().skip(lo).collect())
    }

    /// Total stored episode count.
    pub async fn count(&self, ctx: &ExecutionContext) -> Result<usize> {
        Ok(self.all(ctx).await?.len())
    }

    /// Drop every episode older than `ttl`. Returns the removal
    /// count so callers can log or expose pruning metrics. Atomic
    /// per-thread (single read-modify-write under one store key,
    /// matching [`crate::EntityMemory::prune_older_than`]).
    pub async fn prune_older_than(&self, ctx: &ExecutionContext, ttl: Duration) -> Result<usize> {
        let Some(mut all) = self.store.get(ctx, &self.namespace, DEFAULT_KEY).await? else {
            return Ok(0);
        };
        // chrono::Duration is signed and uses i64 nanoseconds; for
        // pathological ttls (above i64::MAX seconds) saturate to
        // chrono::Duration::MAX so the cutoff stays in the past.
        let cutoff = Utc::now() - chrono::Duration::from_std(ttl).unwrap_or(chrono::Duration::MAX);
        let before = all.len();
        all.retain(|e| e.timestamp >= cutoff);
        let removed = before - all.len();
        if removed > 0 {
            self.store
                .put(ctx, &self.namespace, DEFAULT_KEY, all)
                .await?;
        }
        Ok(removed)
    }

    /// Drop every episode in this namespace.
    pub async fn clear(&self, ctx: &ExecutionContext) -> Result<()> {
        self.store.delete(ctx, &self.namespace, DEFAULT_KEY).await
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::store::InMemoryStore;
    use entelix_core::TenantId;

    fn ns(scope: &str) -> Namespace {
        Namespace::new(TenantId::new("test-tenant")).with_scope(scope)
    }

    fn build() -> EpisodicMemory<String> {
        let store: Arc<dyn Store<Vec<Episode<String>>>> = Arc::new(InMemoryStore::new());
        EpisodicMemory::new(store, ns("conv"))
    }

    #[tokio::test]
    async fn append_then_all_returns_chronological_payloads() {
        let mem = build();
        let ctx = ExecutionContext::new();
        mem.append(&ctx, "first".to_owned()).await.unwrap();
        mem.append(&ctx, "second".to_owned()).await.unwrap();
        let all = mem.all(&ctx).await.unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].payload, "first");
        assert_eq!(all[1].payload, "second");
        assert!(all[0].timestamp <= all[1].timestamp);
    }

    #[tokio::test]
    async fn recent_returns_descending_capped() {
        let mem = build();
        let ctx = ExecutionContext::new();
        for i in 0..5 {
            mem.append(&ctx, format!("ep-{i}")).await.unwrap();
        }
        let recent = mem.recent(&ctx, 3).await.unwrap();
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].payload, "ep-4");
        assert_eq!(recent[1].payload, "ep-3");
        assert_eq!(recent[2].payload, "ep-2");
    }

    #[tokio::test]
    async fn recent_zero_returns_empty() {
        let mem = build();
        let ctx = ExecutionContext::new();
        mem.append(&ctx, "x".to_owned()).await.unwrap();
        assert!(mem.recent(&ctx, 0).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn range_filters_inclusive_endpoints() {
        let mem = build();
        let ctx = ExecutionContext::new();
        let base = Utc::now();
        for offset in [-30, -20, -10, 0, 10] {
            mem.append_at(
                &ctx,
                format!("t{offset}"),
                base + chrono::Duration::seconds(offset),
            )
            .await
            .unwrap();
        }
        let window = mem
            .range(
                &ctx,
                base + chrono::Duration::seconds(-20),
                base + chrono::Duration::seconds(0),
            )
            .await
            .unwrap();
        assert_eq!(
            window
                .iter()
                .map(|e| e.payload.as_str())
                .collect::<Vec<_>>(),
            vec!["t-20", "t-10", "t0"]
        );
    }

    #[tokio::test]
    async fn range_with_start_after_end_is_empty() {
        let mem = build();
        let ctx = ExecutionContext::new();
        mem.append(&ctx, "x".to_owned()).await.unwrap();
        let now = Utc::now();
        let later = now + chrono::Duration::seconds(60);
        assert!(mem.range(&ctx, later, now).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn since_returns_episodes_at_or_after_cutoff() {
        let mem = build();
        let ctx = ExecutionContext::new();
        let base = Utc::now();
        mem.append_at(&ctx, "old".to_owned(), base - chrono::Duration::seconds(60))
            .await
            .unwrap();
        mem.append_at(&ctx, "edge".to_owned(), base).await.unwrap();
        mem.append_at(&ctx, "new".to_owned(), base + chrono::Duration::seconds(60))
            .await
            .unwrap();
        let after = mem.since(&ctx, base).await.unwrap();
        assert_eq!(
            after.iter().map(|e| e.payload.as_str()).collect::<Vec<_>>(),
            vec!["edge", "new"]
        );
    }

    #[tokio::test]
    async fn append_at_preserves_chronological_invariant() {
        let mem = build();
        let ctx = ExecutionContext::new();
        let base = Utc::now();
        // Out-of-order arrivals — store must binary-insert.
        mem.append_at(
            &ctx,
            "late".to_owned(),
            base + chrono::Duration::seconds(60),
        )
        .await
        .unwrap();
        mem.append_at(
            &ctx,
            "early".to_owned(),
            base - chrono::Duration::seconds(60),
        )
        .await
        .unwrap();
        mem.append_at(&ctx, "mid".to_owned(), base).await.unwrap();
        let all = mem.all(&ctx).await.unwrap();
        assert_eq!(
            all.iter().map(|e| e.payload.as_str()).collect::<Vec<_>>(),
            vec!["early", "mid", "late"]
        );
    }

    #[tokio::test]
    async fn prune_older_than_drops_stale_and_returns_count() {
        let mem = build();
        let ctx = ExecutionContext::new();
        let now = Utc::now();
        mem.append_at(&ctx, "old".to_owned(), now - chrono::Duration::seconds(120))
            .await
            .unwrap();
        mem.append_at(&ctx, "fresh".to_owned(), now - chrono::Duration::seconds(5))
            .await
            .unwrap();
        let removed = mem
            .prune_older_than(&ctx, Duration::from_secs(60))
            .await
            .unwrap();
        assert_eq!(removed, 1);
        let remaining = mem.all(&ctx).await.unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].payload, "fresh");
    }

    #[tokio::test]
    async fn prune_on_empty_namespace_is_noop() {
        let mem = build();
        let ctx = ExecutionContext::new();
        assert_eq!(
            mem.prune_older_than(&ctx, Duration::from_secs(0))
                .await
                .unwrap(),
            0
        );
    }

    #[tokio::test]
    async fn count_and_clear_round_trip() {
        let mem = build();
        let ctx = ExecutionContext::new();
        for i in 0..3 {
            mem.append(&ctx, format!("e{i}")).await.unwrap();
        }
        assert_eq!(mem.count(&ctx).await.unwrap(), 3);
        mem.clear(&ctx).await.unwrap();
        assert_eq!(mem.count(&ctx).await.unwrap(), 0);
        assert!(mem.all(&ctx).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn namespaces_are_isolated() {
        let store: Arc<dyn Store<Vec<Episode<String>>>> = Arc::new(InMemoryStore::new());
        let alpha = EpisodicMemory::new(Arc::clone(&store), ns("alpha"));
        let beta = EpisodicMemory::new(store, ns("beta"));
        let ctx = ExecutionContext::new();
        alpha.append(&ctx, "alpha-1".to_owned()).await.unwrap();
        beta.append(&ctx, "beta-1".to_owned()).await.unwrap();
        let alpha_all = alpha.all(&ctx).await.unwrap();
        let beta_all = beta.all(&ctx).await.unwrap();
        assert_eq!(alpha_all.len(), 1);
        assert_eq!(beta_all.len(), 1);
        assert_eq!(alpha_all[0].payload, "alpha-1");
        assert_eq!(beta_all[0].payload, "beta-1");
    }

    #[tokio::test]
    async fn append_record_with_external_id_preserves_id() {
        let mem = build();
        let ctx = ExecutionContext::new();
        let id = EpisodeId::from_uuid(uuid::Uuid::now_v7());
        mem.append_record(
            &ctx,
            Episode {
                id: id.clone(),
                timestamp: Utc::now(),
                payload: "imported".to_owned(),
            },
        )
        .await
        .unwrap();
        let all = mem.all(&ctx).await.unwrap();
        assert_eq!(all[0].id, id);
    }
}
