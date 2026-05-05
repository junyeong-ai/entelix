//! `InMemoryCheckpointer` — in-process `Checkpointer<S>` impl.
//!
//! Uses `parking_lot::Mutex` internally for synchronous access; the
//! trait methods are async only to match the [`Checkpointer`]
//! interface, not because the implementation blocks. Suited for
//! single-process tests and short-lived agents where durability
//! across crashes is not required. Postgres / Redis backed
//! checkpointers live in `entelix-persistence`.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use entelix_core::{Error, Result, TenantId, ThreadKey};
use parking_lot::Mutex;

use crate::checkpoint::{Checkpoint, CheckpointId, Checkpointer};

/// Internal partition key — bypasses the public `ThreadKey` so the
/// `HashMap` can be cheaply cloned for entry lookups without
/// constructing a new `ThreadKey` per operation. Cloning a
/// [`TenantId`] is an `Arc<str>` refcount bump, so the partition
/// remains cheap to materialise.
type Partition = (TenantId, String);

fn partition(key: &ThreadKey) -> Partition {
    (key.tenant_id().clone(), key.thread_id().to_owned())
}

/// In-process checkpointer backed by a
/// `HashMap<(tenant_id, thread_id), Vec<Checkpoint>>`. The
/// composite key encodes Invariant 11 (multi-tenant isolation):
/// the same `thread_id` under two tenants resolves to two distinct
/// histories.
///
/// Cheap to clone — internal state is `Arc<Mutex<...>>`-shared.
#[derive(Clone)]
pub struct InMemoryCheckpointer<S>
where
    S: Clone + Send + Sync + 'static,
{
    inner: Arc<Mutex<HashMap<Partition, Vec<Checkpoint<S>>>>>,
}

impl<S> InMemoryCheckpointer<S>
where
    S: Clone + Send + Sync + 'static,
{
    /// Empty checkpointer.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Total number of checkpoints stored across all
    /// `(tenant_id, thread_id)` partitions. Test helper.
    pub fn total_checkpoints(&self) -> usize {
        self.inner.lock().values().map(Vec::len).sum()
    }

    /// Number of distinct `(tenant_id, thread_id)` partitions that
    /// have at least one checkpoint.
    pub fn thread_count(&self) -> usize {
        self.inner.lock().len()
    }
}

impl<S> Default for InMemoryCheckpointer<S>
where
    S: Clone + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<S> Checkpointer<S> for InMemoryCheckpointer<S>
where
    S: Clone + Send + Sync + 'static,
{
    async fn put(&self, checkpoint: Checkpoint<S>) -> Result<()> {
        let key = (checkpoint.tenant_id.clone(), checkpoint.thread_id.clone());
        // Vec::push may reallocate, dropping the previous backing
        // buffer (and therefore each `S` it held) while we hold the
        // mutex. Per the `Checkpointer` trait contract `S::drop` does
        // not block, so this is safe — but worth flagging for
        // implementors of new state types.
        self.inner.lock().entry(key).or_default().push(checkpoint);
        Ok(())
    }

    async fn latest(&self, key: &ThreadKey) -> Result<Option<Checkpoint<S>>> {
        let guard = self.inner.lock();
        Ok(guard
            .get(&partition(key))
            .and_then(|history| history.last().cloned()))
    }

    async fn by_id(&self, key: &ThreadKey, id: &CheckpointId) -> Result<Option<Checkpoint<S>>> {
        let guard = self.inner.lock();
        Ok(guard
            .get(&partition(key))
            .and_then(|h| h.iter().find(|cp| &cp.id == id).cloned()))
    }

    async fn history(&self, key: &ThreadKey, limit: usize) -> Result<Vec<Checkpoint<S>>> {
        let guard = self.inner.lock();
        Ok(guard
            .get(&partition(key))
            .map(|h| h.iter().rev().take(limit).cloned().collect::<Vec<_>>())
            .unwrap_or_default())
    }

    async fn update_state(
        &self,
        key: &ThreadKey,
        parent_id: &CheckpointId,
        new_state: S,
    ) -> Result<CheckpointId> {
        let part = partition(key);
        // Look up the parent's bits, drop the read guard, then build
        // the error or new checkpoint outside the lock scope.
        let parent_bits: Option<(Option<String>, usize)> = {
            let guard = self.inner.lock();
            guard
                .get(&part)
                .and_then(|h| h.iter().find(|cp| &cp.id == parent_id))
                .map(|cp| (cp.next_node.clone(), cp.step.saturating_add(1)))
        };
        let (next_node, step) = parent_bits.ok_or_else(|| {
            Error::invalid_request(format!(
                "InMemoryCheckpointer::update_state: unknown parent_id in tenant '{}' thread '{}'",
                key.tenant_id(),
                key.thread_id()
            ))
        })?;
        let new_checkpoint =
            Checkpoint::new(key, step, new_state, next_node).with_parent(parent_id.clone());
        let new_id = new_checkpoint.id.clone();
        self.inner
            .lock()
            .entry(part)
            .or_default()
            .push(new_checkpoint);
        Ok(new_id)
    }
}
