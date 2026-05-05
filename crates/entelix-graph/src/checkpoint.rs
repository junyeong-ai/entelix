//! Checkpoint primitives — `Checkpoint<S>`, `CheckpointId`, the
//! addressing tuple [`ThreadKey`], and the [`Checkpointer`] trait.
//!
//! A checkpoint records "after running step N at node X with state S,
//! the graph plans to execute Y next". On crash recovery, a fresh
//! process calls `CompiledGraph::resume(ctx)` to reconstitute state
//! and continue from the saved point.
//!
//! ## Multi-tenant addressing — `ThreadKey`
//!
//! Every persistence operation is keyed by `(tenant_id, thread_id)`.
//! [`ThreadKey`] encodes that tuple as a single type so impls cannot
//! "forget" to scope a query — Invariant 11 holds at the type level
//! rather than relying on each backend to remember to add a `WHERE
//! tenant_id = ...` clause. `ThreadKey::from_ctx(ctx)` is the
//! canonical builder; it requires `ctx.thread_id()` to be set
//! (`ctx.tenant_id()` is always present per ADR-0017).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use entelix_core::error::Result;
use entelix_core::{TenantId, ThreadKey};

/// Stable identifier for a checkpoint. Backed by UUID v7 — time-ordered
/// and globally unique across processes.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CheckpointId(uuid::Uuid);

impl CheckpointId {
    /// Generate a fresh time-ordered id.
    pub fn new() -> Self {
        Self(uuid::Uuid::now_v7())
    }

    /// Reconstruct an id from a `uuid::Uuid` — used by persistence
    /// backends that read checkpoint rows out of storage.
    pub const fn from_uuid(uuid: uuid::Uuid) -> Self {
        Self(uuid)
    }

    /// Borrow the underlying UUID.
    pub const fn as_uuid(&self) -> &uuid::Uuid {
        &self.0
    }

    /// Render as a hyphenated string.
    pub fn to_hyphenated_string(&self) -> String {
        self.0.to_string()
    }
}

impl Default for CheckpointId {
    fn default() -> Self {
        Self::new()
    }
}

/// One snapshot of graph progress for a particular `(tenant_id,
/// thread_id)`. `next_node = None` indicates the graph terminated
/// cleanly (a finish point ran or a conditional edge routed to
/// `END`).
///
/// `#[non_exhaustive]` so post-1.0 additions (e.g. trace-context
/// propagation, schema-version stamping) ship as MINOR. Construct
/// via [`Checkpoint::new`]; attach the optional parent for
/// time-travel writes via [`Checkpoint::with_parent`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Checkpoint<S>
where
    S: Clone + Send + Sync + 'static,
{
    /// Unique identifier (UUID v7).
    pub id: CheckpointId,
    /// Tenant scope this checkpoint belongs to.
    pub tenant_id: TenantId,
    /// Conversation thread this checkpoint belongs to.
    pub thread_id: String,
    /// Optional parent — used by time-travel writes.
    pub parent_id: Option<CheckpointId>,
    /// Monotonic step counter within the thread.
    pub step: usize,
    /// State produced by the most recently executed node.
    pub state: S,
    /// Node the graph is poised to execute next, or `None` if it has
    /// terminated.
    pub next_node: Option<String>,
    /// When the checkpoint was written.
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl<S> Checkpoint<S>
where
    S: Clone + Send + Sync + 'static,
{
    /// Construct a fresh checkpoint addressed by `key`. Generates a
    /// new [`CheckpointId`] (UUID v7) and stamps `timestamp` with
    /// the current wall clock. `parent_id` defaults to `None`;
    /// chain [`Self::with_parent`] for time-travel writes.
    #[must_use]
    pub fn new(key: &ThreadKey, step: usize, state: S, next_node: Option<String>) -> Self {
        Self {
            id: CheckpointId::new(),
            tenant_id: key.tenant_id().clone(),
            thread_id: key.thread_id().to_owned(),
            parent_id: None,
            step,
            state,
            next_node,
            timestamp: chrono::Utc::now(),
        }
    }

    /// Attach a `parent_id` (time-travel branching). Chain after
    /// [`Self::new`].
    #[must_use]
    pub const fn with_parent(mut self, parent_id: CheckpointId) -> Self {
        self.parent_id = Some(parent_id);
        self
    }

    /// Reconstitute a checkpoint from explicit parts. Used by
    /// persistence backends rehydrating rows from storage — the
    /// caller already knows every field's value (id from the row's
    /// PK, timestamp from the column). Agent code reaches for
    /// [`Self::new`] instead.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        id: CheckpointId,
        key: &ThreadKey,
        parent_id: Option<CheckpointId>,
        step: usize,
        state: S,
        next_node: Option<String>,
        timestamp: chrono::DateTime<chrono::Utc>,
    ) -> Self {
        Self {
            id,
            tenant_id: key.tenant_id().to_owned(),
            thread_id: key.thread_id().to_owned(),
            parent_id,
            step,
            state,
            next_node,
            timestamp,
        }
    }

    /// Borrow the addressing tuple this checkpoint belongs to.
    #[must_use]
    pub fn key(&self) -> ThreadKey {
        ThreadKey::new(self.tenant_id.clone(), self.thread_id.clone())
    }
}

/// Persistent (or in-memory) store of `Checkpoint<S>`s addressed by
/// [`ThreadKey`].
///
/// Implementors must be `Send + Sync` so a single instance can serve
/// every concurrent invocation in a multi-pod deployment. The
/// `&ThreadKey` parameter on every read/write enforces tenant scope
/// at the type level — Invariant 11.
///
/// # `S: Drop` contract
///
/// Implementors may evict, replace, or reallocate stored values inside
/// internal locks. `S::drop` therefore **must not block** — no
/// `block_on`, no synchronous IO, no lock acquisition. Spawn a
/// detached task or use a non-blocking sink instead. See ADR-0006
/// §"Amendment 2026-04-30 — State drop semantics".
#[async_trait]
pub trait Checkpointer<S>: Send + Sync + 'static
where
    S: Clone + Send + Sync + 'static,
{
    /// Persist a checkpoint. The checkpoint's own
    /// `(tenant_id, thread_id)` fields define its addressing.
    async fn put(&self, checkpoint: Checkpoint<S>) -> Result<()>;

    /// Load the most recent checkpoint for `key`.
    async fn latest(&self, key: &ThreadKey) -> Result<Option<Checkpoint<S>>>;

    /// Look up a specific checkpoint by id within `key`'s scope.
    async fn by_id(&self, key: &ThreadKey, id: &CheckpointId) -> Result<Option<Checkpoint<S>>>;

    /// Return the thread's checkpoint history, most recent first.
    /// `limit` caps the result size (`usize::MAX` for "all").
    async fn history(&self, key: &ThreadKey, limit: usize) -> Result<Vec<Checkpoint<S>>>;

    /// Time-travel write: create a fresh checkpoint that branches off
    /// `parent_id`, replacing only the state. The new checkpoint
    /// inherits `next_node` from its parent and records `parent_id`
    /// so history renders branches correctly.
    ///
    /// Returns the new id. Returns `Error::InvalidRequest` if the
    /// parent does not exist for the supplied `key`.
    async fn update_state(
        &self,
        key: &ThreadKey,
        parent_id: &CheckpointId,
        new_state: S,
    ) -> Result<CheckpointId>;
}
