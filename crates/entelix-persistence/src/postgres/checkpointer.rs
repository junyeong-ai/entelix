//! `PostgresCheckpointer<S>` — `entelix_graph::Checkpointer<S>` over
//! the `checkpoints` table. Every read/write partitions by
//! `(tenant_id, thread_id)` per Invariant 11 — the trait surface
//! supplies a `&ThreadKey` so cross-tenant reads are not even
//! constructible from this backend.

use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use entelix_core::ThreadKey;
use entelix_core::{Error, Result};
use entelix_graph::{Checkpoint, CheckpointId, Checkpointer};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use sqlx::postgres::PgPool;
use uuid::Uuid;

use crate::error::PersistenceError;
use crate::postgres::tenant::set_tenant_session;
use crate::schema_version::SessionSchemaVersion;

const STATE_KEY: &str = "state";
const SCHEMA_KEY: &str = "schema_version";

/// Postgres-backed [`Checkpointer<S>`].
///
/// State payloads are stamped with [`SessionSchemaVersion`] before
/// serialisation, so a downgrade that can't read the format fails
/// loudly instead of silently corrupting the row.
pub struct PostgresCheckpointer<S> {
    pool: Arc<PgPool>,
    _phantom: PhantomData<fn() -> S>,
}

impl<S> PostgresCheckpointer<S>
where
    S: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    pub(crate) fn new(pool: Arc<PgPool>) -> Self {
        Self {
            pool,
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<S> Checkpointer<S> for PostgresCheckpointer<S>
where
    S: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    async fn put(&self, checkpoint: Checkpoint<S>) -> Result<()> {
        let envelope = wrap_state(&checkpoint.state).map_err(into_core)?;
        let parent = checkpoint.parent_id.as_ref().map(|p| *p.as_uuid());
        let step_i64 = i64::try_from(checkpoint.step).unwrap_or(i64::MAX);

        let mut tx = self.pool.begin().await.map_err(backend_to_core)?;
        set_tenant_session(&mut *tx, &checkpoint.tenant_id).await?;
        sqlx::query(
            r"
            INSERT INTO checkpoints
                (tenant_id, thread_id, id, parent_id, step, state, next_node, ts)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ",
        )
        .bind(checkpoint.tenant_id.as_str())
        .bind(&checkpoint.thread_id)
        .bind(checkpoint.id.as_uuid())
        .bind(parent)
        .bind(step_i64)
        .bind(&envelope)
        .bind(checkpoint.next_node.as_deref())
        .bind(checkpoint.timestamp)
        .execute(&mut *tx)
        .await
        .map_err(backend_to_core)?;
        tx.commit().await.map_err(backend_to_core)?;
        Ok(())
    }

    async fn latest(&self, key: &ThreadKey) -> Result<Option<Checkpoint<S>>> {
        let mut tx = self.pool.begin().await.map_err(backend_to_core)?;
        set_tenant_session(&mut *tx, key.tenant_id()).await?;
        let row: Option<CheckpointRow> = sqlx::query_as::<_, CheckpointRow>(
            r"
            SELECT tenant_id, thread_id, id, parent_id, step, state, next_node, ts
            FROM checkpoints
            WHERE tenant_id = $1 AND thread_id = $2
            ORDER BY step DESC, ts DESC
            LIMIT 1
            ",
        )
        .bind(key.tenant_id().as_str())
        .bind(key.thread_id())
        .fetch_optional(&mut *tx)
        .await
        .map_err(backend_to_core)?;
        tx.commit().await.map_err(backend_to_core)?;
        row.map(|r| r.try_into_checkpoint::<S>())
            .transpose()
            .map_err(into_core)
    }

    async fn by_id(&self, key: &ThreadKey, id: &CheckpointId) -> Result<Option<Checkpoint<S>>> {
        let mut tx = self.pool.begin().await.map_err(backend_to_core)?;
        set_tenant_session(&mut *tx, key.tenant_id()).await?;
        let row: Option<CheckpointRow> = sqlx::query_as::<_, CheckpointRow>(
            r"
            SELECT tenant_id, thread_id, id, parent_id, step, state, next_node, ts
            FROM checkpoints
            WHERE tenant_id = $1 AND thread_id = $2 AND id = $3
            ",
        )
        .bind(key.tenant_id().as_str())
        .bind(key.thread_id())
        .bind(id.as_uuid())
        .fetch_optional(&mut *tx)
        .await
        .map_err(backend_to_core)?;
        tx.commit().await.map_err(backend_to_core)?;
        row.map(|r| r.try_into_checkpoint::<S>())
            .transpose()
            .map_err(into_core)
    }

    async fn history(&self, key: &ThreadKey, limit: usize) -> Result<Vec<Checkpoint<S>>> {
        let limit_i64 = i64::try_from(limit).unwrap_or(i64::MAX);
        let mut tx = self.pool.begin().await.map_err(backend_to_core)?;
        set_tenant_session(&mut *tx, key.tenant_id()).await?;
        let rows: Vec<CheckpointRow> = sqlx::query_as::<_, CheckpointRow>(
            r"
            SELECT tenant_id, thread_id, id, parent_id, step, state, next_node, ts
            FROM checkpoints
            WHERE tenant_id = $1 AND thread_id = $2
            ORDER BY step DESC, ts DESC
            LIMIT $3
            ",
        )
        .bind(key.tenant_id().as_str())
        .bind(key.thread_id())
        .bind(limit_i64)
        .fetch_all(&mut *tx)
        .await
        .map_err(backend_to_core)?;
        tx.commit().await.map_err(backend_to_core)?;
        rows.into_iter()
            .map(CheckpointRow::try_into_checkpoint::<S>)
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(into_core)
    }

    async fn update_state(
        &self,
        key: &ThreadKey,
        parent_id: &CheckpointId,
        new_state: S,
    ) -> Result<CheckpointId> {
        let parent = self.by_id(key, parent_id).await?.ok_or_else(|| {
            Error::invalid_request(format!(
                "PostgresCheckpointer::update_state: parent {} not found in tenant '{}' thread '{}'",
                parent_id.to_hyphenated_string(),
                key.tenant_id(),
                key.thread_id()
            ))
        })?;
        let new_step = parent.step.saturating_add(1);
        let new_checkpoint = Checkpoint::new(key, new_step, new_state, parent.next_node)
            .with_parent(parent_id.clone());
        let new_id = new_checkpoint.id.clone();
        self.put(new_checkpoint).await?;
        Ok(new_id)
    }
}

#[derive(sqlx::FromRow)]
struct CheckpointRow {
    tenant_id: String,
    thread_id: String,
    id: Uuid,
    parent_id: Option<Uuid>,
    step: i64,
    state: Value,
    next_node: Option<String>,
    ts: DateTime<Utc>,
}

impl CheckpointRow {
    fn try_into_checkpoint<S>(self) -> std::result::Result<Checkpoint<S>, PersistenceError>
    where
        S: Clone + Send + Sync + DeserializeOwned + 'static,
    {
        let state = unwrap_state::<S>(&self.state)?;
        // Persistence-layer row hydration runs the validating
        // `TenantId::try_from`; an empty `tenant_id` column (which
        // would otherwise produce a tenantless `Checkpoint` whose
        // RLS policy comparison silently mis-routes) surfaces as
        // `Error::InvalidRequest` rather than a constructed value.
        let tenant = entelix_core::TenantId::try_from(self.tenant_id)
            .map_err(|e| PersistenceError::Backend(format!("invalid persisted tenant_id: {e}")))?;
        let key = ThreadKey::new(tenant, self.thread_id);
        Ok(Checkpoint::from_parts(
            CheckpointId::from_uuid(self.id),
            &key,
            self.parent_id.map(CheckpointId::from_uuid),
            usize::try_from(self.step).unwrap_or(0),
            state,
            self.next_node,
            self.ts,
        ))
    }
}

fn wrap_state<S: Serialize>(state: &S) -> std::result::Result<Value, PersistenceError> {
    let body = serde_json::to_value(state)?;
    Ok(serde_json::json!({
        SCHEMA_KEY: SessionSchemaVersion::CURRENT,
        STATE_KEY: body,
    }))
}

fn unwrap_state<S: DeserializeOwned>(value: &Value) -> std::result::Result<S, PersistenceError> {
    let version = value
        .get(SCHEMA_KEY)
        .and_then(|v| v.as_u64())
        .map(|n| u32::try_from(n).unwrap_or(u32::MAX))
        .map(SessionSchemaVersion)
        .ok_or_else(|| {
            PersistenceError::Backend("checkpoint payload lacks schema_version".into())
        })?;
    version.validate()?;
    let body = value
        .get(STATE_KEY)
        .ok_or_else(|| PersistenceError::Backend("checkpoint payload lacks state".into()))?;
    Ok(serde_json::from_value(body.clone())?)
}

fn backend_to_core(e: sqlx::Error) -> Error {
    PersistenceError::Backend(e.to_string()).into()
}

fn into_core(e: PersistenceError) -> Error {
    e.into()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Persistence-layer row hydration must run the `TenantId`
    /// validator on the persisted column. A row whose `tenant_id`
    /// column is empty (whether from a misconfigured admin script
    /// or a corrupted backup) cannot construct a tenantless
    /// `Checkpoint` whose RLS-filter comparison would then run
    /// against `''` and silently widen the result set
    /// (invariant 11 / ADR-0074).
    #[test]
    fn try_into_checkpoint_rejects_empty_persisted_tenant_id() {
        let row = CheckpointRow {
            tenant_id: String::new(),
            thread_id: "th-1".to_owned(),
            id: Uuid::new_v4(),
            parent_id: None,
            step: 0,
            state: serde_json::json!({
                SCHEMA_KEY: SessionSchemaVersion::CURRENT,
                STATE_KEY: 42,
            }),
            next_node: None,
            ts: chrono::Utc::now(),
        };
        let err = row.try_into_checkpoint::<i32>().unwrap_err();
        assert!(
            matches!(err, PersistenceError::Backend(ref m) if m.contains("invalid persisted tenant_id")),
            "expected Backend(\"invalid persisted tenant_id …\"), got {err:?}"
        );
    }
}
