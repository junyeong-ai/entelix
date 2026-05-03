//! `PostgresSessionLog` — `entelix_session::SessionLog` over the
//! `session_events` table.

use std::sync::Arc;

use async_trait::async_trait;
use entelix_core::{Error, Result, ThreadKey};
use entelix_session::{GraphEvent, SessionLog};
use serde_json::Value;
use sqlx::postgres::PgPool;

use crate::error::PersistenceError;
use crate::postgres::tenant::set_tenant_session;

/// Postgres-backed [`SessionLog`].
pub struct PostgresSessionLog {
    pool: Arc<PgPool>,
}

impl PostgresSessionLog {
    pub(crate) fn new(pool: Arc<PgPool>) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl SessionLog for PostgresSessionLog {
    async fn append(&self, key: &ThreadKey, events: &[GraphEvent]) -> Result<u64> {
        let tenant_id = key.tenant_id();
        let thread_id = key.thread_id();
        if events.is_empty() {
            // Return current head ordinal without writing — still
            // wrap in a tx so the RLS policy sees the tenant scope.
            let mut tx = self.pool.begin().await.map_err(backend_to_core)?;
            set_tenant_session(&mut *tx, tenant_id).await?;
            let row: Option<(i64,)> = sqlx::query_as(
                r"
                SELECT MAX(seq) FROM session_events
                WHERE tenant_id = $1 AND thread_id = $2
                ",
            )
            .bind(tenant_id)
            .bind(thread_id)
            .fetch_optional(&mut *tx)
            .await
            .map_err(backend_to_core)?;
            tx.commit().await.map_err(backend_to_core)?;
            return Ok(row.and_then(|(s,)| u64::try_from(s).ok()).unwrap_or(0));
        }

        let mut tx = self.pool.begin().await.map_err(backend_to_core)?;
        // Stamp the tenant scope before any RLS-protected query in
        // the transaction. The advisory lock below operates outside
        // RLS (it's a session-level lock function, not a table read).
        set_tenant_session(&mut *tx, tenant_id).await?;
        // Lock the per-thread row range so concurrent appends don't
        // interleave ordinals. Uses a thread-scoped advisory lock —
        // shared with `with_session_lock` so they cooperate.
        let advisory = crate::AdvisoryKey::for_session(tenant_id, thread_id);
        let (high, low) = advisory.halves();
        sqlx::query("SELECT pg_advisory_xact_lock($1, $2)")
            .bind(high)
            .bind(low)
            .execute(&mut *tx)
            .await
            .map_err(backend_to_core)?;

        let head: Option<(i64,)> = sqlx::query_as(
            r"
            SELECT MAX(seq) FROM session_events
            WHERE tenant_id = $1 AND thread_id = $2
            ",
        )
        .bind(tenant_id)
        .bind(thread_id)
        .fetch_optional(&mut *tx)
        .await
        .map_err(backend_to_core)?;
        let mut next_seq = head.and_then(|(s,)| s.try_into().ok()).unwrap_or(0u64);

        for event in events {
            next_seq = next_seq.saturating_add(1);
            let payload = serde_json::to_value(event).map_err(Error::Serde)?;
            let seq_i64 = i64::try_from(next_seq).unwrap_or(i64::MAX);
            sqlx::query(
                r"
                INSERT INTO session_events
                    (tenant_id, thread_id, seq, event, ts)
                VALUES ($1, $2, $3, $4, now())
                ",
            )
            .bind(tenant_id)
            .bind(thread_id)
            .bind(seq_i64)
            .bind(&payload)
            .execute(&mut *tx)
            .await
            .map_err(backend_to_core)?;
        }
        tx.commit().await.map_err(backend_to_core)?;
        Ok(next_seq)
    }

    async fn load_since(&self, key: &ThreadKey, cursor: u64) -> Result<Vec<GraphEvent>> {
        let cursor_i64 = i64::try_from(cursor).unwrap_or(i64::MAX);
        let mut tx = self.pool.begin().await.map_err(backend_to_core)?;
        set_tenant_session(&mut *tx, key.tenant_id()).await?;
        let rows: Vec<(Value,)> = sqlx::query_as(
            r"
            SELECT event FROM session_events
            WHERE tenant_id = $1 AND thread_id = $2 AND seq > $3
            ORDER BY seq ASC
            ",
        )
        .bind(key.tenant_id())
        .bind(key.thread_id())
        .bind(cursor_i64)
        .fetch_all(&mut *tx)
        .await
        .map_err(backend_to_core)?;
        tx.commit().await.map_err(backend_to_core)?;
        rows.into_iter()
            .map(|(v,)| serde_json::from_value::<GraphEvent>(v).map_err(Error::Serde))
            .collect()
    }

    async fn archive_before(&self, key: &ThreadKey, watermark: u64) -> Result<usize> {
        let watermark_i64 = i64::try_from(watermark).unwrap_or(i64::MAX);
        let mut tx = self.pool.begin().await.map_err(backend_to_core)?;
        set_tenant_session(&mut *tx, key.tenant_id()).await?;
        let result = sqlx::query(
            r"
            DELETE FROM session_events
            WHERE tenant_id = $1 AND thread_id = $2 AND seq <= $3
            ",
        )
        .bind(key.tenant_id())
        .bind(key.thread_id())
        .bind(watermark_i64)
        .execute(&mut *tx)
        .await
        .map_err(backend_to_core)?;
        tx.commit().await.map_err(backend_to_core)?;
        Ok(usize::try_from(result.rows_affected()).unwrap_or(usize::MAX))
    }
}

fn backend_to_core(e: sqlx::Error) -> Error {
    PersistenceError::Backend(e.to_string()).into()
}
