//! `RedisSessionLog` — [`SessionLog`] over a Redis list per
//! `(tenant, thread)`. Append uses `RPUSH`, ordinal is the new
//! `LLEN`. Archival sets a watermark scalar; reads honour it by
//! adjusting `LRANGE` start.

use std::sync::Arc;

use async_trait::async_trait;
use entelix_core::{Error, Result, TenantId, ThreadKey};
use entelix_session::{GraphEvent, SessionLog};
use redis::aio::ConnectionManager;

use crate::error::PersistenceError;

/// Redis-backed [`SessionLog`].
pub struct RedisSessionLog {
    manager: Arc<ConnectionManager>,
}

impl RedisSessionLog {
    pub(crate) fn new(manager: Arc<ConnectionManager>) -> Self {
        Self { manager }
    }
}

fn events_key(tenant_id: &TenantId, thread_id: &str) -> String {
    format!("entelix:session:{tenant_id}:{thread_id}:events")
}

fn watermark_key(tenant_id: &TenantId, thread_id: &str) -> String {
    format!("entelix:session:{tenant_id}:{thread_id}:watermark")
}

async fn current_watermark(
    conn: &mut ConnectionManager,
    tenant_id: &TenantId,
    thread_id: &str,
) -> std::result::Result<u64, redis::RedisError> {
    let raw: Option<String> = redis::cmd("GET")
        .arg(watermark_key(tenant_id, thread_id))
        .query_async(conn)
        .await?;
    Ok(raw.and_then(|s| s.parse::<u64>().ok()).unwrap_or(0))
}

#[async_trait]
impl SessionLog for RedisSessionLog {
    async fn append(&self, key: &ThreadKey, events: &[GraphEvent]) -> Result<u64> {
        let mut conn = (*self.manager).clone();
        let tenant_id = key.tenant_id();
        let thread_id = key.thread_id();
        if events.is_empty() {
            let len: u64 = redis::cmd("LLEN")
                .arg(events_key(tenant_id, thread_id))
                .query_async(&mut conn)
                .await
                .map_err(backend_to_core)?;
            let watermark = current_watermark(&mut conn, tenant_id, thread_id)
                .await
                .map_err(backend_to_core)?;
            return Ok(watermark.saturating_add(len));
        }
        let mut payloads: Vec<String> = Vec::with_capacity(events.len());
        for event in events {
            payloads.push(serde_json::to_string(event).map_err(Error::Serde)?);
        }
        let new_len: u64 = redis::cmd("RPUSH")
            .arg(events_key(tenant_id, thread_id))
            .arg(&payloads)
            .query_async(&mut conn)
            .await
            .map_err(backend_to_core)?;
        let watermark = current_watermark(&mut conn, tenant_id, thread_id)
            .await
            .map_err(backend_to_core)?;
        Ok(watermark.saturating_add(new_len))
    }

    async fn load_since(&self, key: &ThreadKey, cursor: u64) -> Result<Vec<GraphEvent>> {
        let mut conn = (*self.manager).clone();
        let tenant_id = key.tenant_id();
        let thread_id = key.thread_id();
        let watermark = current_watermark(&mut conn, tenant_id, thread_id)
            .await
            .map_err(backend_to_core)?;
        // Cursor is the absolute ordinal; live events begin at offset
        // `cursor - watermark` within the list (post-archival).
        let live_start = cursor.saturating_sub(watermark);
        let live_start_isize = isize::try_from(live_start).unwrap_or(isize::MAX);
        let raws: Vec<String> = redis::cmd("LRANGE")
            .arg(events_key(tenant_id, thread_id))
            .arg(live_start_isize)
            .arg(-1)
            .query_async(&mut conn)
            .await
            .map_err(backend_to_core)?;
        raws.into_iter()
            .map(|s| serde_json::from_str::<GraphEvent>(&s).map_err(Error::Serde))
            .collect()
    }

    async fn archive_before(&self, key: &ThreadKey, watermark: u64) -> Result<usize> {
        let mut conn = (*self.manager).clone();
        let tenant_id = key.tenant_id();
        let thread_id = key.thread_id();
        let prior = current_watermark(&mut conn, tenant_id, thread_id)
            .await
            .map_err(backend_to_core)?;
        if watermark <= prior {
            return Ok(0);
        }
        let trim_count_u64 = watermark.saturating_sub(prior);
        let trim_count_isize = isize::try_from(trim_count_u64).unwrap_or(isize::MAX);
        // LTRIM keeps elements in [start, stop]; we drop the first
        // `trim_count_isize` entries.
        let _: () = redis::cmd("LTRIM")
            .arg(events_key(tenant_id, thread_id))
            .arg(trim_count_isize)
            .arg(-1)
            .query_async(&mut conn)
            .await
            .map_err(backend_to_core)?;
        let _: () = redis::cmd("SET")
            .arg(watermark_key(tenant_id, thread_id))
            .arg(watermark.to_string())
            .query_async(&mut conn)
            .await
            .map_err(backend_to_core)?;
        Ok(usize::try_from(trim_count_u64).unwrap_or(usize::MAX))
    }
}

fn backend_to_core(e: redis::RedisError) -> Error {
    PersistenceError::Backend(e.to_string()).into()
}
