//! `RedisCheckpointer<S>` — [`Checkpointer<S>`] over Redis sorted
//! sets keyed by `step`. A companion HASH provides O(1) lookup by
//! checkpoint id. Keys partition by `(tenant_id, thread_id)` per
//! Invariant 11 — cross-tenant reads are not constructible from
//! this surface.

use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use entelix_core::ThreadKey;
use entelix_core::{Error, Result};
use entelix_graph::{Checkpoint, CheckpointId, Checkpointer};
use redis::aio::ConnectionManager;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::error::PersistenceError;
use crate::schema_version::SessionSchemaVersion;

/// Redis-backed [`Checkpointer<S>`].
pub struct RedisCheckpointer<S> {
    manager: Arc<ConnectionManager>,
    _phantom: PhantomData<fn() -> S>,
}

impl<S> RedisCheckpointer<S>
where
    S: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    pub(crate) fn new(manager: Arc<ConnectionManager>) -> Self {
        Self {
            manager,
            _phantom: PhantomData,
        }
    }
}

fn zset_key(key: &ThreadKey) -> String {
    format!("entelix:cp:{}:{}:bystep", key.tenant_id(), key.thread_id())
}

fn hash_key(key: &ThreadKey) -> String {
    format!("entelix:cp:{}:{}:byid", key.tenant_id(), key.thread_id())
}

#[async_trait]
impl<S> Checkpointer<S> for RedisCheckpointer<S>
where
    S: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    async fn put(&self, checkpoint: Checkpoint<S>) -> Result<()> {
        let key = checkpoint.key();
        let envelope = wrap_envelope(&checkpoint).map_err(into_core)?;
        let id_str = checkpoint.id.to_hyphenated_string();
        let mut conn = (*self.manager).clone();
        let step_score = i64::try_from(checkpoint.step).unwrap_or(i64::MAX) as f64;
        // Two-step write — Redis pipeline keeps the round-trip minimal.
        redis::pipe()
            .atomic()
            .zadd(zset_key(&key), &id_str, step_score)
            .hset(hash_key(&key), &id_str, envelope.to_string())
            .query_async::<()>(&mut conn)
            .await
            .map_err(backend_to_core)?;
        Ok(())
    }

    async fn latest(&self, key: &ThreadKey) -> Result<Option<Checkpoint<S>>> {
        let mut conn = (*self.manager).clone();
        let ids: Vec<String> = redis::cmd("ZREVRANGE")
            .arg(zset_key(key))
            .arg(0)
            .arg(0)
            .query_async(&mut conn)
            .await
            .map_err(backend_to_core)?;
        let Some(id) = ids.into_iter().next() else {
            return Ok(None);
        };
        load_by_id::<S>(&mut conn, key, &id).await
    }

    async fn by_id(&self, key: &ThreadKey, id: &CheckpointId) -> Result<Option<Checkpoint<S>>> {
        let mut conn = (*self.manager).clone();
        load_by_id::<S>(&mut conn, key, &id.to_hyphenated_string()).await
    }

    async fn history(&self, key: &ThreadKey, limit: usize) -> Result<Vec<Checkpoint<S>>> {
        let mut conn = (*self.manager).clone();
        let stop = if limit == 0 || limit == usize::MAX {
            -1isize
        } else {
            isize::try_from(limit.saturating_sub(1)).unwrap_or(isize::MAX)
        };
        let ids: Vec<String> = redis::cmd("ZREVRANGE")
            .arg(zset_key(key))
            .arg(0)
            .arg(stop)
            .query_async(&mut conn)
            .await
            .map_err(backend_to_core)?;
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(cp) = load_by_id::<S>(&mut conn, key, &id).await? {
                out.push(cp);
            }
        }
        Ok(out)
    }

    async fn update_state(
        &self,
        key: &ThreadKey,
        parent_id: &CheckpointId,
        new_state: S,
    ) -> Result<CheckpointId> {
        let parent = self.by_id(key, parent_id).await?.ok_or_else(|| {
            Error::invalid_request(format!(
                "RedisCheckpointer::update_state: parent {} not found in tenant '{}' thread '{}'",
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

async fn load_by_id<S>(
    conn: &mut ConnectionManager,
    key: &ThreadKey,
    id: &str,
) -> Result<Option<Checkpoint<S>>>
where
    S: Clone + Send + Sync + DeserializeOwned + 'static,
{
    let raw: Option<String> = redis::cmd("HGET")
        .arg(hash_key(key))
        .arg(id)
        .query_async(conn)
        .await
        .map_err(backend_to_core)?;
    let Some(raw) = raw else { return Ok(None) };
    let value: Value = serde_json::from_str(&raw).map_err(Error::Serde)?;
    let cp = unwrap_envelope::<S>(&value).map_err(into_core)?;
    Ok(Some(cp))
}

fn wrap_envelope<S>(cp: &Checkpoint<S>) -> std::result::Result<Value, PersistenceError>
where
    S: Clone + Send + Sync + Serialize + 'static,
{
    let body = serde_json::json!({
        "id": cp.id,
        "tenant_id": cp.tenant_id,
        "thread_id": cp.thread_id,
        "parent_id": cp.parent_id,
        "step": cp.step,
        "state": serde_json::to_value(&cp.state)?,
        "next_node": cp.next_node,
        "timestamp": cp.timestamp,
    });
    Ok(serde_json::json!({
        "schema_version": SessionSchemaVersion::CURRENT,
        "body": body,
    }))
}

fn unwrap_envelope<S>(value: &Value) -> std::result::Result<Checkpoint<S>, PersistenceError>
where
    S: Clone + Send + Sync + DeserializeOwned + 'static,
{
    let version = value
        .get("schema_version")
        .and_then(|v| v.as_u64())
        .map(|n| u32::try_from(n).unwrap_or(u32::MAX))
        .map(SessionSchemaVersion)
        .ok_or_else(|| {
            PersistenceError::Backend("checkpoint envelope lacks schema_version".into())
        })?;
    version.validate()?;
    let body = value
        .get("body")
        .ok_or_else(|| PersistenceError::Backend("checkpoint envelope lacks body".into()))?;
    let id: CheckpointId = serde_json::from_value(
        body.get("id")
            .cloned()
            .ok_or_else(|| PersistenceError::Backend("checkpoint missing id".into()))?,
    )?;
    let tenant_id: String = body
        .get("tenant_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| PersistenceError::Backend("checkpoint missing tenant_id".into()))?
        .to_owned();
    let thread_id: String = body
        .get("thread_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| PersistenceError::Backend("checkpoint missing thread_id".into()))?
        .to_owned();
    let parent_id = match body.get("parent_id") {
        Some(Value::Null) | None => None,
        Some(v) => Some(serde_json::from_value::<CheckpointId>(v.clone())?),
    };
    let step = body
        .get("step")
        .and_then(|v| v.as_u64())
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| PersistenceError::Backend("checkpoint missing step".into()))?;
    let state: S = body
        .get("state")
        .map(|s| serde_json::from_value(s.clone()))
        .ok_or_else(|| PersistenceError::Backend("checkpoint missing state".into()))??;
    let next_node = body
        .get("next_node")
        .and_then(|v| v.as_str())
        .map(ToOwned::to_owned);
    let timestamp = body
        .get("timestamp")
        .map(|v| serde_json::from_value::<chrono::DateTime<chrono::Utc>>(v.clone()))
        .ok_or_else(|| PersistenceError::Backend("checkpoint missing timestamp".into()))??;
    let key = ThreadKey::new(tenant_id, thread_id);
    Ok(Checkpoint::from_parts(
        id, &key, parent_id, step, state, next_node, timestamp,
    ))
}

fn backend_to_core(e: redis::RedisError) -> Error {
    PersistenceError::Backend(e.to_string()).into()
}

fn into_core(e: PersistenceError) -> Error {
    e.into()
}
