//! `RedisStore<V>` — [`Store<V>`] over Redis string keys. TTL is
//! native via `SET ... PX`; Redis lazily evicts expired keys, so
//! [`Store::evict_expired`] is a no-op (the default `Ok(0)`).

use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use entelix_core::{Error, ExecutionContext, Result};
use entelix_memory::{Namespace, PutOptions, Store};
use redis::aio::ConnectionManager;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::error::PersistenceError;

/// Redis-backed [`Store<V>`].
pub struct RedisStore<V> {
    manager: Arc<ConnectionManager>,
    _phantom: PhantomData<fn() -> V>,
}

impl<V> RedisStore<V>
where
    V: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    pub(crate) fn new(manager: Arc<ConnectionManager>) -> Self {
        Self {
            manager,
            _phantom: PhantomData,
        }
    }
}

fn item_key(ns: &Namespace, key: &str) -> String {
    format!("entelix:store:{}:{}:{}", ns.tenant_id(), ns.render(), key)
}

fn list_pattern(ns: &Namespace, prefix: Option<&str>) -> String {
    format!(
        "entelix:store:{}:{}:{}*",
        ns.tenant_id(),
        ns.render(),
        prefix.unwrap_or("")
    )
}

#[async_trait]
impl<V> Store<V> for RedisStore<V>
where
    V: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    async fn put_with_options(
        &self,
        ctx: &ExecutionContext,
        ns: &Namespace,
        key: &str,
        value: V,
        options: PutOptions,
    ) -> Result<()> {
        let mut conn = (*self.manager).clone();
        let body = serde_json::to_string(&value).map_err(Error::Serde)?;
        let item_key = item_key(ns, key);
        let fut = async {
            let mut cmd = redis::cmd("SET");
            cmd.arg(&item_key).arg(body);
            if let Some(ttl) = options.ttl {
                let ms = u64::try_from(ttl.as_millis()).unwrap_or(u64::MAX);
                cmd.arg("PX").arg(ms);
            }
            cmd.query_async::<()>(&mut conn).await
        };
        cancel_aware(ctx, fut).await?.map_err(backend_to_core)?;
        Ok(())
    }

    async fn get(&self, ctx: &ExecutionContext, ns: &Namespace, key: &str) -> Result<Option<V>> {
        let mut conn = (*self.manager).clone();
        let fut = async {
            redis::cmd("GET")
                .arg(item_key(ns, key))
                .query_async::<Option<String>>(&mut conn)
                .await
        };
        let raw = cancel_aware(ctx, fut).await?.map_err(backend_to_core)?;
        match raw {
            None => Ok(None),
            Some(s) => Ok(Some(serde_json::from_str(&s).map_err(Error::Serde)?)),
        }
    }

    async fn delete(&self, ctx: &ExecutionContext, ns: &Namespace, key: &str) -> Result<()> {
        let mut conn = (*self.manager).clone();
        let fut = async {
            redis::cmd("DEL")
                .arg(item_key(ns, key))
                .query_async::<()>(&mut conn)
                .await
        };
        cancel_aware(ctx, fut).await?.map_err(backend_to_core)?;
        Ok(())
    }

    async fn list(
        &self,
        ctx: &ExecutionContext,
        ns: &Namespace,
        prefix: Option<&str>,
    ) -> Result<Vec<String>> {
        let mut conn = (*self.manager).clone();
        let pattern = list_pattern(ns, prefix);
        let prefix_to_strip = format!("entelix:store:{}:{}:", ns.tenant_id(), ns.render());
        // SCAN avoids blocking the server on large keyspaces.
        let mut cursor: u64 = 0;
        let mut out = Vec::new();
        loop {
            let fut = async {
                redis::cmd("SCAN")
                    .arg(cursor)
                    .arg("MATCH")
                    .arg(&pattern)
                    .arg("COUNT")
                    .arg(200)
                    .query_async::<(u64, Vec<String>)>(&mut conn)
                    .await
            };
            let (next_cursor, batch) = cancel_aware(ctx, fut).await?.map_err(backend_to_core)?;
            for full_key in batch {
                if let Some(suffix) = full_key.strip_prefix(&prefix_to_strip) {
                    out.push(suffix.to_owned());
                }
            }
            if next_cursor == 0 {
                break;
            }
            cursor = next_cursor;
        }
        out.sort();
        Ok(out)
    }
}

/// Run a future under the caller's cancellation token. Returns
/// `Err(Error::Cancelled)` if the token fires before the future
/// resolves.
async fn cancel_aware<F, T, E>(ctx: &ExecutionContext, fut: F) -> Result<std::result::Result<T, E>>
where
    F: std::future::Future<Output = std::result::Result<T, E>>,
{
    let cancel = ctx.cancellation();
    tokio::select! {
        biased;
        () = cancel.cancelled() => Err(Error::Cancelled),
        out = fut => Ok(out),
    }
}

fn backend_to_core(e: redis::RedisError) -> Error {
    PersistenceError::Backend(e.to_string()).into()
}
