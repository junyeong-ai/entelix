//! `PostgresStore<V>` — `entelix_memory::Store<V>` over the
//! `memory_items` table. Tenant scope is sourced from `Namespace`
//! (invariant 11 / F2). TTL'd rows expire via the sweeper
//! ([`Store::evict_expired`]) which also surfaces a cleanup count
//! for operator dashboards.

use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use entelix_core::{Error, ExecutionContext, Result};
use entelix_memory::{Namespace, NamespacePrefix, PutOptions, Store};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use sqlx::postgres::PgPool;

use crate::error::PersistenceError;
use crate::postgres::tenant::set_tenant_session;

/// Postgres-backed [`Store<V>`].
pub struct PostgresStore<V> {
    pool: Arc<PgPool>,
    _phantom: PhantomData<fn() -> V>,
}

impl<V> PostgresStore<V>
where
    V: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    pub(crate) fn new(pool: Arc<PgPool>) -> Self {
        Self {
            pool,
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<V> Store<V> for PostgresStore<V>
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
        let value_json = serde_json::to_value(&value).map_err(into_core_serde)?;
        let expires_at = options.ttl.and_then(|d| {
            chrono::Duration::from_std(d)
                .ok()
                .map(|cd| chrono::Utc::now() + cd)
        });
        let fut = async {
            let mut tx = self.pool.begin().await.map_err(backend_to_core)?;
            set_tenant_session(&mut *tx, ns.tenant_id()).await?;
            sqlx::query(
                r"
                INSERT INTO memory_items (tenant_id, namespace, key, value, ts, expires_at)
                VALUES ($1, $2, $3, $4, now(), $5)
                ON CONFLICT (tenant_id, namespace, key)
                DO UPDATE SET
                    value = EXCLUDED.value,
                    ts = EXCLUDED.ts,
                    expires_at = EXCLUDED.expires_at
                ",
            )
            .bind(ns.tenant_id())
            .bind(ns.render())
            .bind(key)
            .bind(&value_json)
            .bind(expires_at)
            .execute(&mut *tx)
            .await
            .map_err(backend_to_core)?;
            tx.commit().await.map_err(backend_to_core)?;
            Ok::<(), Error>(())
        };
        cancel_aware(ctx, fut).await??;
        Ok(())
    }

    async fn get(&self, ctx: &ExecutionContext, ns: &Namespace, key: &str) -> Result<Option<V>> {
        let fut = async {
            let mut tx = self.pool.begin().await.map_err(backend_to_core)?;
            set_tenant_session(&mut *tx, ns.tenant_id()).await?;
            let row: Option<(Value,)> = sqlx::query_as(
                r"
                SELECT value FROM memory_items
                WHERE tenant_id = $1 AND namespace = $2 AND key = $3
                  AND (expires_at IS NULL OR expires_at > now())
                ",
            )
            .bind(ns.tenant_id())
            .bind(ns.render())
            .bind(key)
            .fetch_optional(&mut *tx)
            .await
            .map_err(backend_to_core)?;
            tx.commit().await.map_err(backend_to_core)?;
            Ok::<_, Error>(row)
        };
        let row = cancel_aware(ctx, fut).await??;
        match row {
            None => Ok(None),
            Some((value,)) => {
                let parsed: V = serde_json::from_value(value).map_err(into_core_serde)?;
                Ok(Some(parsed))
            }
        }
    }

    async fn delete(&self, ctx: &ExecutionContext, ns: &Namespace, key: &str) -> Result<()> {
        let fut = async {
            let mut tx = self.pool.begin().await.map_err(backend_to_core)?;
            set_tenant_session(&mut *tx, ns.tenant_id()).await?;
            sqlx::query(
                r"
                DELETE FROM memory_items
                WHERE tenant_id = $1 AND namespace = $2 AND key = $3
                ",
            )
            .bind(ns.tenant_id())
            .bind(ns.render())
            .bind(key)
            .execute(&mut *tx)
            .await
            .map_err(backend_to_core)?;
            tx.commit().await.map_err(backend_to_core)?;
            Ok::<(), Error>(())
        };
        cancel_aware(ctx, fut).await??;
        Ok(())
    }

    async fn list(
        &self,
        ctx: &ExecutionContext,
        ns: &Namespace,
        prefix: Option<&str>,
    ) -> Result<Vec<String>> {
        let pattern = format!("{}%", prefix.unwrap_or(""));
        let fut = async {
            let mut tx = self.pool.begin().await.map_err(backend_to_core)?;
            set_tenant_session(&mut *tx, ns.tenant_id()).await?;
            let rows: Vec<(String,)> = sqlx::query_as(
                r"
                SELECT key FROM memory_items
                WHERE tenant_id = $1 AND namespace = $2 AND key LIKE $3
                  AND (expires_at IS NULL OR expires_at > now())
                ORDER BY key ASC
                ",
            )
            .bind(ns.tenant_id())
            .bind(ns.render())
            .bind(pattern)
            .fetch_all(&mut *tx)
            .await
            .map_err(backend_to_core)?;
            tx.commit().await.map_err(backend_to_core)?;
            Ok::<_, Error>(rows)
        };
        let rows = cancel_aware(ctx, fut).await??;
        Ok(rows.into_iter().map(|(k,)| k).collect())
    }

    async fn list_namespaces(
        &self,
        ctx: &ExecutionContext,
        prefix: &NamespacePrefix,
    ) -> Result<Vec<Namespace>> {
        // Render the prefix as Namespace::render does so the LIKE
        // pattern matches the stored key. Append `:%` to allow
        // strict-subscope matches plus the prefix itself.
        let mut tmp = Namespace::new(prefix.tenant_id());
        for s in prefix.scope() {
            tmp = tmp.with_scope(s.clone());
        }
        let prefix_render = tmp.render();
        let exact_pattern = prefix_render.clone();
        let nested_pattern = format!("{prefix_render}:%");
        let fut = async {
            let mut tx = self.pool.begin().await.map_err(backend_to_core)?;
            set_tenant_session(&mut *tx, prefix.tenant_id()).await?;
            let rows: Vec<(String,)> = sqlx::query_as(
                r"
                SELECT DISTINCT namespace FROM memory_items
                WHERE tenant_id = $1
                  AND (namespace = $2 OR namespace LIKE $3)
                  AND (expires_at IS NULL OR expires_at > now())
                ORDER BY namespace ASC
                ",
            )
            .bind(prefix.tenant_id())
            .bind(exact_pattern)
            .bind(nested_pattern)
            .fetch_all(&mut *tx)
            .await
            .map_err(backend_to_core)?;
            tx.commit().await.map_err(backend_to_core)?;
            Ok::<_, Error>(rows)
        };
        let rows = cancel_aware(ctx, fut).await??;
        // `Namespace::parse` is the inverse of `Namespace::render`,
        // so each stored row's namespace string round-trips back
        // into the original typed scope (tenant boundary + nested
        // segments preserved). The trait contract — "every distinct
        // Namespace under prefix" — is honoured structurally rather
        // than approximated with a prefix-shape clone.
        rows.into_iter()
            .map(|(rendered,)| Namespace::parse(&rendered))
            .collect()
    }

    async fn evict_expired(&self, ctx: &ExecutionContext) -> Result<usize> {
        // Cross-tenant maintenance — the per-query `SET LOCAL
        // entelix.tenant_id` pattern other methods use cannot apply
        // here (no single tenant scope to set). Under the RLS
        // policy the SDK installs (invariant #11 defense in depth),
        // this query returns 0 rows when run by a role *subject* to
        // the policy — every row's `tenant_id` differs from the
        // unset `current_setting('entelix.tenant_id', true)`.
        // Operators run TTL sweepers from a separate database role
        // configured with the `BYPASSRLS` attribute, scheduled
        // outside the per-request application path.
        let fut = sqlx::query(
            r"
            DELETE FROM memory_items
            WHERE expires_at IS NOT NULL AND expires_at <= now()
            ",
        )
        .execute(&*self.pool);
        let result = cancel_aware(ctx, fut).await?.map_err(backend_to_core)?;
        Ok(usize::try_from(result.rows_affected()).unwrap_or(usize::MAX))
    }
}

/// Run a future under the caller's cancellation token. Returns
/// `Err(Error::Cancelled)` if the token fires before the future
/// resolves, otherwise yields the inner future's result.
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

fn backend_to_core(e: sqlx::Error) -> Error {
    PersistenceError::Backend(e.to_string()).into()
}

fn into_core_serde(e: serde_json::Error) -> Error {
    Error::Serde(e)
}
