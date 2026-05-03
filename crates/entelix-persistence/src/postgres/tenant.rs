//! Tenant-scoped session-variable helper for RLS-enforced
//! persistence tables (invariant #11 defense in depth).
//!
//! Postgres row-level security policies on `memory_items`,
//! `session_events`, and `checkpoints` filter rows by
//! `current_setting('entelix.tenant_id', true)`. Without the
//! variable set, the policy treats every row as
//! `tenant_id = NULL` (unknown / false) — no row is visible, no
//! row may be inserted. The SDK stamps the variable per
//! transaction before issuing tenant-scoped queries.
//!
//! ## Usage shape
//!
//! Each tenant-scoped query method opens a transaction, calls
//! [`set_tenant_session`], runs its query, and commits. The SET
//! LOCAL semantics of `set_config(name, value, true)` scope the
//! variable to the enclosing transaction — pool connections that
//! return to the pool carry no leftover variable state.
//!
//! ```ignore
//! let mut tx = pool.begin().await?;
//! set_tenant_session(&mut tx, ns.tenant_id()).await?;
//! sqlx::query("INSERT INTO memory_items …")
//!     .execute(&mut *tx)
//!     .await?;
//! tx.commit().await?;
//! ```
//!
//! ## Cross-tenant maintenance operations
//!
//! Operations that legitimately span tenants — typically
//! [`entelix_memory::Store::evict_expired`] TTL sweepers — cannot
//! work through the SDK's RLS-enforced role. Operators run those
//! sweepers from a separate database role configured with
//! `BYPASSRLS`, scheduled outside the per-request application path.

use entelix_core::{Error, Result};
use sqlx::Executor;
use sqlx::postgres::Postgres;

use crate::error::PersistenceError;

/// Stamp the current transaction's `entelix.tenant_id` session
/// variable. The third argument to `set_config` is `is_local =
/// true`, scoping the assignment to the enclosing transaction
/// (mirrors `SET LOCAL` semantics).
pub(super) async fn set_tenant_session<'e, E>(executor: E, tenant_id: &str) -> Result<()>
where
    E: Executor<'e, Database = Postgres>,
{
    sqlx::query("SELECT set_config('entelix.tenant_id', $1, true)")
        .bind(tenant_id)
        .execute(executor)
        .await
        .map_err(backend_to_core)?;
    Ok(())
}

fn backend_to_core(e: sqlx::Error) -> Error {
    PersistenceError::Backend(e.to_string()).into()
}
