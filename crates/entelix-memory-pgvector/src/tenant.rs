//! Tenant-scoped session-variable helper for the RLS-enforced
//! `entelix_vectors` table (invariant #11 defense in depth,
//! mirroring 's treatment of the
//! `entelix-persistence` and `entelix-graphmemory-pg` tables).
//!
//! Every tenant-scoped query opens a transaction, calls
//! [`set_tenant_session`] to stamp `entelix.tenant_id` for the
//! duration of the transaction, runs its query, and commits.
//! `set_config(name, value, true)` mirrors `SET LOCAL` semantics
//! — the variable is scoped to the enclosing transaction; pool
//! connections that return to the pool carry no leftover state.
//!
//! Per-companion replication of the helper (rather than a
//! centralised crate) is deliberate: `entelix-memory`
//! is sqlx-free by, and inventing a new utility crate
//! for one helper is over-engineered. The 6-line function is
//! trivial and identical across companions.

use entelix_core::TenantId;
use entelix_core::error::Error;
use sqlx::Executor;
use sqlx::postgres::Postgres;

use crate::error::PgVectorStoreError;

/// Stamp the current transaction's `entelix.tenant_id` session
/// variable. The third argument to `set_config` is `is_local =
/// true`, scoping the assignment to the enclosing transaction
/// (mirrors `SET LOCAL` semantics). Takes the typed [`TenantId`]
/// so the policy cannot be armed with a tenantless value.
pub(super) async fn set_tenant_session<'e, E>(
    executor: E,
    tenant_id: &TenantId,
) -> entelix_core::Result<()>
where
    E: Executor<'e, Database = Postgres>,
{
    sqlx::query("SELECT set_config('entelix.tenant_id', $1, true)")
        .bind(tenant_id.as_str())
        .execute(executor)
        .await
        .map_err(into_core_sqlx)?;
    Ok(())
}

fn into_core_sqlx(e: sqlx::Error) -> Error {
    PgVectorStoreError::from(e).into()
}
