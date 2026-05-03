//! Tenant-scoped session-variable helper for the RLS-enforced
//! `graph_nodes` + `graph_edges` tables (invariant #11 defense
//! in depth, mirroring ADR-0041's treatment of the
//! `entelix-persistence` tables).
//!
//! Every tenant-scoped query opens a transaction, calls
//! [`set_tenant_session`] to stamp `entelix.tenant_id` for the
//! duration of the transaction, runs its query, and commits.
//! `set_config(name, value, true)` mirrors `SET LOCAL` semantics
//! — the variable is scoped to the enclosing transaction; pool
//! connections that return to the pool carry no leftover state.
//!
//! See `crates/entelix-persistence/src/postgres/tenant.rs` for the
//! sibling helper used by the SDK's other Postgres-backed
//! storage tables.

use entelix_core::error::Error;
use sqlx::Executor;
use sqlx::postgres::Postgres;

use crate::error::PgGraphMemoryError;

/// Stamp the current transaction's `entelix.tenant_id` session
/// variable. The third argument to `set_config` is `is_local =
/// true`, scoping the assignment to the enclosing transaction
/// (mirrors `SET LOCAL` semantics).
pub(super) async fn set_tenant_session<'e, E>(
    executor: E,
    tenant_id: &str,
) -> entelix_core::Result<()>
where
    E: Executor<'e, Database = Postgres>,
{
    sqlx::query("SELECT set_config('entelix.tenant_id', $1, true)")
        .bind(tenant_id)
        .execute(executor)
        .await
        .map_err(into_core_sqlx)?;
    Ok(())
}

fn into_core_sqlx(e: sqlx::Error) -> Error {
    PgGraphMemoryError::from(e).into()
}
