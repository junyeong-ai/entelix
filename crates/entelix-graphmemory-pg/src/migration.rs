//! Idempotent schema bootstrap for `PgGraphMemory`.
//!
//! Run unconditionally at builder finalize time when
//! `auto_migrate=true`. Operators that own the schema externally
//! (DBA-managed deployments, infrastructure-as-code) opt out via
//! `with_auto_migrate(false)`.
//!
//! Migration shape:
//!
//! 1. `CREATE TABLE IF NOT EXISTS <nodes> …` — composite PK
//!    `(namespace_key, id)` plus a denormalised `tenant_id TEXT`
//!    column so the row-level-security policy can filter by
//!    tenant without parsing `namespace_key` per query.
//! 2. `CREATE TABLE IF NOT EXISTS <edges> …` — same composite PK
//!    and `tenant_id` shape, plus separate covering indexes for
//!    `(namespace_key, from_node)`, `(namespace_key, to_node)`,
//!    `(namespace_key, ts)` so neighbour and temporal queries
//!    stay O(log n).
//! 3. `ALTER TABLE … ENABLE ROW LEVEL SECURITY` +
//!    `FORCE ROW LEVEL SECURITY` on both tables (mirrors the
//!    `entelix-persistence` treatment from ADR-0041 for invariant
//!    11 defense in depth).
//! 4. `CREATE POLICY tenant_isolation` — single policy spanning
//!    `USING` (reads) + `WITH CHECK` (writes), gating on
//!    `tenant_id = current_setting('entelix.tenant_id', true)`.
//!    The SDK stamps that variable per transaction; an unset
//!    variable surfaces as `NULL`, which the comparison treats
//!    as false → fail-closed (invariant #15).
//!
//! All statements are guarded by `IF NOT EXISTS` /
//! `IF EXISTS … DROP POLICY` patterns so the migration is
//! idempotent — calling it on an already-bootstrapped database
//! is a no-op.

use sqlx::PgPool;

use crate::error::{PgGraphMemoryError, PgGraphMemoryResult};

pub(crate) async fn bootstrap(
    pool: &PgPool,
    nodes_table: &str,
    edges_table: &str,
) -> PgGraphMemoryResult<()> {
    if !is_safe_identifier(nodes_table) {
        return Err(PgGraphMemoryError::Config(format!(
            "nodes table name '{nodes_table}' contains characters disallowed for SQL identifiers"
        )));
    }
    if !is_safe_identifier(edges_table) {
        return Err(PgGraphMemoryError::Config(format!(
            "edges table name '{edges_table}' contains characters disallowed for SQL identifiers"
        )));
    }

    let create_nodes = format!(
        "CREATE TABLE IF NOT EXISTS {nodes_table} (\n\
            tenant_id TEXT NOT NULL,\n\
            namespace_key TEXT NOT NULL,\n\
            id TEXT NOT NULL,\n\
            payload JSONB NOT NULL,\n\
            PRIMARY KEY (namespace_key, id)\n\
         )"
    );
    sqlx::query(&create_nodes).execute(pool).await?;

    let create_edges = format!(
        "CREATE TABLE IF NOT EXISTS {edges_table} (\n\
            tenant_id TEXT NOT NULL,\n\
            namespace_key TEXT NOT NULL,\n\
            id TEXT NOT NULL,\n\
            from_node TEXT NOT NULL,\n\
            to_node TEXT NOT NULL,\n\
            payload JSONB NOT NULL,\n\
            ts TIMESTAMPTZ NOT NULL,\n\
            PRIMARY KEY (namespace_key, id)\n\
         )"
    );
    sqlx::query(&create_edges).execute(pool).await?;

    let create_from_idx = format!(
        "CREATE INDEX IF NOT EXISTS {edges_table}_from_idx \
         ON {edges_table} (namespace_key, from_node)"
    );
    sqlx::query(&create_from_idx).execute(pool).await?;

    let create_to_idx = format!(
        "CREATE INDEX IF NOT EXISTS {edges_table}_to_idx \
         ON {edges_table} (namespace_key, to_node)"
    );
    sqlx::query(&create_to_idx).execute(pool).await?;

    let create_ts_idx = format!(
        "CREATE INDEX IF NOT EXISTS {edges_table}_ts_idx \
         ON {edges_table} (namespace_key, ts)"
    );
    sqlx::query(&create_ts_idx).execute(pool).await?;

    enable_rls(pool, nodes_table).await?;
    enable_rls(pool, edges_table).await?;

    Ok(())
}

/// Enable + FORCE ROW LEVEL SECURITY on `table` and install the
/// `tenant_isolation` policy (USING + WITH CHECK on
/// `current_setting('entelix.tenant_id', true)`).
///
/// Idempotent: `ENABLE` and `FORCE` are no-ops on already-enabled
/// tables; the policy is dropped (`IF EXISTS`) before being
/// re-created so re-running the bootstrap doesn't accumulate
/// duplicate policies.
async fn enable_rls(pool: &sqlx::PgPool, table: &str) -> PgGraphMemoryResult<()> {
    sqlx::query(&format!("ALTER TABLE {table} ENABLE ROW LEVEL SECURITY"))
        .execute(pool)
        .await?;
    sqlx::query(&format!("ALTER TABLE {table} FORCE ROW LEVEL SECURITY"))
        .execute(pool)
        .await?;
    sqlx::query(&format!(
        "DROP POLICY IF EXISTS tenant_isolation ON {table}"
    ))
    .execute(pool)
    .await?;
    sqlx::query(&format!(
        "CREATE POLICY tenant_isolation ON {table} \
         USING (tenant_id = current_setting('entelix.tenant_id', true)) \
         WITH CHECK (tenant_id = current_setting('entelix.tenant_id', true))"
    ))
    .execute(pool)
    .await?;
    Ok(())
}

/// Conservative SQL-identifier check — reject anything that isn't a
/// straightforward unquoted identifier so we can interpolate the
/// table name into raw SQL without quoting.
fn is_safe_identifier(s: &str) -> bool {
    if s.is_empty() || s.len() > 63 {
        return false;
    }
    let mut chars = s.chars();
    let first = chars.next().expect("non-empty checked above");
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_identifiers_accepted() {
        assert!(is_safe_identifier("graph_nodes"));
        assert!(is_safe_identifier("graph_edges"));
        assert!(is_safe_identifier("_private"));
    }

    #[test]
    fn unsafe_identifiers_rejected() {
        assert!(!is_safe_identifier(""));
        assert!(!is_safe_identifier("9starts_with_digit"));
        assert!(!is_safe_identifier("with space"));
        assert!(!is_safe_identifier("drop;--"));
        assert!(!is_safe_identifier(&"x".repeat(64)));
    }
}
