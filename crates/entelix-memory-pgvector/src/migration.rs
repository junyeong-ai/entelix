//! Idempotent schema bootstrap for `PgVectorStore`.
//!
//! Run unconditionally at builder finalize time when
//! `auto_migrate=true`. Operators that own the schema externally
//! (DBA-managed deployments, infrastructure-as-code) opt out via
//! `with_auto_migrate(false)`.
//!
//! Migration shape:
//!
//! 1. `CREATE EXTENSION IF NOT EXISTS vector` — Postgres extension
//!    enabling the `VECTOR(N)` type and ANN operators.
//! 2. `CREATE TABLE IF NOT EXISTS …` — primary table keyed
//!    `(namespace_key, doc_id)`. The composite PK doubles as the
//!    btree index every namespace-anchored query relies on, so no
//!    separate `CREATE INDEX namespace_key_idx` is needed.
//! 3. `CREATE INDEX … USING hnsw …` — vector index with the
//!    operator class matching the configured distance metric.
//!    HNSW is the default; IVFFlat is selected via the operator
//!    `IndexKind` (note: IVFFlat needs operator-side `lists` tuning).
//! 4. `CREATE INDEX … USING gin (metadata jsonb_path_ops)` —
//!    metadata predicate acceleration for filtered searches.
//!
//! All four are guarded by `IF NOT EXISTS` so the migration is
//! idempotent — calling it on an already-bootstrapped database
//! is a no-op.

use sqlx::PgPool;

use crate::error::{PgVectorStoreError, PgVectorStoreResult};
use crate::store::{DistanceMetric, IndexKind};

pub(crate) async fn bootstrap(
    pool: &PgPool,
    table: &str,
    dimension: usize,
    distance: DistanceMetric,
    index_kind: IndexKind,
) -> PgVectorStoreResult<()> {
    if !is_safe_identifier(table) {
        return Err(PgVectorStoreError::Config(format!(
            "table name '{table}' contains characters disallowed for SQL identifiers; \
             use [a-zA-Z_][a-zA-Z0-9_]*"
        )));
    }
    sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
        .execute(pool)
        .await?;

    let create_table = format!(
        "CREATE TABLE IF NOT EXISTS {table} (\n\
            tenant_id TEXT NOT NULL,\n\
            namespace_key TEXT NOT NULL,\n\
            doc_id TEXT NOT NULL,\n\
            content TEXT NOT NULL DEFAULT '',\n\
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,\n\
            embedding VECTOR({dimension}) NOT NULL,\n\
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),\n\
            PRIMARY KEY (namespace_key, doc_id)\n\
         )"
    );
    sqlx::query(&create_table).execute(pool).await?;

    let index_method = match index_kind {
        IndexKind::Hnsw => "hnsw",
        IndexKind::IvfFlat => "ivfflat",
    };
    let op_class = match distance {
        DistanceMetric::Cosine => "vector_cosine_ops",
        DistanceMetric::L2 => "vector_l2_ops",
        DistanceMetric::InnerProduct => "vector_ip_ops",
    };
    let create_vec_idx = format!(
        "CREATE INDEX IF NOT EXISTS {table}_embedding_idx \
         ON {table} USING {index_method} (embedding {op_class})"
    );
    sqlx::query(&create_vec_idx).execute(pool).await?;

    let create_meta_idx = format!(
        "CREATE INDEX IF NOT EXISTS {table}_metadata_idx \
         ON {table} USING gin (metadata jsonb_path_ops)"
    );
    sqlx::query(&create_meta_idx).execute(pool).await?;

    enable_rls(pool, table).await?;

    Ok(())
}

/// Enable + FORCE ROW LEVEL SECURITY on `table` and install the
/// `tenant_isolation` policy (USING + WITH CHECK on
/// `current_setting('entelix.tenant_id', true)`). Mirrors the
/// `entelix-graphmemory-pg` treatment from — companion-
/// crate RLS pattern (defense in depth for invariant #11).
///
/// Idempotent: `ENABLE` and `FORCE` are no-ops on already-enabled
/// tables; the policy is dropped (`IF EXISTS`) before being
/// re-created so re-running the bootstrap doesn't accumulate
/// duplicate policies.
async fn enable_rls(pool: &PgPool, table: &str) -> PgVectorStoreResult<()> {
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
        assert!(is_safe_identifier("entelix_vectors"));
        assert!(is_safe_identifier("Tbl"));
        assert!(is_safe_identifier("_private_table"));
        assert!(is_safe_identifier("vec_2026"));
    }

    #[test]
    fn unsafe_identifiers_rejected() {
        assert!(!is_safe_identifier(""));
        assert!(!is_safe_identifier("9starts_with_digit"));
        assert!(!is_safe_identifier("with space"));
        assert!(!is_safe_identifier("drop;--"));
        assert!(!is_safe_identifier("with'quote"));
        assert!(!is_safe_identifier(&"x".repeat(64)));
    }
}
