//! `VectorFilter` → SQL `WHERE` projection.
//!
//! Every variant in [`entelix_memory::VectorFilter`] maps onto a
//! parameterized SQL fragment built through `sqlx::QueryBuilder`,
//! so values ride as bind parameters (no string interpolation,
//! no SQL injection surface). JSON values that can't be coerced
//! to the SQL form a variant requires surface as
//! [`crate::PgVectorStoreError::FilterProjection`].
//!
//! Every projection is anchored by `namespace_key = $...` —
//! invariant 11 / F2 demand structural tenant isolation, and a
//! single forgotten anchor would silently cross tenants.

use entelix_memory::VectorFilter;
use serde_json::Value;
use sqlx::{Postgres, QueryBuilder};

use crate::error::{PgVectorStoreError, PgVectorStoreResult};

/// Append a `WHERE` clause to `qb` that anchors on
/// `namespace_key = ns_key` and conjoins the operator-supplied
/// filter (when present).
///
/// `qb` must be positioned right after the table reference
/// (`SELECT … FROM tbl ` is acceptable; the function emits its
/// own `WHERE`).
pub(crate) fn append_where(
    qb: &mut QueryBuilder<'_, Postgres>,
    namespace_key: &str,
    filter: Option<&VectorFilter>,
) -> PgVectorStoreResult<()> {
    qb.push(" WHERE namespace_key = ");
    qb.push_bind(namespace_key.to_owned());
    if let Some(f) = filter {
        qb.push(" AND (");
        append_filter_expr(qb, f)?;
        qb.push(")");
    }
    Ok(())
}

fn append_filter_expr(
    qb: &mut QueryBuilder<'_, Postgres>,
    filter: &VectorFilter,
) -> PgVectorStoreResult<()> {
    match filter {
        VectorFilter::All => {
            qb.push("TRUE");
        }
        VectorFilter::Eq { key, value } => {
            // JSON-aware containment: `metadata @> jsonb_build_object($k, $v)`.
            // Type-honest — comparing strings against integers yields
            // `false` instead of an SQL cast error.
            qb.push("metadata @> jsonb_build_object(");
            qb.push_bind(key.clone());
            qb.push(", ");
            qb.push_bind(sqlx::types::Json(value.clone()));
            qb.push(")");
        }
        VectorFilter::Lt { key, value } => append_numeric_compare(qb, key, "<", value)?,
        VectorFilter::Lte { key, value } => append_numeric_compare(qb, key, "<=", value)?,
        VectorFilter::Gt { key, value } => append_numeric_compare(qb, key, ">", value)?,
        VectorFilter::Gte { key, value } => append_numeric_compare(qb, key, ">=", value)?,
        VectorFilter::Range { key, min, max } => {
            qb.push("(metadata->>");
            qb.push_bind(key.clone());
            qb.push(")::numeric BETWEEN ");
            qb.push_bind(json_to_f64(min, key)?);
            qb.push(" AND ");
            qb.push_bind(json_to_f64(max, key)?);
        }
        VectorFilter::In { key, values } => append_in(qb, key, values)?,
        VectorFilter::Exists { key } => {
            qb.push("metadata ? ");
            qb.push_bind(key.clone());
        }
        VectorFilter::And(children) => append_combined(qb, children, " AND ")?,
        VectorFilter::Or(children) => append_combined(qb, children, " OR ")?,
        VectorFilter::Not(child) => {
            qb.push("NOT (");
            append_filter_expr(qb, child)?;
            qb.push(")");
        }
        // `VectorFilter` is `#[non_exhaustive]`. Future variants
        // surface here as a hard error rather than a silent
        // fallthrough so the projection is forced to evolve in
        // lockstep with the trait.
        other => {
            return Err(PgVectorStoreError::FilterProjection(format!(
                "unsupported VectorFilter variant for pgvector projection: {other:?}"
            )));
        }
    }
    Ok(())
}

fn append_numeric_compare(
    qb: &mut QueryBuilder<'_, Postgres>,
    key: &str,
    op: &'static str,
    value: &Value,
) -> PgVectorStoreResult<()> {
    qb.push("(metadata->>");
    qb.push_bind(key.to_owned());
    qb.push(")::numeric ");
    qb.push(op);
    qb.push(" ");
    qb.push_bind(json_to_f64(value, key)?);
    Ok(())
}

fn append_in(
    qb: &mut QueryBuilder<'_, Postgres>,
    key: &str,
    values: &[Value],
) -> PgVectorStoreResult<()> {
    if values.is_empty() {
        // `metadata ? key AND FALSE` — short-circuits to no match.
        // Keeps the SQL well-formed without touching `IN ()`,
        // which Postgres rejects as a syntax error.
        qb.push("FALSE");
        return Ok(());
    }
    if values.iter().all(serde_json::Value::is_string) {
        qb.push("metadata->>");
        qb.push_bind(key.to_owned());
        qb.push(" = ANY(");
        let strs: Vec<String> = values
            .iter()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect();
        qb.push_bind(strs);
        qb.push(")");
        return Ok(());
    }
    if values.iter().all(|v| v.is_i64()) {
        qb.push("(metadata->>");
        qb.push_bind(key.to_owned());
        qb.push(")::bigint = ANY(");
        let ints: Vec<i64> = values
            .iter()
            .filter_map(serde_json::Value::as_i64)
            .collect();
        qb.push_bind(ints);
        qb.push(")");
        return Ok(());
    }
    Err(PgVectorStoreError::FilterProjection(format!(
        "VectorFilter In for key '{key}': pgvector requires uniform \
         element types (all-string or all-i64); mixed list rejected"
    )))
}

fn append_combined(
    qb: &mut QueryBuilder<'_, Postgres>,
    children: &[VectorFilter],
    sep: &'static str,
) -> PgVectorStoreResult<()> {
    if children.is_empty() {
        // Empty And should match everything (identity for AND);
        // empty Or should match nothing (identity for OR). We
        // pick the conservatively-correct answer per operator.
        qb.push(if sep.contains("AND") { "TRUE" } else { "FALSE" });
        return Ok(());
    }
    qb.push("(");
    for (i, child) in children.iter().enumerate() {
        if i > 0 {
            qb.push(sep);
        }
        qb.push("(");
        append_filter_expr(qb, child)?;
        qb.push(")");
    }
    qb.push(")");
    Ok(())
}

fn json_to_f64(value: &Value, key: &str) -> PgVectorStoreResult<f64> {
    match value {
        Value::Number(n) => n
            .as_f64()
            .or_else(|| n.as_i64().map(|i| i as f64))
            .ok_or_else(|| {
                PgVectorStoreError::FilterProjection(format!(
                    "VectorFilter range bound for '{key}': numeric value out of f64 range"
                ))
            }),
        other => Err(PgVectorStoreError::FilterProjection(format!(
            "VectorFilter range bound for '{key}': numeric expected; got {other:?}"
        ))),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use serde_json::json;

    fn project(filter: Option<&VectorFilter>) -> PgVectorStoreResult<String> {
        let mut qb: QueryBuilder<'_, Postgres> = QueryBuilder::new("SELECT 1 FROM tbl");
        append_where(&mut qb, "tenant-a:default", filter)?;
        Ok(qb.into_sql())
    }

    #[test]
    fn anchor_is_always_first() {
        let sql = project(None).unwrap();
        assert!(sql.contains("namespace_key = "));
    }

    #[test]
    fn all_variant_yields_just_anchor_and_true() {
        let sql = project(Some(&VectorFilter::All)).unwrap();
        assert!(sql.contains("TRUE"));
    }

    #[test]
    fn eq_emits_jsonb_containment() {
        let sql = project(Some(&VectorFilter::Eq {
            key: "category".into(),
            value: json!("books"),
        }))
        .unwrap();
        assert!(sql.contains("jsonb_build_object"));
    }

    #[test]
    fn range_emits_between_clause() {
        let sql = project(Some(&VectorFilter::Range {
            key: "score".into(),
            min: json!(0.5),
            max: json!(0.9),
        }))
        .unwrap();
        assert!(sql.contains("BETWEEN"));
    }

    #[test]
    fn lt_lte_gt_gte_emit_numeric_cast() {
        for (f, op) in [
            (
                VectorFilter::Lt {
                    key: "k".into(),
                    value: json!(10),
                },
                "<",
            ),
            (
                VectorFilter::Lte {
                    key: "k".into(),
                    value: json!(10),
                },
                "<=",
            ),
            (
                VectorFilter::Gt {
                    key: "k".into(),
                    value: json!(10),
                },
                ">",
            ),
            (
                VectorFilter::Gte {
                    key: "k".into(),
                    value: json!(10),
                },
                ">=",
            ),
        ] {
            let sql = project(Some(&f)).unwrap();
            assert!(sql.contains("::numeric"), "{sql}");
            assert!(sql.contains(op), "{sql}");
        }
    }

    #[test]
    fn in_with_strings_emits_text_any() {
        let sql = project(Some(&VectorFilter::In {
            key: "tag".into(),
            values: vec![json!("a"), json!("b")],
        }))
        .unwrap();
        assert!(sql.contains("ANY("), "{sql}");
        assert!(sql.contains("metadata->>"), "{sql}");
    }

    #[test]
    fn in_with_integers_emits_bigint_cast() {
        let sql = project(Some(&VectorFilter::In {
            key: "year".into(),
            values: vec![json!(2024), json!(2025)],
        }))
        .unwrap();
        assert!(sql.contains("::bigint = ANY("), "{sql}");
    }

    #[test]
    fn in_with_mixed_types_is_rejected() {
        let err = project(Some(&VectorFilter::In {
            key: "k".into(),
            values: vec![json!("a"), json!(1)],
        }))
        .unwrap_err();
        assert!(matches!(err, PgVectorStoreError::FilterProjection(_)));
    }

    #[test]
    fn exists_emits_jsonb_question_operator() {
        let sql = project(Some(&VectorFilter::Exists {
            key: "field".into(),
        }))
        .unwrap();
        assert!(sql.contains("metadata ? "), "{sql}");
    }

    #[test]
    fn empty_in_emits_false_short_circuit() {
        let sql = project(Some(&VectorFilter::In {
            key: "k".into(),
            values: vec![],
        }))
        .unwrap();
        assert!(sql.contains("FALSE"), "{sql}");
    }

    #[test]
    fn and_or_not_compose_recursively() {
        let sql = project(Some(&VectorFilter::And(vec![
            VectorFilter::Eq {
                key: "a".into(),
                value: json!("x"),
            },
            VectorFilter::Or(vec![
                VectorFilter::Eq {
                    key: "b".into(),
                    value: json!(1),
                },
                VectorFilter::Not(Box::new(VectorFilter::Exists { key: "c".into() })),
            ]),
        ])))
        .unwrap();
        assert!(sql.contains(" AND "), "{sql}");
        assert!(sql.contains(" OR "), "{sql}");
        assert!(sql.contains("NOT ("), "{sql}");
    }

    #[test]
    fn empty_and_yields_true_identity() {
        let sql = project(Some(&VectorFilter::And(vec![]))).unwrap();
        assert!(sql.contains("TRUE"));
    }

    #[test]
    fn empty_or_yields_false_identity() {
        let sql = project(Some(&VectorFilter::Or(vec![]))).unwrap();
        assert!(sql.contains("FALSE"));
    }
}
