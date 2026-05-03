//! `VectorFilter` → qdrant `Filter` projection.
//!
//! Every variant in [`entelix_memory::VectorFilter`] maps onto qdrant's
//! native filter taxonomy — no silent drops. JSON values that cannot
//! be coerced to a qdrant `MatchValue` or numeric range bound surface
//! as [`crate::QdrantStoreError::FilterProjection`] so operators see
//! exactly what the back-end could not represent.
//!
//! Every projection is anchored by a `must`-clause pinning
//! [`NAMESPACE_KEY`] to the rendered [`entelix_memory::Namespace`].
//! That anchor is mandatory — invariant 11 / F2 demand structural
//! tenant isolation, and a single forgotten anchor would silently
//! cross tenants.

use entelix_memory::VectorFilter;
use qdrant_client::qdrant::{Condition, Filter, Range, r#match::MatchValue};
use serde_json::Value;

use crate::error::{QdrantStoreError, QdrantStoreResult};

/// Payload key under which the rendered `Namespace` is stamped on
/// every point. Indexed at collection-create time so the
/// `must`-clause that always rides on every search/count/list
/// query is a single index probe rather than a full scan.
pub const NAMESPACE_KEY: &str = "entelix_namespace_key";
/// Payload key carrying the document's content text.
pub const CONTENT_KEY: &str = "entelix_content";
/// Payload key carrying the document's metadata (flat map). User
/// metadata fields ride here as nested JSON; filter expressions
/// targeting metadata reference them via `metadata.<key>` paths.
pub const METADATA_KEY: &str = "entelix_metadata";
/// Payload key carrying the operator-supplied `doc_id`. Distinct
/// from qdrant's internal `PointId` so filters / scrolls can
/// surface the operator id without a server-side reverse mapping.
pub const DOC_ID_KEY: &str = "entelix_doc_id";

/// Project an optional [`VectorFilter`] onto a qdrant [`Filter`],
/// always anchored by a `must` clause that pins
/// [`NAMESPACE_KEY`] to the rendered `Namespace`.
///
/// Cross-tenant data leak is structurally impossible — every
/// read/write/count/list rides this anchor.
pub(crate) fn project(
    filter: Option<&VectorFilter>,
    namespace_key: &str,
) -> QdrantStoreResult<Filter> {
    let mut anchor = Filter::default();
    anchor
        .must
        .push(Condition::matches(NAMESPACE_KEY, namespace_key.to_owned()));
    if let Some(f) = filter {
        let projected = project_filter(f)?;
        anchor.must.push(Condition::from(projected));
    }
    Ok(anchor)
}

fn project_filter(filter: &VectorFilter) -> QdrantStoreResult<Filter> {
    match filter {
        VectorFilter::All => Ok(Filter::default()),
        VectorFilter::Eq { key, value } => {
            let mv = json_to_match_value(value, key)?;
            Ok(Filter::must([Condition::matches(metadata_path(key), mv)]))
        }
        VectorFilter::Lt { key, value } => Ok(Filter::must([Condition::range(
            metadata_path(key),
            Range {
                lt: Some(json_to_f64(value, key)?),
                ..Default::default()
            },
        )])),
        VectorFilter::Lte { key, value } => Ok(Filter::must([Condition::range(
            metadata_path(key),
            Range {
                lte: Some(json_to_f64(value, key)?),
                ..Default::default()
            },
        )])),
        VectorFilter::Gt { key, value } => Ok(Filter::must([Condition::range(
            metadata_path(key),
            Range {
                gt: Some(json_to_f64(value, key)?),
                ..Default::default()
            },
        )])),
        VectorFilter::Gte { key, value } => Ok(Filter::must([Condition::range(
            metadata_path(key),
            Range {
                gte: Some(json_to_f64(value, key)?),
                ..Default::default()
            },
        )])),
        VectorFilter::Range { key, min, max } => Ok(Filter::must([Condition::range(
            metadata_path(key),
            Range {
                gte: Some(json_to_f64(min, key)?),
                lte: Some(json_to_f64(max, key)?),
                ..Default::default()
            },
        )])),
        VectorFilter::In { key, values } => {
            let mv = json_array_to_match_value(values, key)?;
            Ok(Filter::must([Condition::matches(metadata_path(key), mv)]))
        }
        VectorFilter::Exists { key } => {
            // qdrant's `is_null(field)` returns true when the key is
            // unset OR explicitly null. "exists" is the negation:
            // present and not-null. We project as `must_not(is_null)`
            // — the qdrant docs guarantee that an unset payload key
            // is reported as null.
            Ok(Filter::must_not([Condition::is_null(metadata_path(key))]))
        }
        VectorFilter::And(children) => {
            let mut out = Filter::default();
            for child in children {
                out.must.push(Condition::from(project_filter(child)?));
            }
            Ok(out)
        }
        VectorFilter::Or(children) => {
            let mut out = Filter::default();
            for child in children {
                out.should.push(Condition::from(project_filter(child)?));
            }
            Ok(out)
        }
        VectorFilter::Not(child) => Ok(Filter::must_not([Condition::from(project_filter(child)?)])),
        // `VectorFilter` is `#[non_exhaustive]`. Future variants
        // surface here as a hard error rather than a silent
        // fallthrough so the projection is forced to evolve in
        // lockstep with the trait.
        other => Err(QdrantStoreError::FilterProjection(format!(
            "unsupported VectorFilter variant for qdrant projection: {other:?}"
        ))),
    }
}

/// Compose the metadata-prefixed path qdrant expects when filtering
/// on a user-supplied metadata key. Keeps user-key namespaces
/// disjoint from entelix's own bookkeeping (`entelix_*`) so a
/// user metadata key collision with `namespace_key` can never
/// shadow the tenant anchor.
fn metadata_path(user_key: &str) -> String {
    format!("{METADATA_KEY}.{user_key}")
}

fn json_to_match_value(value: &Value, key: &str) -> QdrantStoreResult<MatchValue> {
    match value {
        Value::Bool(b) => Ok(MatchValue::Boolean(*b)),
        Value::Number(n) if n.is_i64() => Ok(MatchValue::Integer(n.as_i64().unwrap_or(0))),
        Value::String(s) => Ok(MatchValue::Keyword(s.clone())),
        other => Err(QdrantStoreError::FilterProjection(format!(
            "VectorFilter Eq for key '{key}': qdrant matches only \
             support bool / i64 / string scalars; got {other:?}"
        ))),
    }
}

fn json_array_to_match_value(values: &[Value], key: &str) -> QdrantStoreResult<MatchValue> {
    if values.is_empty() {
        return Err(QdrantStoreError::FilterProjection(format!(
            "VectorFilter In for key '{key}': empty values list — \
             upstream is expected to short-circuit before projection"
        )));
    }
    if values.iter().all(serde_json::Value::is_string) {
        let strings: Vec<String> = values
            .iter()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect();
        return Ok(MatchValue::Keywords(
            qdrant_client::qdrant::RepeatedStrings { strings },
        ));
    }
    if values.iter().all(|v| v.is_i64()) {
        let integers: Vec<i64> = values
            .iter()
            .filter_map(serde_json::Value::as_i64)
            .collect();
        return Ok(MatchValue::Integers(
            qdrant_client::qdrant::RepeatedIntegers { integers },
        ));
    }
    Err(QdrantStoreError::FilterProjection(format!(
        "VectorFilter In for key '{key}': qdrant requires uniform \
         element types (all-string or all-i64); mixed list rejected"
    )))
}

fn json_to_f64(value: &Value, key: &str) -> QdrantStoreResult<f64> {
    match value {
        Value::Number(n) => n
            .as_f64()
            .or_else(|| n.as_i64().map(|i| i as f64))
            .ok_or_else(|| {
                QdrantStoreError::FilterProjection(format!(
                    "VectorFilter range bound for '{key}': numeric value out of f64 range"
                ))
            }),
        other => Err(QdrantStoreError::FilterProjection(format!(
            "VectorFilter range bound for '{key}': numeric expected; got {other:?}"
        ))),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn anchor_is_always_present() {
        let f = project(None, "tenant-a:default").unwrap();
        assert_eq!(f.must.len(), 1, "anchor namespace must clause is mandatory");
    }

    #[test]
    fn all_variant_yields_just_the_anchor() {
        let f = project(Some(&VectorFilter::All), "t:s").unwrap();
        // Anchor + projected (empty) filter wrapper.
        assert_eq!(f.must.len(), 2);
    }

    #[test]
    fn eq_string_projects_to_keyword_match() {
        let f = project(
            Some(&VectorFilter::Eq {
                key: "category".into(),
                value: json!("books"),
            }),
            "t:s",
        )
        .unwrap();
        assert_eq!(f.must.len(), 2, "anchor + Eq wrapper");
    }

    #[test]
    fn eq_integer_projects_to_integer_match() {
        let f = project(
            Some(&VectorFilter::Eq {
                key: "year".into(),
                value: json!(2026),
            }),
            "t:s",
        )
        .unwrap();
        assert_eq!(f.must.len(), 2);
    }

    #[test]
    fn range_projects_lower_and_upper_bounds() {
        project(
            Some(&VectorFilter::Range {
                key: "score".into(),
                min: json!(0.5),
                max: json!(0.9),
            }),
            "t:s",
        )
        .unwrap();
    }

    #[test]
    fn in_with_strings_projects_to_keywords() {
        project(
            Some(&VectorFilter::In {
                key: "tag".into(),
                values: vec![json!("a"), json!("b")],
            }),
            "t:s",
        )
        .unwrap();
    }

    #[test]
    fn in_with_integers_projects_to_integers() {
        project(
            Some(&VectorFilter::In {
                key: "year".into(),
                values: vec![json!(2024), json!(2025)],
            }),
            "t:s",
        )
        .unwrap();
    }

    #[test]
    fn in_with_mixed_types_is_rejected() {
        let err = project(
            Some(&VectorFilter::In {
                key: "k".into(),
                values: vec![json!("a"), json!(1)],
            }),
            "t:s",
        )
        .unwrap_err();
        assert!(
            matches!(err, QdrantStoreError::FilterProjection(_)),
            "{err:?}"
        );
    }

    #[test]
    fn exists_projects_to_must_not_is_null() {
        project(
            Some(&VectorFilter::Exists {
                key: "field".into(),
            }),
            "t:s",
        )
        .unwrap();
    }

    #[test]
    fn and_or_not_compose_recursively() {
        project(
            Some(&VectorFilter::And(vec![
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
            ])),
            "t:s",
        )
        .unwrap();
    }

    #[test]
    fn metadata_path_namespaces_under_metadata_prefix() {
        assert_eq!(metadata_path("category"), "entelix_metadata.category");
    }

    #[test]
    fn lt_lte_gt_gte_each_set_one_bound() {
        for f in [
            VectorFilter::Lt {
                key: "k".into(),
                value: json!(10),
            },
            VectorFilter::Lte {
                key: "k".into(),
                value: json!(10),
            },
            VectorFilter::Gt {
                key: "k".into(),
                value: json!(10),
            },
            VectorFilter::Gte {
                key: "k".into(),
                value: json!(10),
            },
        ] {
            project(Some(&f), "t:s").unwrap();
        }
    }
}
