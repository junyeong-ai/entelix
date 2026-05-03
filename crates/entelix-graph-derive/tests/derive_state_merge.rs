//! Integration tests for `#[derive(StateMerge)]` — verify the
//! generated impl on (a) a struct mixing `Annotated` and plain
//! fields, (b) a struct with only `Annotated` fields, and (c) a
//! struct with only plain fields.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use entelix_graph::{Annotated, Append, Max, MergeMap, Replace, StateMerge};
use std::collections::HashMap;

#[derive(Clone, Default, StateMerge)]
struct Mixed {
    log: Annotated<Vec<String>, Append<String>>,
    score: Annotated<i32, Max<i32>>,
    map: Annotated<HashMap<String, i32>, MergeMap<String, i32>>,
    last_message: String,
    counter: u32,
}

#[test]
fn mixed_struct_applies_per_field_reducer() {
    let mut map_a = HashMap::new();
    map_a.insert("a".into(), 1);
    map_a.insert("b".into(), 2);
    let mut map_b = HashMap::new();
    map_b.insert("b".into(), 20);
    map_b.insert("c".into(), 3);

    let current = Mixed {
        log: Annotated::new(vec!["seed".into()], Append::new()),
        score: Annotated::new(80, Max::new()),
        map: Annotated::new(map_a, MergeMap::new()),
        last_message: "before".into(),
        counter: 5,
    };
    let update = Mixed {
        log: Annotated::new(vec!["new".into()], Append::new()),
        score: Annotated::new(50, Max::new()),
        map: Annotated::new(map_b, MergeMap::new()),
        last_message: "after".into(),
        counter: 1,
    };
    let merged = current.merge(update);

    assert_eq!(merged.log.value, vec!["seed".to_owned(), "new".into()]);
    assert_eq!(merged.score.value, 80);
    assert_eq!(merged.map.value.get("a"), Some(&1));
    assert_eq!(merged.map.value.get("b"), Some(&20));
    assert_eq!(merged.map.value.get("c"), Some(&3));
    assert_eq!(merged.last_message, "after");
    assert_eq!(merged.counter, 1);
}

#[derive(Clone, Default, StateMerge)]
struct AllAnnotated {
    a: Annotated<Vec<i32>, Append<i32>>,
    b: Annotated<i32, Max<i32>>,
}

#[test]
fn struct_with_only_annotated_fields_merges_each() {
    let merged = AllAnnotated {
        a: Annotated::new(vec![1, 2], Append::new()),
        b: Annotated::new(3, Max::new()),
    }
    .merge(AllAnnotated {
        a: Annotated::new(vec![3], Append::new()),
        b: Annotated::new(7, Max::new()),
    });
    assert_eq!(merged.a.value, vec![1, 2, 3]);
    assert_eq!(merged.b.value, 7);
}

#[derive(Clone, Default, StateMerge)]
struct AllPlain {
    name: String,
    value: i64,
}

#[test]
fn struct_with_only_plain_fields_replaces_each() {
    let merged = AllPlain {
        name: "old".into(),
        value: 1,
    }
    .merge(AllPlain {
        name: "new".into(),
        value: 99,
    });
    assert_eq!(merged.name, "new");
    assert_eq!(merged.value, 99);
}

#[derive(Clone, Default, StateMerge)]
struct ExplicitReplace {
    // `Replace` reducer through the `Annotated` wrapper exists for
    // completeness — should behave identically to a plain field.
    label: Annotated<String, Replace>,
}

#[test]
fn annotated_with_replace_reducer_behaves_like_plain_field() {
    let merged = ExplicitReplace {
        label: Annotated::new("old".into(), Replace),
    }
    .merge(ExplicitReplace {
        label: Annotated::new("new".into(), Replace),
    });
    assert_eq!(merged.label.value, "new");
}

#[derive(Clone, Default, StateMerge)]
struct WithGenerics<T>
where
    T: Clone + Default + Send + Sync + 'static + Ord,
{
    items: Annotated<T, Max<T>>,
    label: String,
}

#[test]
fn contribution_companion_with_builder_methods() {
    // The derive emits a `<Name>Contribution` companion struct
    // with a `with_<field>` builder per field. Annotated fields'
    // builder accepts raw inner T (auto-wrapped).
    let contribution = MixedContribution::default()
        .with_log(vec!["entry".into()])
        .with_score(50)
        .with_last_message("hello".into());
    let current = Mixed {
        log: Annotated::new(vec!["seed".into()], Append::new()),
        score: Annotated::new(80, Max::new()),
        map: Annotated::new(HashMap::new(), MergeMap::new()),
        last_message: "before".into(),
        counter: 5,
    };
    let merged = current.merge_contribution(contribution);
    // log: appended
    assert_eq!(merged.log.value, vec!["seed".to_owned(), "entry".into()]);
    // score: 80 (max of 80, 50)
    assert_eq!(merged.score.value, 80);
    // last_message: replaced
    assert_eq!(merged.last_message, "hello");
    // map / counter: not in contribution → kept as-is
    assert!(merged.map.value.is_empty());
    assert_eq!(merged.counter, 5);
}

#[test]
fn contribution_with_no_slots_keeps_every_current_value() {
    // Empty contribution must not regress *any* slot — including
    // edge cases like `Annotated<i32, Max>` with negative current
    // value, where a default-zero contribution would have
    // collapsed -100 to 0 under same-shape merge.
    let contribution = MixedContribution::default();
    let current = Mixed {
        log: Annotated::new(vec!["seed".into()], Append::new()),
        score: Annotated::new(-100, Max::new()),
        map: Annotated::new(HashMap::new(), MergeMap::new()),
        last_message: "keep-me".into(),
        counter: 42,
    };
    let merged = current.merge_contribution(contribution);
    assert_eq!(merged.log.value, vec!["seed"]);
    assert_eq!(
        merged.score.value, -100,
        "max-reducer must not see default 0"
    );
    assert_eq!(merged.last_message, "keep-me");
    assert_eq!(merged.counter, 42);
}

#[test]
fn generic_struct_supported() {
    let merged = WithGenerics::<i32> {
        items: Annotated::new(10, Max::new()),
        label: "before".into(),
    }
    .merge(WithGenerics::<i32> {
        items: Annotated::new(7, Max::new()),
        label: "after".into(),
    });
    assert_eq!(merged.items.value, 10);
    assert_eq!(merged.label, "after");
}
