//! Property-based regression for the four built-in `Reducer` impls
//! (`Replace`, `Append`, `Max`, `MergeMap`). Hand-written unit tests
//! in `reducer.rs` cover representative cases; these properties pin
//! the algebraic invariants every operator's `add_node_with` /
//! `derive(StateMerge)` composition silently relies on.

#![allow(clippy::unwrap_used)]

use std::collections::HashMap;

use entelix_graph::{Append, Max, MergeMap, Reducer, Replace};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        ..ProptestConfig::default()
    })]

    /// `Replace` always returns the update — current is discarded.
    /// This is the right-bias contract every node returning a full
    /// `S` relies on (LangGraph default reducer parity).
    #[test]
    fn replace_returns_update(current: i64, update: i64) {
        let r = Replace;
        prop_assert_eq!(r.reduce(current, update), update);
    }

    /// `Append` is concatenation: length is the sum, contents are
    /// `current ++ update` in order. Lacking this, parallel-branch
    /// joins via `add_send_edges` would re-order observations.
    #[test]
    fn append_concatenates(current: Vec<i32>, update: Vec<i32>) {
        let r = Append::<i32>::new();
        let merged = r.reduce(current.clone(), update.clone());
        prop_assert_eq!(merged.len(), current.len() + update.len());
        let mut expected = current;
        expected.extend(update);
        prop_assert_eq!(merged, expected);
    }

    /// `Append` is associative: `((a ++ b) ++ c) == (a ++ (b ++ c))`.
    /// Order-of-fold variations across parallel super-steps must
    /// converge on the same final value.
    #[test]
    fn append_is_associative(a: Vec<i32>, b: Vec<i32>, c: Vec<i32>) {
        let r = Append::<i32>::new();
        let left = r.reduce(r.reduce(a.clone(), b.clone()), c.clone());
        let right = r.reduce(a, r.reduce(b, c));
        prop_assert_eq!(left, right);
    }

    /// `Max` returns the greater value. Equivalent to `cmp::max`
    /// for any total-ordered `T: Ord`.
    #[test]
    fn max_returns_greater(current: i64, update: i64) {
        let r = Max::<i64>::new();
        prop_assert_eq!(r.reduce(current, update), current.max(update));
    }

    /// `Max` is commutative: order of arguments doesn't matter.
    #[test]
    fn max_is_commutative(a: i64, b: i64) {
        let r = Max::<i64>::new();
        prop_assert_eq!(r.reduce(a, b), r.reduce(b, a));
    }

    /// `Max` is idempotent: `reduce(a, a) == a`.
    #[test]
    fn max_is_idempotent(a: i64) {
        let r = Max::<i64>::new();
        prop_assert_eq!(r.reduce(a, a), a);
    }

    /// `MergeMap` is right-biased — every key in `update` appears
    /// in the result with `update`'s value. Keys present only in
    /// `current` are preserved.
    #[test]
    fn merge_map_right_biased(
        current_pairs in proptest::collection::vec((".{0,8}", any::<i32>()), 0..16),
        update_pairs in proptest::collection::vec((".{0,8}", any::<i32>()), 0..16),
    ) {
        let current: HashMap<String, i32> = current_pairs.into_iter().collect();
        let update: HashMap<String, i32> = update_pairs.into_iter().collect();
        let r = MergeMap::<String, i32>::new();
        let merged = r.reduce(current.clone(), update.clone());
        // Every update key wins.
        for (k, v) in &update {
            prop_assert_eq!(merged.get(k), Some(v));
        }
        // Every current key absent from update is preserved.
        for (k, v) in &current {
            if !update.contains_key(k) {
                prop_assert_eq!(merged.get(k), Some(v));
            }
        }
        // No new keys spawned.
        for k in merged.keys() {
            prop_assert!(current.contains_key(k) || update.contains_key(k));
        }
    }

    /// `MergeMap::reduce(empty, update) == update` — left identity.
    #[test]
    fn merge_map_left_identity(update_pairs in proptest::collection::vec((".{0,8}", any::<i32>()), 0..16)) {
        let update: HashMap<String, i32> = update_pairs.into_iter().collect();
        let r = MergeMap::<String, i32>::new();
        let merged = r.reduce(HashMap::new(), update.clone());
        prop_assert_eq!(merged, update);
    }
}
