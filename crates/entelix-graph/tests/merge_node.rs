//! `add_node_with` integration tests — proves the delta-node
//! ergonomic composes inside a real `StateGraph` and that the
//! `Reducer<T>` impls (`Append`, `Max`, `MergeMap`) plug in naturally
//! from the merger closure.

#![allow(clippy::unwrap_used)]

use std::collections::HashMap;

use entelix_core::context::ExecutionContext;
use entelix_graph::{Append, Max, MergeMap, Reducer, StateGraph};
use entelix_runnable::{Runnable, RunnableLambda};

#[derive(Clone, Debug, PartialEq)]
struct PlanState {
    log: Vec<String>,
    score: i32,
    tags: HashMap<String, String>,
}

#[derive(Clone, Debug)]
struct PlanDelta {
    new_log_entries: Vec<String>,
    score_candidate: i32,
    new_tags: HashMap<String, String>,
}

#[tokio::test]
async fn delta_node_merges_via_per_field_reducers() {
    let planner = RunnableLambda::new(|state: PlanState, _ctx| async move {
        Ok::<_, _>(PlanDelta {
            new_log_entries: vec![format!("plan@{}", state.score)],
            score_candidate: state.score + 5,
            new_tags: {
                let mut m = HashMap::new();
                m.insert("phase".into(), "plan".into());
                m
            },
        })
    });

    let executor = RunnableLambda::new(|state: PlanState, _ctx| async move {
        Ok::<_, _>(PlanDelta {
            new_log_entries: vec!["execute".into()],
            // Lower than current — Max should reject it.
            score_candidate: state.score - 3,
            new_tags: {
                let mut m = HashMap::new();
                m.insert("phase".into(), "execute".into());
                m.insert("worker".into(), "alpha".into());
                m
            },
        })
    });

    let graph = StateGraph::<PlanState>::new()
        .add_node_with("plan", planner, |state: PlanState, delta: PlanDelta| {
            Ok(merge(state, delta))
        })
        .add_node_with("execute", executor, |state: PlanState, delta: PlanDelta| {
            Ok(merge(state, delta))
        })
        .set_entry_point("plan")
        .add_finish_point("execute")
        .add_edge("plan", "execute")
        .compile()
        .unwrap();

    let initial = PlanState {
        log: vec!["seed".into()],
        score: 10,
        tags: {
            let mut m = HashMap::new();
            m.insert("origin".into(), "test".into());
            m
        },
    };
    let final_state = graph
        .invoke(initial, &ExecutionContext::new())
        .await
        .unwrap();

    // Append reducer: log accumulates across both nodes.
    assert_eq!(
        final_state.log,
        vec![
            "seed".to_owned(),
            "plan@10".to_owned(),
            "execute".to_owned()
        ]
    );
    // Max reducer: 15 (from plan) wins over 12 (planner output) and 7 (executor's lower candidate).
    assert_eq!(final_state.score, 15);
    // MergeMap reducer: right-biased — `phase` overwritten by execute, `origin` preserved, `worker` added.
    assert_eq!(final_state.tags.get("origin"), Some(&"test".to_owned()));
    assert_eq!(final_state.tags.get("phase"), Some(&"execute".to_owned()));
    assert_eq!(final_state.tags.get("worker"), Some(&"alpha".to_owned()));
}

/// Helper that demonstrates how `Reducer<T>` impls plug into a merger.
fn merge(state: PlanState, delta: PlanDelta) -> PlanState {
    PlanState {
        log: Append::<String>::new().reduce(state.log, delta.new_log_entries),
        score: Max::<i32>::new().reduce(state.score, delta.score_candidate),
        tags: MergeMap::<String, String>::new().reduce(state.tags, delta.new_tags),
    }
}

#[tokio::test]
async fn delta_node_coexists_with_full_state_node() {
    // Mix a delta-style node and a full-state replace node in one graph.
    let delta_node = RunnableLambda::new(|state: PlanState, _ctx| async move {
        Ok::<_, _>(PlanDelta {
            new_log_entries: vec![format!("delta@{}", state.score)],
            score_candidate: 100,
            new_tags: HashMap::new(),
        })
    });

    let replace_node = RunnableLambda::new(|mut state: PlanState, _ctx| async move {
        state.log.push("replace".into());
        Ok::<_, _>(state)
    });

    let graph = StateGraph::<PlanState>::new()
        .add_node_with("delta", delta_node, |state: PlanState, d: PlanDelta| {
            Ok(merge(state, d))
        })
        .add_node("replace", replace_node)
        .set_entry_point("delta")
        .add_finish_point("replace")
        .add_edge("delta", "replace")
        .compile()
        .unwrap();

    let initial = PlanState {
        log: Vec::new(),
        score: 0,
        tags: HashMap::new(),
    };
    let final_state = graph
        .invoke(initial, &ExecutionContext::new())
        .await
        .unwrap();

    assert_eq!(
        final_state.log,
        vec!["delta@0".to_owned(), "replace".to_owned()]
    );
    assert_eq!(final_state.score, 100);
}
