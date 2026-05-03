//! `RunnableRouter` tests.

#![allow(clippy::unwrap_used)]

use entelix_core::{Error, ExecutionContext, Result};
use entelix_runnable::{Runnable, RunnableLambda, RunnableRouter};

fn doubler() -> RunnableLambda<i32, i32> {
    RunnableLambda::new(|x: i32, _ctx| async move { Ok::<_, _>(x * 2) })
}

fn negator() -> RunnableLambda<i32, i32> {
    RunnableLambda::new(|x: i32, _ctx| async move { Ok::<_, _>(-x) })
}

fn zero() -> RunnableLambda<i32, i32> {
    RunnableLambda::new(|_x: i32, _ctx| async move { Ok::<_, _>(0) })
}

#[tokio::test]
async fn first_matching_predicate_wins() -> Result<()> {
    let router = RunnableRouter::<i32, i32>::new()
        .route(|x| *x > 10, doubler())
        .route(|x| *x < 0, negator());

    assert_eq!(router.invoke(20, &ExecutionContext::new()).await?, 40);
    assert_eq!(router.invoke(-5, &ExecutionContext::new()).await?, 5);
    Ok(())
}

#[tokio::test]
async fn predicates_evaluate_in_registration_order() -> Result<()> {
    // Both predicates would match `5`, but the doubler is registered first.
    let router = RunnableRouter::<i32, i32>::new()
        .route(|_x| true, doubler())
        .route(|_x| true, negator());

    assert_eq!(router.invoke(5, &ExecutionContext::new()).await?, 10);
    Ok(())
}

#[tokio::test]
async fn fallback_runs_when_no_predicate_matches() -> Result<()> {
    let router = RunnableRouter::<i32, i32>::new()
        .route(|x| *x > 100, doubler())
        .fallback(zero());

    assert_eq!(router.invoke(5, &ExecutionContext::new()).await?, 0);
    Ok(())
}

#[tokio::test]
async fn no_match_no_fallback_returns_invalid_request() {
    let router = RunnableRouter::<i32, i32>::new().route(|x| *x > 1000, doubler());
    let err = router
        .invoke(1, &ExecutionContext::new())
        .await
        .unwrap_err();
    assert!(matches!(err, Error::InvalidRequest(_)));
}

#[tokio::test]
async fn empty_router_with_only_fallback_works() -> Result<()> {
    let router = RunnableRouter::<i32, i32>::new().fallback(doubler());
    assert!(router.is_empty());
    assert_eq!(router.invoke(7, &ExecutionContext::new()).await?, 14);
    Ok(())
}

#[tokio::test]
async fn fallback_setter_replaces_previous() -> Result<()> {
    let router = RunnableRouter::<i32, i32>::new()
        .fallback(doubler())
        .fallback(zero());
    assert_eq!(router.invoke(7, &ExecutionContext::new()).await?, 0);
    Ok(())
}

#[tokio::test]
async fn len_counts_routes_only() {
    let router = RunnableRouter::<i32, i32>::new()
        .route(|_x| true, doubler())
        .route(|_x| true, negator())
        .fallback(zero());
    assert_eq!(router.len(), 2); // fallback excluded
}
