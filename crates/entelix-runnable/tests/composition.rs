//! Smoke tests proving `Runnable` composition through `.pipe()`.
//!
//! These tests are intentionally trivial — they validate the
//! contract, not production code. Real integration tests with
//! codecs/transports live in their own crates.

#![allow(clippy::unwrap_used)] // tests may panic; we want backtraces, not control flow

use entelix_core::{ExecutionContext, Result};
use entelix_runnable::{
    Runnable, RunnableExt, RunnableLambda, RunnablePassthrough, RunnableSequence,
};

#[tokio::test]
async fn pipe_chains_two_lambdas() {
    let ctx = ExecutionContext::new();
    let double = RunnableLambda::new(|x: i32, _ctx| async move { Ok::<_, _>(x * 2) });
    let add_one = RunnableLambda::new(|x: i32, _ctx| async move { Ok::<_, _>(x + 1) });

    let chain = double.pipe(add_one);
    let result: i32 = chain.invoke(3, &ctx).await.unwrap();
    assert_eq!(result, 7); // (3 * 2) + 1
}

#[tokio::test]
async fn passthrough_returns_input_unchanged() -> Result<()> {
    let ctx = ExecutionContext::new();
    let pt: RunnablePassthrough = RunnablePassthrough;
    let out: String = pt.invoke("hello".to_owned(), &ctx).await?;
    assert_eq!(out, "hello");
    Ok(())
}

#[tokio::test]
async fn batch_default_runs_sequentially() -> Result<()> {
    let ctx = ExecutionContext::new();
    let triple = RunnableLambda::new(|x: i32, _ctx| async move { Ok::<_, _>(x * 3) });
    let outs = triple.batch(vec![1, 2, 3], &ctx).await?;
    assert_eq!(outs, vec![3, 6, 9]);
    Ok(())
}

#[tokio::test]
async fn batch_short_circuits_on_cancellation() {
    let ctx = ExecutionContext::new();
    let lambda = RunnableLambda::new(|x: i32, _ctx| async move { Ok::<_, _>(x * 2) });
    ctx.cancellation().cancel();
    let res = lambda.batch(vec![1, 2, 3], &ctx).await;
    assert!(matches!(res, Err(entelix_core::Error::Cancelled)));
}

#[tokio::test]
async fn three_stage_pipe_typechecks() -> Result<()> {
    let ctx = ExecutionContext::new();
    let to_string = RunnableLambda::new(|x: i32, _ctx| async move { Ok::<_, _>(format!("{x}")) });
    let to_len = RunnableLambda::new(|s: String, _ctx| async move { Ok::<_, _>(s.len()) });
    let to_bool = RunnableLambda::new(|n: usize, _ctx| async move { Ok::<_, _>(n > 0) });

    // Verify .pipe() chains across heterogeneous types.
    let chain: RunnableSequence<i32, usize, bool> = to_string.pipe(to_len).pipe(to_bool);
    let out: bool = chain.invoke(42, &ctx).await?;
    assert!(out);
    Ok(())
}
