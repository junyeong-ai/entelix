//! `RunnableParallel` tests.

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use entelix_core::{Error, ExecutionContext, Result};
use entelix_runnable::{Runnable, RunnableLambda, RunnableParallel};

#[tokio::test]
async fn fans_out_to_two_branches() -> Result<()> {
    let double = RunnableLambda::new(|x: i32, _ctx| async move { Ok::<_, _>(x * 2) });
    let triple = RunnableLambda::new(|x: i32, _ctx| async move { Ok::<_, _>(x * 3) });

    let par = RunnableParallel::new()
        .branch("double", double)
        .branch("triple", triple);
    let out = par.invoke(5, &ExecutionContext::new()).await?;
    assert_eq!(out["double"], 10);
    assert_eq!(out["triple"], 15);
    assert_eq!(out.len(), 2);
    Ok(())
}

#[tokio::test]
async fn empty_parallel_returns_empty_map() -> Result<()> {
    let par: RunnableParallel<i32, i32> = RunnableParallel::new();
    assert!(par.is_empty());
    let out = par.invoke(5, &ExecutionContext::new()).await?;
    assert!(out.is_empty());
    Ok(())
}

#[tokio::test]
async fn first_failing_branch_aborts_others() {
    let ok = RunnableLambda::new(|x: i32, _ctx| async move { Ok::<_, _>(x) });
    let bad = RunnableLambda::new(|_x: i32, _ctx| async move {
        Err::<i32, _>(Error::invalid_request("nope"))
    });

    let par = RunnableParallel::new().branch("ok", ok).branch("bad", bad);
    let err = par.invoke(1, &ExecutionContext::new()).await.unwrap_err();
    assert!(matches!(err, Error::InvalidRequest(_)));
}

#[tokio::test]
async fn each_branch_receives_a_clone_of_the_input() -> Result<()> {
    // Verify Clone happens (not the same allocation): two branches mutate
    // their copy of an owned String differently.
    let pre = RunnableLambda::new(|s: String, _ctx| async move { Ok::<_, _>(format!("pre-{s}")) });
    let suf = RunnableLambda::new(|s: String, _ctx| async move { Ok::<_, _>(format!("{s}-suf")) });

    let par = RunnableParallel::new()
        .branch("pre", pre)
        .branch("suf", suf);
    let out = par
        .invoke("hello".to_owned(), &ExecutionContext::new())
        .await?;
    assert_eq!(out["pre"], "pre-hello");
    assert_eq!(out["suf"], "hello-suf");
    Ok(())
}

#[tokio::test]
async fn len_reports_branch_count() {
    let par: RunnableParallel<i32, i32> = RunnableParallel::new()
        .branch(
            "a",
            RunnableLambda::new(|x, _| async move { Ok::<_, _>(x) }),
        )
        .branch(
            "b",
            RunnableLambda::new(|x, _| async move { Ok::<_, _>(x) }),
        )
        .branch(
            "c",
            RunnableLambda::new(|x, _| async move { Ok::<_, _>(x) }),
        );
    assert_eq!(par.len(), 3);
}
