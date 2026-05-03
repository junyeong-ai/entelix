//! `AnyRunnable` + `erase` tests — F12 mitigation surface.

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use entelix_core::{Error, ExecutionContext, Result};
use entelix_runnable::{
    AnyRunnable, AnyRunnableHandle, RunnableLambda, RunnablePassthrough, erase,
};

#[tokio::test]
async fn erase_makes_typed_runnable_dispatchable_via_value() -> Result<()> {
    let typed = RunnableLambda::new(|x: i32, _ctx| async move { Ok::<_, _>(x * 2) });
    let erased: AnyRunnableHandle = erase(typed);

    let out = erased
        .invoke_any(serde_json::json!(21), &ExecutionContext::new())
        .await?;
    assert_eq!(out, serde_json::json!(42));
    Ok(())
}

#[tokio::test]
async fn registry_pattern_collects_heterogeneous_runnables() -> Result<()> {
    // Two runnables with different I/O types share one Vec.
    let int_doubler = erase(RunnableLambda::new(|x: i32, _ctx| async move {
        Ok::<_, _>(x * 2)
    }));
    let string_upper = erase(RunnableLambda::new(|s: String, _ctx| async move {
        Ok::<_, _>(s.to_uppercase())
    }));

    let registry: Vec<AnyRunnableHandle> = vec![int_doubler, string_upper];
    assert_eq!(
        registry[0]
            .invoke_any(serde_json::json!(5), &ExecutionContext::new())
            .await?,
        serde_json::json!(10)
    );
    assert_eq!(
        registry[1]
            .invoke_any(serde_json::json!("hi"), &ExecutionContext::new())
            .await?,
        serde_json::json!("HI")
    );
    Ok(())
}

#[tokio::test]
async fn invalid_input_shape_surfaces_serde_error() {
    let erased = erase(RunnableLambda::new(
        |x: i32, _ctx| async move { Ok::<_, _>(x) },
    ));
    // Pass a string where an integer is required.
    let err = erased
        .invoke_any(serde_json::json!("not a number"), &ExecutionContext::new())
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Serde(_)));
}

#[tokio::test]
async fn passthrough_can_be_erased_and_round_trips_value() -> Result<()> {
    let pt: RunnablePassthrough = RunnablePassthrough;
    let erased: AnyRunnableHandle = erase::<_, serde_json::Value, serde_json::Value>(pt);

    let payload = serde_json::json!({ "k": "v" });
    let out = erased
        .invoke_any(payload.clone(), &ExecutionContext::new())
        .await?;
    assert_eq!(out, payload);
    Ok(())
}

#[tokio::test]
async fn erased_inherits_runnable_name() {
    let erased = erase(RunnableLambda::new(
        |x: i32, _ctx| async move { Ok::<_, _>(x) },
    ));
    let n = AnyRunnable::name(erased.as_ref());
    // Default Runnable::name is the type-name; lambda's wrapper type should
    // appear somewhere in the string.
    assert!(n.contains("RunnableLambda"), "got '{n}'");
}
