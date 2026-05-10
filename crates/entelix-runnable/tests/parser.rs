//! `JsonOutputParser<T>` tests — extracts text from `Message`, parses as
//! JSON into `T`.

#![allow(clippy::unwrap_used)]

use entelix_core::ir::{ContentPart, Message, Role};
use entelix_core::{Error, ExecutionContext, Result};
use entelix_runnable::{JsonOutputParser, Runnable, RunnableExt, RunnableLambda};
use serde::Deserialize;

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct Reply {
    answer: String,
    confidence: u8,
}

fn assistant_text(s: &str) -> Message {
    Message::new(Role::Assistant, vec![ContentPart::text(s)])
}

#[tokio::test]
async fn parses_well_formed_json() -> Result<()> {
    let parser = JsonOutputParser::<Reply>::new();
    let msg = assistant_text(r#"{"answer":"42","confidence":99}"#);
    let out: Reply = parser.invoke(msg, &ExecutionContext::new()).await?;
    assert_eq!(
        out,
        Reply {
            answer: "42".into(),
            confidence: 99,
        }
    );
    Ok(())
}

#[tokio::test]
async fn concatenates_multiple_text_parts() -> Result<()> {
    let parser = JsonOutputParser::<Reply>::new();
    let msg = Message::new(
        Role::Assistant,
        vec![
            ContentPart::text(r#"{"answer":"#),
            ContentPart::text(r#""ok","confidence":1}"#),
        ],
    );
    let out: Reply = parser.invoke(msg, &ExecutionContext::new()).await?;
    assert_eq!(out.answer, "ok");
    Ok(())
}

#[tokio::test]
async fn ignores_non_text_parts() -> Result<()> {
    let parser = JsonOutputParser::<Reply>::new();
    let msg = Message::new(
        Role::Assistant,
        vec![
            ContentPart::ToolUse {
                id: "x".into(),
                name: "calc".into(),
                input: serde_json::json!({}),
                provider_echoes: Vec::new(),
            },
            ContentPart::text(r#"{"answer":"ok","confidence":1}"#),
        ],
    );
    let out: Reply = parser.invoke(msg, &ExecutionContext::new()).await?;
    assert_eq!(out.answer, "ok");
    Ok(())
}

#[tokio::test]
async fn empty_message_returns_invalid_request() {
    let parser = JsonOutputParser::<Reply>::new();
    let msg = Message::new(Role::Assistant, vec![]);
    let err = parser
        .invoke(msg, &ExecutionContext::new())
        .await
        .unwrap_err();
    assert!(matches!(err, Error::InvalidRequest(_)));
}

#[tokio::test]
async fn malformed_json_surfaces_serde_error() {
    let parser = JsonOutputParser::<Reply>::new();
    let msg = assistant_text("not json at all");
    let err = parser
        .invoke(msg, &ExecutionContext::new())
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Serde(_)));
}

#[tokio::test]
async fn pipes_after_a_lambda_producing_message() -> Result<()> {
    // Demonstrate composition: lambda produces a Message, parser consumes it.
    let stub = RunnableLambda::new(|s: String, _ctx| async move { Ok::<_, _>(assistant_text(&s)) });
    let parser = JsonOutputParser::<Reply>::new();
    let chain = stub.pipe(parser);

    let out = chain
        .invoke(
            r#"{"answer":"piped","confidence":7}"#.to_owned(),
            &ExecutionContext::new(),
        )
        .await?;
    assert_eq!(out.answer, "piped");
    Ok(())
}
