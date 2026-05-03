//! `RetryParser<I, O, P>` + `FixingOutputParser<O, P, M>` tests.
//! Validate retry budget exhaustion, success after a flaky inner
//! parser, cancellation mid-retry, and the fix-prompt repair cycle.

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use entelix_core::ir::{ContentPart, Message};
use entelix_core::{Error, ExecutionContext, Result};
use entelix_runnable::{FixingOutputParser, JsonOutputParser, RetryParser, Runnable};

/// Inner parser that fails the first `fail_count` invocations then
/// succeeds.
struct FlakyParser {
    invocations: Arc<AtomicUsize>,
    fail_count: usize,
}

#[async_trait]
impl Runnable<i32, i32> for FlakyParser {
    async fn invoke(&self, input: i32, _ctx: &ExecutionContext) -> Result<i32> {
        let n = self.invocations.fetch_add(1, Ordering::SeqCst);
        if n < self.fail_count {
            Err(Error::invalid_request(format!("flaky #{n}")))
        } else {
            Ok(input * 2)
        }
    }
}

#[tokio::test]
async fn retry_succeeds_after_failures_within_budget() -> Result<()> {
    let invocations = Arc::new(AtomicUsize::new(0));
    let inner = FlakyParser {
        invocations: invocations.clone(),
        fail_count: 2,
    };
    let parser: RetryParser<i32, i32, _> = RetryParser::new(inner).with_max_retries(3);
    let out = parser.invoke(7, &ExecutionContext::new()).await?;
    assert_eq!(out, 14);
    assert_eq!(invocations.load(Ordering::SeqCst), 3); // 2 fails + 1 success
    Ok(())
}

#[tokio::test]
async fn retry_exhausts_and_surfaces_last_error() {
    let invocations = Arc::new(AtomicUsize::new(0));
    let inner = FlakyParser {
        invocations: invocations.clone(),
        fail_count: 99, // always fail
    };
    let parser: RetryParser<i32, i32, _> = RetryParser::new(inner).with_max_retries(2);
    let err = parser
        .invoke(1, &ExecutionContext::new())
        .await
        .unwrap_err();
    assert!(matches!(err, Error::InvalidRequest(_)));
    // total_attempts = max_retries + 1 = 3
    assert_eq!(invocations.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn retry_zero_budget_runs_exactly_once() -> Result<()> {
    let invocations = Arc::new(AtomicUsize::new(0));
    let inner = FlakyParser {
        invocations: invocations.clone(),
        fail_count: 0,
    };
    let parser: RetryParser<i32, i32, _> = RetryParser::new(inner).with_max_retries(0);
    let out = parser.invoke(3, &ExecutionContext::new()).await?;
    assert_eq!(out, 6);
    assert_eq!(invocations.load(Ordering::SeqCst), 1);
    Ok(())
}

#[tokio::test]
async fn retry_respects_cancellation_between_attempts() {
    let invocations = Arc::new(AtomicUsize::new(0));
    let inner = FlakyParser {
        invocations: invocations.clone(),
        fail_count: 99,
    };
    let parser: RetryParser<i32, i32, _> = RetryParser::new(inner).with_max_retries(5);
    let ctx = ExecutionContext::new();
    ctx.cancellation().cancel(); // cancel before any attempt
    let err = parser.invoke(1, &ctx).await.unwrap_err();
    assert!(matches!(err, Error::Cancelled));
    assert_eq!(invocations.load(Ordering::SeqCst), 0);
}

// ─── FixingOutputParser ──────────────────────────────────────────────────

/// Stub fixer that records the prompt it received and returns a
/// pre-canned valid JSON message.
struct StubFixer {
    canned_reply: String,
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl Runnable<Vec<Message>, Message> for StubFixer {
    async fn invoke(&self, _input: Vec<Message>, _ctx: &ExecutionContext) -> Result<Message> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(Message::new(
            entelix_core::ir::Role::Assistant,
            vec![ContentPart::text(self.canned_reply.clone())],
        ))
    }
}

#[derive(Debug, serde::Deserialize, PartialEq, Eq)]
struct Answer {
    answer: String,
}

#[tokio::test]
async fn fixing_repairs_malformed_json_then_succeeds() -> Result<()> {
    let inner = JsonOutputParser::<Answer>::new();
    let fixer_calls = Arc::new(AtomicUsize::new(0));
    let fixer = StubFixer {
        canned_reply: r#"{"answer":"fixed"}"#.to_owned(),
        calls: fixer_calls.clone(),
    };
    let parser = FixingOutputParser::new(inner, fixer).with_max_retries(2);

    let bad = Message::new(
        entelix_core::ir::Role::Assistant,
        vec![ContentPart::text("not json")],
    );
    let out: Answer = parser.invoke(bad, &ExecutionContext::new()).await?;
    assert_eq!(out.answer, "fixed");
    assert_eq!(fixer_calls.load(Ordering::SeqCst), 1);
    Ok(())
}

#[tokio::test]
async fn fixing_exhausts_when_fixer_keeps_returning_garbage() {
    let inner = JsonOutputParser::<Answer>::new();
    let fixer_calls = Arc::new(AtomicUsize::new(0));
    let fixer = StubFixer {
        canned_reply: "still not json".to_owned(),
        calls: fixer_calls.clone(),
    };
    let parser = FixingOutputParser::new(inner, fixer).with_max_retries(2);

    let bad = Message::new(
        entelix_core::ir::Role::Assistant,
        vec![ContentPart::text("not json")],
    );
    let err = parser
        .invoke(bad, &ExecutionContext::new())
        .await
        .unwrap_err();
    // Final attempt's parse error is the surfaced error.
    assert!(matches!(err, Error::Serde(_)));
    // total_attempts = 3 → fixer is called twice (between attempts).
    assert_eq!(fixer_calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn fixing_passes_through_when_initial_parse_succeeds() -> Result<()> {
    let inner = JsonOutputParser::<Answer>::new();
    let fixer_calls = Arc::new(AtomicUsize::new(0));
    let fixer = StubFixer {
        canned_reply: r#"{"answer":"unused"}"#.to_owned(),
        calls: fixer_calls.clone(),
    };
    let parser = FixingOutputParser::new(inner, fixer).with_max_retries(3);

    let good = Message::new(
        entelix_core::ir::Role::Assistant,
        vec![ContentPart::text(r#"{"answer":"first-try"}"#)],
    );
    let out: Answer = parser.invoke(good, &ExecutionContext::new()).await?;
    assert_eq!(out.answer, "first-try");
    assert_eq!(fixer_calls.load(Ordering::SeqCst), 0);
    Ok(())
}

#[tokio::test]
async fn fixing_respects_cancellation_before_first_attempt() {
    let inner = JsonOutputParser::<Answer>::new();
    let fixer_calls = Arc::new(AtomicUsize::new(0));
    let fixer = StubFixer {
        canned_reply: r#"{"answer":"x"}"#.to_owned(),
        calls: fixer_calls.clone(),
    };
    let parser = FixingOutputParser::new(inner, fixer).with_max_retries(3);
    let ctx = ExecutionContext::new();
    ctx.cancellation().cancel();
    let bad = Message::new(
        entelix_core::ir::Role::Assistant,
        vec![ContentPart::text("not json")],
    );
    let err = parser.invoke(bad, &ctx).await.unwrap_err();
    assert!(matches!(err, Error::Cancelled));
    assert_eq!(fixer_calls.load(Ordering::SeqCst), 0);
}
