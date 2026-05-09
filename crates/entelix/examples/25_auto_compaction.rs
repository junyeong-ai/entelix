//! Example 25 — auto-compaction wiring.
//!
//! Build: `cargo run --example 25_auto_compaction -p entelix`
//! Run: same.
//!
//! Demonstrates `MessageRunnableCompactionExt::with_compaction(.)` —
//! the operator-facing one-liner that wraps any
//! `Runnable<Vec<Message>, Message>` with threshold-driven
//! compaction. Below the threshold the wrapper is a transparent
//! delegate; at or above, it routes the conversation through a
//! `Compactor` and forwards the trimmed messages to the model.
//!
//! Wires `HeadDropCompactor` (the reference "drop-oldest" strategy)
//! against a deterministic in-process model so the example runs in
//! CI without API keys. The model echoes the count of messages it
//! actually sees, making the trim observable.

#![allow(clippy::print_stdout)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use entelix::prelude::*;
use entelix::{Compactor, HeadDropCompactor, MessageRunnableCompactionExt};

/// Deterministic stand-in for a real `ChatModel`. Records the message
/// count it received so we can assert compaction triggered.
struct CountingModel {
    last_observed: Arc<AtomicUsize>,
}

#[async_trait]
impl Runnable<Vec<Message>, Message> for CountingModel {
    async fn invoke(&self, input: Vec<Message>, _ctx: &ExecutionContext) -> Result<Message> {
        self.last_observed.store(input.len(), Ordering::SeqCst);
        Ok(Message::new(
            Role::Assistant,
            vec![ContentPart::text(format!("saw {} messages", input.len()))],
        ))
    }
}

fn turn_pair(user_text: &str, assistant_text: &str) -> Vec<Message> {
    vec![
        Message::new(Role::User, vec![ContentPart::text(user_text)]),
        Message::new(Role::Assistant, vec![ContentPart::text(assistant_text)]),
    ]
}

#[tokio::main]
async fn main() -> Result<()> {
    let observed = Arc::new(AtomicUsize::new(0));
    let model = CountingModel {
        last_observed: observed.clone(),
    };

    // Wrap the model with auto-compaction. Threshold (chars) sized so
    // the third turn pair pushes the conversation over budget.
    let compactor: Arc<dyn Compactor> = Arc::new(HeadDropCompactor);
    let model = model.with_compaction(compactor, 80);

    // ── Pass 1 — short conversation under threshold ──────────────────
    let short = turn_pair("hi", "hello");
    model.invoke(short.clone(), &ExecutionContext::new()).await?;
    let observed_short = observed.load(Ordering::SeqCst);
    assert_eq!(
        observed_short,
        short.len(),
        "below threshold: model sees the original input unchanged"
    );

    // ── Pass 2 — long conversation triggers compaction ───────────────
    let long: Vec<Message> = (0..6)
        .flat_map(|i| {
            turn_pair(
                &format!("question number {i} with extra padding text"),
                &format!("response number {i} with extra padding text"),
            )
        })
        .collect();
    model.invoke(long.clone(), &ExecutionContext::new()).await?;
    let observed_long = observed.load(Ordering::SeqCst);
    assert!(
        observed_long < long.len(),
        "above threshold: compaction must trim — observed {observed_long}, input was {}",
        long.len()
    );

    println!(
        "auto-compaction wired: short pass observed {observed_short} messages, \
         long pass trimmed to {observed_long}/{}",
        long.len()
    );
    Ok(())
}
