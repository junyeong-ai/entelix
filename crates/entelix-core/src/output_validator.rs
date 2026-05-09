//! Post-decode validators for typed structured output.
//!
//! [`OutputValidator<O>`] runs after `complete_typed_validated`
//! parses the model's response into `O`. Validators that detect a
//! semantic problem the JSON schema cannot catch — value out of
//! range, business-rule violation, cross-field invariant — return
//! [`crate::Error::ModelRetry`] with a corrective hint. The chat model's
//! retry loop catches the variant, reflects the hint to the model
//! as a user message (mirror of the schema-mismatch retry path,
//!), and re-invokes within the configured
//! [`ChatModelConfig::validation_retries`](crate::ChatModelConfig::validation_retries)
//! budget.
//!
//! # Closures fit too
//!
//! Any `Fn(&O) -> Result<()> + Send + Sync + 'static` satisfies
//! the trait via the blanket impl, so the common case is a
//! one-liner:
//!
//! ```ignore
//! let model = ChatModel::anthropic("…", "claude-…")
//!     .with_validation_retries(2);
//! let output: MyOutput = model.complete_typed_validated(
//!     messages,
//!     |out: &MyOutput| {
//!         if out.score >= 0 && out.score <= 100 {
//!             Ok(())
//!         } else {
//!             Err(Error::model_retry(
//!                 "score must be between 0 and 100".to_owned().for_llm(),
//!                 0,
//!             ))
//!         }
//!     },
//!     &ctx,
//! ).await?;
//! ```

use crate::error::Result;

/// Post-decode validator. Implementors return [`Error::ModelRetry`]
/// to signal the `ChatModel` retry loop should re-prompt the model
/// with corrective feedback. Other error variants bubble unchanged.
///
/// [`Error::ModelRetry`]: crate::Error::ModelRetry
pub trait OutputValidator<O>: Send + Sync + 'static
where
    O: Send + 'static,
{
    /// Inspect `output` and return `Ok(())` when it satisfies the
    /// operator's semantic constraints. Return `Err(Error::ModelRetry { … })`
    /// to push a corrective re-prompt onto the conversation;
    /// any other `Err` bubbles unchanged.
    fn validate(&self, output: &O) -> Result<()>;
}

impl<O, F> OutputValidator<O> for F
where
    O: Send + 'static,
    F: Fn(&O) -> Result<()> + Send + Sync + 'static,
{
    fn validate(&self, output: &O) -> Result<()> {
        self(output)
    }
}
