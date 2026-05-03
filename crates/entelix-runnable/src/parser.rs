//! Output parsers — `Runnable`s that turn model output (`Message`) into
//! strongly-typed Rust values.
//!
//! Surface:
//! - [`JsonOutputParser<T>`] — direct serde-JSON deserializer.
//! - [`RetryParser<I, O, P>`] — retries an inner `Runnable<I, O>` on failure
//!   up to a configurable cap.
//! - [`FixingOutputParser<O, P, M>`] — on parse failure, asks an inner
//!   chat model (`Runnable<Vec<Message>, Message>`) to repair the
//!   malformed output, then re-runs the inner parser.

use std::marker::PhantomData;

use entelix_core::ir::{ContentPart, Message};
use entelix_core::{Error, ExecutionContext, Result};
use serde::de::DeserializeOwned;

use crate::runnable::Runnable;

/// Parses an assistant `Message` as JSON into `T`.
///
/// The parser concatenates every [`ContentPart::Text`] in the message and
/// runs `serde_json::from_str` on the result. Non-text parts are ignored;
/// callers that need stricter handling can wrap this with a
/// [`crate::RunnableLambda`] preprocessor.
///
/// ```ignore
/// use entelix_runnable::JsonOutputParser;
/// use serde::Deserialize;
///
/// #[derive(Deserialize)]
/// struct Reply { answer: String }
///
/// let parser = JsonOutputParser::<Reply>::new();
/// let chain = prompt.pipe(model).pipe(parser);
/// ```
pub struct JsonOutputParser<T> {
    _phantom: PhantomData<fn() -> T>,
}

impl<T> JsonOutputParser<T> {
    /// Build a fresh parser. Consumers usually let type inference at the
    /// call site pin `T`.
    pub const fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T> Default for JsonOutputParser<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Clone for JsonOutputParser<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for JsonOutputParser<T> {}

#[async_trait::async_trait]
impl<T> Runnable<Message, T> for JsonOutputParser<T>
where
    T: DeserializeOwned + Send + 'static,
{
    async fn invoke(&self, input: Message, _ctx: &ExecutionContext) -> Result<T> {
        let mut text = String::new();
        for part in &input.content {
            if let ContentPart::Text { text: t, .. } = part {
                text.push_str(t);
            }
        }
        if text.is_empty() {
            return Err(Error::invalid_request(
                "JsonOutputParser: message contains no text parts",
            ));
        }
        Ok(serde_json::from_str(&text)?)
    }
}

/// Default retry budget for [`RetryParser`] when no explicit cap is given.
pub const DEFAULT_PARSER_RETRIES: usize = 3;

/// Wraps any `Runnable<I, O>` with a fixed retry budget.
///
/// On failure the parser re-invokes the inner runnable up to
/// `max_retries` additional times before surfacing the last error.
/// `I: Clone` is required because each attempt consumes the input.
/// Cooperative cancellation is honoured between attempts.
pub struct RetryParser<I, O, P>
where
    I: Clone + Send + 'static,
    O: Send + 'static,
    P: Runnable<I, O>,
{
    inner: P,
    max_retries: usize,
    _phantom: PhantomData<fn() -> (I, O)>,
}

impl<I, O, P> RetryParser<I, O, P>
where
    I: Clone + Send + 'static,
    O: Send + 'static,
    P: Runnable<I, O>,
{
    /// Build a `RetryParser` with the default retry budget
    /// ([`DEFAULT_PARSER_RETRIES`]).
    pub const fn new(inner: P) -> Self {
        Self {
            inner,
            max_retries: DEFAULT_PARSER_RETRIES,
            _phantom: PhantomData,
        }
    }

    /// Override the retry budget. `0` means "no retries" — exactly one
    /// attempt — which is functionally equivalent to invoking `inner`
    /// directly.
    #[must_use]
    pub const fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }
}

#[async_trait::async_trait]
impl<I, O, P> Runnable<I, O> for RetryParser<I, O, P>
where
    I: Clone + Send + 'static,
    O: Send + 'static,
    P: Runnable<I, O> + 'static,
{
    async fn invoke(&self, input: I, ctx: &ExecutionContext) -> Result<O> {
        let mut last_err: Option<Error> = None;
        let total_attempts = self.max_retries.saturating_add(1);
        for _ in 0..total_attempts {
            if ctx.is_cancelled() {
                return Err(Error::Cancelled);
            }
            match self.inner.invoke(input.clone(), ctx).await {
                Ok(o) => return Ok(o),
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err
            .unwrap_or_else(|| Error::invalid_request("RetryParser: zero attempts configured")))
    }
}

/// Repair-on-failure wrapper around a `Runnable<Message, O>`.
///
/// When parsing fails, an inner chat model is asked to repair the
/// malformed output; the fixed message then feeds back through the
/// parser. Up to `max_retries` repair cycles are attempted before the
/// last error is surfaced.
///
/// Typical use: pipe a model into a `JsonOutputParser`, then wrap the
/// parser with a `FixingOutputParser` that delegates repair to the same
/// (or a smaller) model.
pub struct FixingOutputParser<O, P, M>
where
    O: Send + 'static,
    P: Runnable<Message, O>,
    M: Runnable<Vec<Message>, Message>,
{
    inner: P,
    fixer: M,
    max_retries: usize,
    instructions: String,
    _phantom: PhantomData<fn() -> O>,
}

impl<O, P, M> FixingOutputParser<O, P, M>
where
    O: Send + 'static,
    P: Runnable<Message, O>,
    M: Runnable<Vec<Message>, Message>,
{
    /// Build a `FixingOutputParser` with the default retry budget and
    /// repair instructions.
    pub fn new(inner: P, fixer: M) -> Self {
        Self {
            inner,
            fixer,
            max_retries: DEFAULT_PARSER_RETRIES,
            instructions: DEFAULT_FIX_INSTRUCTIONS.to_owned(),
            _phantom: PhantomData,
        }
    }

    /// Override the retry budget. `0` means the parser runs once and
    /// surfaces any error without invoking the fixer.
    #[must_use]
    pub const fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Override the fixer's system instructions. The default tells the
    /// model to "return only the corrected JSON".
    #[must_use]
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = instructions.into();
        self
    }

    fn build_fix_prompt(&self, original: &Message, error: &Error) -> Vec<Message> {
        let original_text = collect_text(original);
        let body = format!(
            "The previous output failed to parse with error:\n  {error}\n\nOriginal output:\n{original_text}\n\nReturn only the corrected output."
        );
        vec![
            Message::system(self.instructions.clone()),
            Message::user(body),
        ]
    }
}

const DEFAULT_FIX_INSTRUCTIONS: &str = "You are a strict output corrector. Given malformed output and the parser error, \
     return only the corrected version with no commentary, no preamble, and no markdown fences.";

fn collect_text(message: &Message) -> String {
    let mut s = String::new();
    for part in &message.content {
        if let ContentPart::Text { text, .. } = part {
            s.push_str(text);
        }
    }
    s
}

#[async_trait::async_trait]
impl<O, P, M> Runnable<Message, O> for FixingOutputParser<O, P, M>
where
    O: Send + 'static,
    P: Runnable<Message, O> + 'static,
    M: Runnable<Vec<Message>, Message> + 'static,
{
    async fn invoke(&self, input: Message, ctx: &ExecutionContext) -> Result<O> {
        let mut current = input;
        let mut last_err: Option<Error> = None;
        let total_attempts = self.max_retries.saturating_add(1);
        for attempt in 0..total_attempts {
            if ctx.is_cancelled() {
                return Err(Error::Cancelled);
            }
            match self.inner.invoke(current.clone(), ctx).await {
                Ok(o) => return Ok(o),
                Err(e) => {
                    let last_attempt = attempt.saturating_add(1) == total_attempts;
                    if last_attempt {
                        last_err = Some(e);
                        break;
                    }
                    let fix_prompt = self.build_fix_prompt(&current, &e);
                    last_err = Some(e);
                    current = self.fixer.invoke(fix_prompt, ctx).await?;
                }
            }
        }
        Err(last_err.unwrap_or_else(|| {
            Error::invalid_request("FixingOutputParser: zero attempts configured")
        }))
    }
}
