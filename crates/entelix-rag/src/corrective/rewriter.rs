//! Query rewriter — `(original_query, failed_attempts) → corrected_query`.
//!
//! When the [`crate::RetrievalGrader`] tells the corrective recipe
//! the retrieved batch is mostly off-topic, the recipe asks the
//! rewriter for a corrected query and re-runs retrieval. The
//! rewriter sees the original query plus every prior failed
//! attempt so it doesn't re-emit the same wording the failed
//! retrieval already used.
//!
//! This module ships the *rewriter primitive* (trait + reference
//! LLM impl). Operators with a heuristic rewriter (HyDE,
//! query-expansion via thesaurus, BM25 keyword extraction) write
//! their own [`QueryRewriter`] and bypass the LLM cost entirely.

use std::sync::Arc;

use async_trait::async_trait;
use entelix_core::ir::{ContentPart, Message, Role};
use entelix_core::{Error, ExecutionContext, Result};
use entelix_runnable::Runnable;

/// Async trait the corrective-RAG recipe calls when retrieval
/// quality requires another attempt with a different query.
/// Implementations may be LLM-driven, heuristic
/// (query-expansion / synonym-bag), classifier-routed, or any
/// hybrid — the recipe takes whatever string comes back and
/// re-runs retrieval with it.
#[async_trait]
pub trait QueryRewriter: Send + Sync {
    /// Stable rewriter identifier — surfaces in audit dashboards
    /// alongside the per-attempt query string.
    fn name(&self) -> &'static str;

    /// Produce a corrected query. `original` is the user's
    /// untouched first attempt; `previous_attempts` is every
    /// failed retry since (in chronological order, oldest first)
    /// so the rewriter can avoid emitting an attempt the recipe
    /// has already failed on. An empty `previous_attempts` slice
    /// means this is the first rewrite — the rewriter sees only
    /// the original.
    async fn rewrite(
        &self,
        original: &str,
        previous_attempts: &[String],
        ctx: &ExecutionContext,
    ) -> Result<String>;
}

/// Default instruction prepended to every model call. Verbatim
/// matches the CRAG-paper rewriter framing — the model produces
/// one corrected query string, no surrounding explanation.
pub const DEFAULT_REWRITER_INSTRUCTION: &str = "\
You are a query rewriter. Given the user's original query and any prior failed attempts \
(retrieval did not return useful results), produce a single corrected query that captures \
the user's intent in different words. Reply with only the corrected query string — no \
quotes, no explanation, no surrounding text.";

/// Stable rewriter identifier for [`LlmQueryRewriter`].
const LLM_REWRITER_NAME: &str = "llm-query-rewriter";

/// Builder for [`LlmQueryRewriter`].
pub struct LlmQueryRewriterBuilder<M> {
    model: Arc<M>,
    instruction: String,
}

impl<M> LlmQueryRewriterBuilder<M>
where
    M: Runnable<Vec<Message>, Message> + 'static,
{
    /// Override the operator-facing instruction. Default matches
    /// [`DEFAULT_REWRITER_INSTRUCTION`] verbatim.
    #[must_use]
    pub fn with_instruction(mut self, instruction: impl Into<String>) -> Self {
        self.instruction = instruction.into();
        self
    }

    /// Finalise into a runnable rewriter.
    #[must_use]
    pub fn build(self) -> LlmQueryRewriter<M> {
        LlmQueryRewriter {
            model: self.model,
            instruction: Arc::from(self.instruction),
        }
    }
}

/// Reference LLM-driven [`QueryRewriter`]. Asks the supplied
/// `Runnable<Vec<Message>, Message>` model for a corrected
/// query, then trims surrounding whitespace and quote marks.
pub struct LlmQueryRewriter<M> {
    model: Arc<M>,
    instruction: Arc<str>,
}

impl<M> LlmQueryRewriter<M>
where
    M: Runnable<Vec<Message>, Message> + 'static,
{
    /// Start a builder bound to the supplied model.
    #[must_use]
    pub fn builder(model: Arc<M>) -> LlmQueryRewriterBuilder<M> {
        LlmQueryRewriterBuilder {
            model,
            instruction: DEFAULT_REWRITER_INSTRUCTION.to_owned(),
        }
    }

    /// Build the user message that frames one rewrite call. Three
    /// text parts: instruction, original query, prior attempts.
    /// Single-message shape so any
    /// `Runnable<Vec<Message>, Message>` impl executes without
    /// recipe-side wiring.
    fn build_prompt(&self, original: &str, previous_attempts: &[String]) -> Vec<Message> {
        let prior_block = if previous_attempts.is_empty() {
            "(none)".to_owned()
        } else {
            previous_attempts
                .iter()
                .enumerate()
                .map(|(idx, attempt)| format!("attempt {}: {attempt}", idx + 1))
                .collect::<Vec<_>>()
                .join("\n")
        };
        vec![Message::new(
            Role::User,
            vec![
                ContentPart::text(self.instruction.to_string()),
                ContentPart::text(format!("<original>\n{original}\n</original>")),
                ContentPart::text(format!(
                    "<failed_attempts>\n{prior_block}\n</failed_attempts>"
                )),
            ],
        )]
    }
}

impl<M> Clone for LlmQueryRewriter<M> {
    fn clone(&self) -> Self {
        Self {
            model: Arc::clone(&self.model),
            instruction: Arc::clone(&self.instruction),
        }
    }
}

impl<M> std::fmt::Debug for LlmQueryRewriter<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmQueryRewriter").finish_non_exhaustive()
    }
}

#[async_trait]
impl<M> QueryRewriter for LlmQueryRewriter<M>
where
    M: Runnable<Vec<Message>, Message> + 'static,
{
    fn name(&self) -> &'static str {
        LLM_REWRITER_NAME
    }

    async fn rewrite(
        &self,
        original: &str,
        previous_attempts: &[String],
        ctx: &ExecutionContext,
    ) -> Result<String> {
        let prompt = self.build_prompt(original, previous_attempts);
        let reply = self.model.invoke(prompt, ctx).await?;
        let cleaned = clean_reply(&reply);
        if cleaned.is_empty() {
            return Err(Error::invalid_request(
                "LlmQueryRewriter: model returned no text — rewrite failed",
            ));
        }
        Ok(cleaned)
    }
}

/// Strip surrounding whitespace + quote marks from the model's
/// reply. Pulls every `Text` part out of the message and
/// concatenates with single newlines; non-text parts (tool-use,
/// image-output) are skipped — a rewriter that emits tool calls
/// is a misconfiguration we silently degrade rather than fail
/// on.
fn clean_reply(message: &Message) -> String {
    let mut buf = String::new();
    for part in &message.content {
        if let ContentPart::Text { text, .. } = part {
            if !buf.is_empty() {
                buf.push('\n');
            }
            buf.push_str(text);
        }
    }
    let trimmed = buf.trim();
    // Strip a single layer of surrounding quotes the model might
    // emit despite the instruction. Done after `trim` so
    // whitespace inside the quote pair survives.
    let stripped = trimmed
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .or_else(|| {
            trimmed
                .strip_prefix('\'')
                .and_then(|s| s.strip_suffix('\''))
        })
        .unwrap_or(trimmed);
    stripped.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn assistant(text: &str) -> Message {
        Message::new(Role::Assistant, vec![ContentPart::text(text)])
    }

    /// Scripted model — pops next reply per invocation, exposes
    /// the prompts every call observed for prompt-shape pinning.
    struct ScriptedModel {
        script: Mutex<Vec<Result<Message>>>,
        observed: Mutex<Vec<Vec<Message>>>,
    }

    impl ScriptedModel {
        fn new(replies: Vec<Message>) -> Self {
            Self {
                script: Mutex::new(replies.into_iter().map(Ok).rev().collect()),
                observed: Mutex::new(Vec::new()),
            }
        }
        fn observed(&self) -> Vec<Vec<Message>> {
            self.observed.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl Runnable<Vec<Message>, Message> for ScriptedModel {
        async fn invoke(&self, input: Vec<Message>, _ctx: &ExecutionContext) -> Result<Message> {
            self.observed.lock().unwrap().push(input);
            self.script.lock().unwrap().pop().expect("script exhausted")
        }
    }

    #[tokio::test]
    async fn first_attempt_sees_only_original_query() {
        let model = Arc::new(ScriptedModel::new(vec![assistant(
            "alpha letter explanation",
        )]));
        let rewriter = LlmQueryRewriter::builder(Arc::clone(&model)).build();
        let out = rewriter
            .rewrite("what is alpha?", &[], &ExecutionContext::new())
            .await
            .unwrap();
        assert_eq!(out, "alpha letter explanation");

        // The prompt's third part records "(none)" when no prior
        // attempts exist — pin that the rewriter signals an empty
        // history rather than emitting an empty `<failed_attempts>`
        // block that the model could mistake for a malformed prompt.
        let prompts = model.observed();
        let parts = &prompts[0][0].content;
        let prior_text = match &parts[2] {
            ContentPart::Text { text, .. } => text.clone(),
            _ => panic!("third part must be Text"),
        };
        assert!(prior_text.contains("(none)"));
    }

    #[tokio::test]
    async fn subsequent_attempts_carry_prior_history() {
        let model = Arc::new(ScriptedModel::new(vec![assistant(
            "what does alpha denote in linear algebra?",
        )]));
        let rewriter = LlmQueryRewriter::builder(Arc::clone(&model)).build();
        let prior = vec!["alpha?".to_owned(), "alpha letter".to_owned()];
        rewriter
            .rewrite("alpha", &prior, &ExecutionContext::new())
            .await
            .unwrap();
        let prompts = model.observed();
        let prior_text = match &prompts[0][0].content[2] {
            ContentPart::Text { text, .. } => text.clone(),
            _ => panic!("third part must be Text"),
        };
        assert!(prior_text.contains("attempt 1: alpha?"));
        assert!(prior_text.contains("attempt 2: alpha letter"));
    }

    #[tokio::test]
    async fn double_quotes_stripped_from_reply() {
        let model = Arc::new(ScriptedModel::new(vec![assistant(
            "\"alpha definition with examples\"",
        )]));
        let rewriter = LlmQueryRewriter::builder(model).build();
        let out = rewriter
            .rewrite("alpha", &[], &ExecutionContext::new())
            .await
            .unwrap();
        assert_eq!(out, "alpha definition with examples");
    }

    #[tokio::test]
    async fn single_quotes_stripped_from_reply() {
        let model = Arc::new(ScriptedModel::new(vec![assistant("'alpha primer'")]));
        let rewriter = LlmQueryRewriter::builder(model).build();
        let out = rewriter
            .rewrite("alpha", &[], &ExecutionContext::new())
            .await
            .unwrap();
        assert_eq!(out, "alpha primer");
    }

    #[tokio::test]
    async fn whitespace_around_reply_trimmed() {
        let model = Arc::new(ScriptedModel::new(vec![assistant("   alpha primer\n")]));
        let rewriter = LlmQueryRewriter::builder(model).build();
        let out = rewriter
            .rewrite("alpha", &[], &ExecutionContext::new())
            .await
            .unwrap();
        assert_eq!(out, "alpha primer");
    }

    #[tokio::test]
    async fn empty_reply_surfaces_invalid_request_error() {
        // A model that produces no usable text is a structural
        // failure — degrading silently to an empty rewrite would
        // loop the corrective recipe on the same retrieval.
        let model = Arc::new(ScriptedModel::new(vec![assistant("   \n  ")]));
        let rewriter = LlmQueryRewriter::builder(model).build();
        let err = rewriter
            .rewrite("alpha", &[], &ExecutionContext::new())
            .await
            .unwrap_err();
        assert!(matches!(err, Error::InvalidRequest(_)));
    }

    #[tokio::test]
    async fn model_error_propagates() {
        struct FailingModel;
        #[async_trait]
        impl Runnable<Vec<Message>, Message> for FailingModel {
            async fn invoke(
                &self,
                _input: Vec<Message>,
                _ctx: &ExecutionContext,
            ) -> Result<Message> {
                Err(Error::provider_http(503, "transient"))
            }
        }
        let rewriter = LlmQueryRewriter::builder(Arc::new(FailingModel)).build();
        let err = rewriter
            .rewrite("alpha", &[], &ExecutionContext::new())
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            Error::Provider {
                kind: entelix_core::ProviderErrorKind::Http(503),
                ..
            }
        ));
    }
}
