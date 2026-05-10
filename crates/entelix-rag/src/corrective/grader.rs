//! Retrieval grader — `(query, document) → GradeVerdict`.
//!
//! Per CRAG (Yan et al. 2024), the grader is the decision the
//! corrective recipe routes on: every retrieved document collects
//! a verdict, and the recipe combines the per-document verdicts
//! into a per-batch action (proceed to generation, rewrite the
//! query, or fall back to an external source). This module ships
//! the *grader primitive* (trait + reference LLM impl); the
//! routing logic lives in the recipe slice that consumes it.

use std::sync::Arc;

use async_trait::async_trait;
use entelix_core::ir::{ContentPart, Message, Role};
use entelix_core::{ExecutionContext, Result};
use entelix_memory::Document as RetrievedDocument;
use entelix_runnable::Runnable;
use serde::{Deserialize, Serialize};

/// Three-way verdict the grader emits per `(query, document)`
/// pair, matching the CRAG paper's relevance classes.
///
/// `Correct` — the document directly answers the query; safe to
///   feed downstream generation as-is.
/// `Ambiguous` — partial relevance; the document is on-topic but
///   the recipe should refine the query or augment with another
///   source before generation.
/// `Incorrect` — the document does not answer the query; the
///   recipe should either drop it from the context or trigger a
///   query rewrite.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum GradeVerdict {
    /// Direct answer — feed downstream verbatim.
    Correct,
    /// On-topic but not a direct answer — refine before
    /// generation.
    Ambiguous,
    /// Off-topic or contradictory — drop or rewrite.
    Incorrect,
}

impl GradeVerdict {
    /// Whether this verdict alone justifies feeding the document
    /// to the generator without further refinement.
    #[must_use]
    pub const fn is_actionable(self) -> bool {
        matches!(self, Self::Correct)
    }

    /// Stable canonical lowercase label for prompts and audit
    /// dashboards. The grader prompt asks the model to reply with
    /// one of these exact strings.
    #[must_use]
    pub const fn as_label(self) -> &'static str {
        match self {
            Self::Correct => "correct",
            Self::Ambiguous => "ambiguous",
            Self::Incorrect => "incorrect",
        }
    }
}

/// Async trait the corrective-RAG recipe consults for every
/// retrieved document. Implementations may be LLM-driven (the
/// canonical case, see [`LlmRetrievalGrader`]) or keyword /
/// heuristic / classifier-model based — the recipe doesn't care
/// as long as the verdict is one of the three [`GradeVerdict`]
/// variants.
#[async_trait]
pub trait RetrievalGrader: Send + Sync {
    /// Stable grader identifier — surfaces in audit dashboards
    /// alongside the per-document verdict.
    fn name(&self) -> &'static str;

    /// Classify how well `document` answers `query`. The
    /// `document.content` is the model-facing payload; metadata
    /// (filtering hints, source provenance) is also available
    /// when the grader needs richer context.
    async fn grade(
        &self,
        query: &str,
        document: &RetrievedDocument,
        ctx: &ExecutionContext,
    ) -> Result<GradeVerdict>;
}

/// Default instruction prepended to every model call. Frames the
/// task verbatim in the CRAG-paper terms so the model emits one
/// of the three canonical labels.
pub const DEFAULT_GRADER_INSTRUCTION: &str = "\
You are a retrieval grader. Given a user query and one retrieved document, decide whether \
the document answers the query. Reply with exactly one of: `correct` (the document directly \
answers), `ambiguous` (the document is on-topic but does not directly answer), or `incorrect` \
(the document does not answer or is off-topic). Reply with only the single label — no \
explanation, no quotes, no surrounding text.";

/// Stable grader identifier for [`LlmRetrievalGrader`].
const LLM_GRADER_NAME: &str = "llm-retrieval-grader";

/// Builder for [`LlmRetrievalGrader`].
pub struct LlmRetrievalGraderBuilder<M> {
    model: Arc<M>,
    instruction: String,
}

impl<M> LlmRetrievalGraderBuilder<M>
where
    M: Runnable<Vec<Message>, Message> + 'static,
{
    /// Override the operator-facing instruction. The default
    /// matches [`DEFAULT_GRADER_INSTRUCTION`] verbatim.
    #[must_use]
    pub fn with_instruction(mut self, instruction: impl Into<String>) -> Self {
        self.instruction = instruction.into();
        self
    }

    /// Finalise into a runnable grader.
    #[must_use]
    pub fn build(self) -> LlmRetrievalGrader<M> {
        LlmRetrievalGrader {
            model: self.model,
            instruction: Arc::from(self.instruction),
        }
    }
}

/// Reference LLM-driven [`RetrievalGrader`]. Asks the supplied
/// `Runnable<Vec<Message>, Message>` model to classify relevance,
/// then parses the reply into a [`GradeVerdict`]. Operators
/// inheriting from this default tune the prompt via
/// [`LlmRetrievalGraderBuilder::with_instruction`] or write their
/// own grader from scratch.
pub struct LlmRetrievalGrader<M> {
    model: Arc<M>,
    instruction: Arc<str>,
}

impl<M> LlmRetrievalGrader<M>
where
    M: Runnable<Vec<Message>, Message> + 'static,
{
    /// Start a builder bound to the supplied model.
    #[must_use]
    pub fn builder(model: Arc<M>) -> LlmRetrievalGraderBuilder<M> {
        LlmRetrievalGraderBuilder {
            model,
            instruction: DEFAULT_GRADER_INSTRUCTION.to_owned(),
        }
    }

    /// Build the user message that frames one grade call. Three
    /// text parts: instruction, query, document — operator's
    /// model picks them up as one user turn.
    fn build_prompt(&self, query: &str, document: &RetrievedDocument) -> Vec<Message> {
        vec![Message::new(
            Role::User,
            vec![
                ContentPart::text(self.instruction.to_string()),
                ContentPart::text(format!("<query>\n{query}\n</query>")),
                ContentPart::text(format!("<document>\n{}\n</document>", document.content)),
            ],
        )]
    }
}

impl<M> Clone for LlmRetrievalGrader<M> {
    fn clone(&self) -> Self {
        Self {
            model: Arc::clone(&self.model),
            instruction: Arc::clone(&self.instruction),
        }
    }
}

impl<M> std::fmt::Debug for LlmRetrievalGrader<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmRetrievalGrader").finish_non_exhaustive()
    }
}

#[async_trait]
impl<M> RetrievalGrader for LlmRetrievalGrader<M>
where
    M: Runnable<Vec<Message>, Message> + 'static,
{
    fn name(&self) -> &'static str {
        LLM_GRADER_NAME
    }

    async fn grade(
        &self,
        query: &str,
        document: &RetrievedDocument,
        ctx: &ExecutionContext,
    ) -> Result<GradeVerdict> {
        let prompt = self.build_prompt(query, document);
        let reply = self.model.invoke(prompt, ctx).await?;
        Ok(parse_verdict(&reply))
    }
}

/// Parse the model's reply into a [`GradeVerdict`]. Tolerant of
/// surrounding whitespace, punctuation, and case — looks for the
/// first canonical label substring. An unparseable reply
/// degrades to [`GradeVerdict::Ambiguous`] so the recipe routes
/// it through the safer rewrite path rather than treating it as
/// a confident `Correct` or `Incorrect`.
fn parse_verdict(message: &Message) -> GradeVerdict {
    let mut text = String::new();
    for part in &message.content {
        if let ContentPart::Text { text: t, .. } = part {
            text.push_str(t);
        }
    }
    let lower = text.to_lowercase();
    if lower.contains("incorrect") {
        // Ordered before "correct" because "incorrect" contains
        // "correct" as a substring.
        GradeVerdict::Incorrect
    } else if lower.contains("ambiguous") {
        GradeVerdict::Ambiguous
    } else if lower.contains("correct") {
        GradeVerdict::Correct
    } else {
        GradeVerdict::Ambiguous
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn doc(content: &str) -> RetrievedDocument {
        RetrievedDocument::new(content)
    }

    fn assistant(text: &str) -> Message {
        Message::new(Role::Assistant, vec![ContentPart::text(text)])
    }

    /// Scripted model — same shape used in contextual chunker
    /// tests; pop next reply per invocation.
    struct ScriptedModel {
        script: Mutex<Vec<Result<Message>>>,
    }

    impl ScriptedModel {
        fn new(replies: Vec<Message>) -> Self {
            Self {
                script: Mutex::new(replies.into_iter().map(Ok).rev().collect()),
            }
        }
    }

    #[async_trait]
    impl Runnable<Vec<Message>, Message> for ScriptedModel {
        async fn invoke(&self, _input: Vec<Message>, _ctx: &ExecutionContext) -> Result<Message> {
            self.script.lock().unwrap().pop().expect("script exhausted")
        }
    }

    #[test]
    fn verdict_is_actionable_only_for_correct() {
        assert!(GradeVerdict::Correct.is_actionable());
        assert!(!GradeVerdict::Ambiguous.is_actionable());
        assert!(!GradeVerdict::Incorrect.is_actionable());
    }

    #[test]
    fn verdict_label_round_trips() {
        // The prompt instructs the model to emit the lowercase
        // label verbatim; this pin keeps the prompt and parser in
        // sync so a label rename can never silently break the
        // recipe.
        assert_eq!(GradeVerdict::Correct.as_label(), "correct");
        assert_eq!(GradeVerdict::Ambiguous.as_label(), "ambiguous");
        assert_eq!(GradeVerdict::Incorrect.as_label(), "incorrect");
    }

    #[test]
    fn parser_accepts_canonical_lowercase() {
        assert_eq!(parse_verdict(&assistant("correct")), GradeVerdict::Correct);
        assert_eq!(
            parse_verdict(&assistant("ambiguous")),
            GradeVerdict::Ambiguous
        );
        assert_eq!(
            parse_verdict(&assistant("incorrect")),
            GradeVerdict::Incorrect
        );
    }

    #[test]
    fn parser_tolerates_whitespace_punctuation_and_case() {
        assert_eq!(parse_verdict(&assistant("Correct.")), GradeVerdict::Correct);
        assert_eq!(
            parse_verdict(&assistant("  AMBIGUOUS\n")),
            GradeVerdict::Ambiguous
        );
        assert_eq!(
            parse_verdict(&assistant("Verdict: incorrect")),
            GradeVerdict::Incorrect
        );
    }

    #[test]
    fn parser_disambiguates_incorrect_from_correct() {
        // "incorrect" contains "correct" as a substring — the
        // parser must classify "incorrect" before "correct" or
        // the design property breaks. Pin it.
        let reply = assistant("incorrect");
        assert_eq!(parse_verdict(&reply), GradeVerdict::Incorrect);
    }

    #[test]
    fn parser_degrades_unknown_reply_to_ambiguous() {
        // Misformed reply — the recipe should route through the
        // rewrite path rather than treat it as confident
        // correct/incorrect. Ambiguous is the safer default.
        assert_eq!(
            parse_verdict(&assistant("the document looks fine to me")),
            GradeVerdict::Ambiguous
        );
        assert_eq!(parse_verdict(&assistant("")), GradeVerdict::Ambiguous);
    }

    #[tokio::test]
    async fn grader_dispatches_through_model_and_returns_parsed_verdict() {
        let model = Arc::new(ScriptedModel::new(vec![assistant("correct")]));
        let grader = LlmRetrievalGrader::builder(model).build();
        let verdict = grader
            .grade(
                "alpha?",
                &doc("alpha is the first letter"),
                &ExecutionContext::new(),
            )
            .await
            .unwrap();
        assert_eq!(verdict, GradeVerdict::Correct);
        assert_eq!(grader.name(), LLM_GRADER_NAME);
    }

    #[tokio::test]
    async fn grader_propagates_model_error() {
        struct FailingModel;
        #[async_trait]
        impl Runnable<Vec<Message>, Message> for FailingModel {
            async fn invoke(
                &self,
                _input: Vec<Message>,
                _ctx: &ExecutionContext,
            ) -> Result<Message> {
                Err(entelix_core::Error::provider_http(503, "transient"))
            }
        }
        let grader = LlmRetrievalGrader::builder(Arc::new(FailingModel)).build();
        let err = grader
            .grade("query", &doc("text"), &ExecutionContext::new())
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            entelix_core::Error::Provider {
                kind: entelix_core::ProviderErrorKind::Http(503),
                ..
            }
        ));
    }
}
