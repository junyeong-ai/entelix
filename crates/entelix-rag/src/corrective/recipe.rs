//! `create_corrective_rag_agent` — CRAG (Yan et al. 2024)
//! topology assembled into a runnable [`Agent`] over the
//! primitives this module ships.
//!
//! ## Topology
//!
//! ```text
//!         ┌─────────┐    ┌──────┐    ┌────────┐
//! query ─▶│retrieve │───▶│grade │───▶│decide  │
//!         └─────────┘    └──────┘    └───┬────┘
//!              ▲                         │
//!              │                  ┌──────┴──────┐
//!              │                  ▼             ▼
//!         ┌────┴────┐         ┌────────┐   ┌─────────┐
//!         │rewrite  │◀────────│ retry  │   │generate │──▶ END
//!         └─────────┘         └────────┘   └─────────┘
//! ```
//!
//! Retrieve runs against the operator's [`Retriever`]; every
//! result is graded by the [`RetrievalGrader`]; `decide` (a pure
//! state-based router, no LLM call) routes on the fraction of
//! [`GradeVerdict::Correct`] verdicts vs the configured
//! [`CragConfig::min_correct_fraction`] threshold; `rewrite`
//! re-issues the query through the [`QueryRewriter`] and loops back
//! to retrieve; `generate` produces the final answer over the
//! operator-supplied generator model. Configurable knobs
//! ([`CragConfig`]) control the routing threshold and the
//! rewrite-loop attempt cap.
//!
//! The CRAG paper's full three-way decision (Correct vs Ambiguous
//! vs Incorrect, with a web-search branch on Incorrect) is
//! intentionally collapsed here to a Correct-vs-not-Correct
//! routing — the SDK ships no built-in web-search primitive, and
//! operators wanting the third branch wire it as a fallback inside
//! their custom [`Retriever`] (try local KB → escalate to web on
//! empty / low-confidence hits).
//!
//! ## When to reach for this recipe
//!
//! Use this when the corpus is messy enough that naive
//! retrieve-then-generate produces low-quality grounded answers
//! and the operator wants the LLM-driven self-correction loop the
//! CRAG paper describes. For corpora where retrieval quality is
//! already high (well-curated technical docs, reference KB), the
//! plain `IngestionPipeline` + manual `Retriever::retrieve` +
//! `ChatModel` composition is cheaper and simpler — corrective
//! routing only pays off when retrieval failures are common
//! enough to amortise the grader's per-document cost.

use std::sync::Arc;

use entelix_agents::Agent;
use entelix_core::ir::{ContentPart, Message, Role, SystemPrompt};
use entelix_core::{ExecutionContext, Result};
use entelix_graph::{CompiledGraph, StateGraph};
use entelix_memory::{Document as RetrievedDocument, RetrievalQuery, Retriever};
use entelix_runnable::{Runnable, RunnableLambda};

use crate::corrective::grader::{GradeVerdict, RetrievalGrader};
use crate::corrective::rewriter::QueryRewriter;

/// Default minimum fraction of retrieved documents that must
/// grade [`GradeVerdict::Correct`] for the recipe to skip rewriting
/// and proceed directly to generation. `0.5` matches the CRAG
/// paper's mid-confidence threshold — operators tuning for higher
/// retrieval precision raise it; tuning for lower model spend
/// (fewer rewrites at the cost of weaker grounding) lower it.
pub const DEFAULT_MIN_CORRECT_FRACTION: f32 = 0.5;

/// Default top-k passed into the retriever on every retrieval
/// pass. Operator-overridable via
/// [`CragConfig::with_retrieval_top_k`].
pub const DEFAULT_RETRIEVAL_TOP_K: usize = 5;

/// Default cap on rewrite-loop attempts before the recipe
/// surrenders and generates over whatever was retrieved last.
/// `3` is the CRAG paper's reported sweet spot (retrieval rarely
/// improves beyond the third rewrite).
pub const DEFAULT_MAX_REWRITE_ATTEMPTS: u32 = 3;

/// Default system prompt the generator node prepends to every
/// answer-generation call. Vendor-neutral, focused on grounded
/// answer style.
pub const DEFAULT_GENERATOR_SYSTEM_PROMPT: &str = "\
You are a helpful assistant. Answer the user's question using only the supplied retrieved \
documents as your evidence base. If the documents don't contain enough information to \
answer with confidence, say so explicitly. Never fabricate facts that the documents do \
not support.";

/// Operator-tunable knobs for the corrective-RAG recipe. Construct
/// via [`Self::new`] or [`Self::default`]; chain `with_*` setters.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct CragConfig {
    min_correct_fraction: f32,
    retrieval_top_k: usize,
    max_rewrite_attempts: u32,
    generator_system_prompt: SystemPrompt,
}

impl CragConfig {
    /// Build with the default thresholds + retrieval top-k +
    /// system prompt.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the minimum fraction of `Correct` verdicts the
    /// recipe needs before generating without a rewrite. Values
    /// are clamped to `[0.0, 1.0]` at decision time — operators
    /// supplying out-of-range values get the clamped value, no
    /// error.
    #[must_use]
    pub const fn with_min_correct_fraction(mut self, fraction: f32) -> Self {
        self.min_correct_fraction = fraction;
        self
    }

    /// Override the retrieval top-k.
    #[must_use]
    pub const fn with_retrieval_top_k(mut self, top_k: usize) -> Self {
        self.retrieval_top_k = top_k;
        self
    }

    /// Override the rewrite-loop attempt cap. After this many
    /// rewrites, the recipe generates over whatever the last
    /// retrieval returned (even if every document graded
    /// `Incorrect`) — surrender beats infinite loop.
    #[must_use]
    pub const fn with_max_rewrite_attempts(mut self, max: u32) -> Self {
        self.max_rewrite_attempts = max;
        self
    }

    /// Override the system prompt the generator node uses. Default
    /// is [`DEFAULT_GENERATOR_SYSTEM_PROMPT`].
    #[must_use]
    pub fn with_generator_system_prompt(mut self, prompt: SystemPrompt) -> Self {
        self.generator_system_prompt = prompt;
        self
    }

    /// Effective minimum-correct fraction.
    #[must_use]
    pub const fn min_correct_fraction(&self) -> f32 {
        self.min_correct_fraction
    }

    /// Effective retrieval top-k.
    #[must_use]
    pub const fn retrieval_top_k(&self) -> usize {
        self.retrieval_top_k
    }

    /// Effective rewrite attempt cap.
    #[must_use]
    pub const fn max_rewrite_attempts(&self) -> u32 {
        self.max_rewrite_attempts
    }

    /// Borrow the configured generator system prompt.
    #[must_use]
    pub const fn generator_system_prompt(&self) -> &SystemPrompt {
        &self.generator_system_prompt
    }
}

impl Default for CragConfig {
    fn default() -> Self {
        Self {
            min_correct_fraction: DEFAULT_MIN_CORRECT_FRACTION,
            retrieval_top_k: DEFAULT_RETRIEVAL_TOP_K,
            max_rewrite_attempts: DEFAULT_MAX_REWRITE_ATTEMPTS,
            generator_system_prompt: SystemPrompt::text(DEFAULT_GENERATOR_SYSTEM_PROMPT),
        }
    }
}

/// State the corrective-RAG graph drives across nodes. Carries
/// the original + current query, the rewrite history, the last
/// retrieval batch + verdicts, the surviving correct subset, and
/// the terminal answer.
///
/// `attempt` counts rewrite passes — `0` is the original query,
/// `n > 0` is the n-th rewrite. Compared against
/// [`CragConfig::max_rewrite_attempts`] before each rewrite to
/// short-circuit the loop.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct CorrectiveRagState {
    /// Original user query. Untouched across the run.
    pub original_query: String,
    /// Current query being retrieved against. Updated by each
    /// rewrite pass.
    pub query: String,
    /// Every prior failed query attempt, oldest first. The
    /// rewriter sees this so it doesn't re-emit a previous
    /// attempt.
    pub previous_attempts: Vec<String>,
    /// Per-document grade from the most recent grading pass.
    pub graded: Vec<(RetrievedDocument, GradeVerdict)>,
    /// Filtered subset of `graded` containing only documents
    /// with [`GradeVerdict::Correct`]. The generator node sees
    /// this as evidence base.
    pub correct_documents: Vec<RetrievedDocument>,
    /// Number of rewrite attempts performed so far. `0` before
    /// the first rewrite.
    pub attempt: u32,
    /// Final answer text. `None` until the generator runs.
    pub answer: Option<String>,
}

impl CorrectiveRagState {
    /// Seed a fresh state with the user's query.
    #[must_use]
    pub fn from_query(query: impl Into<String>) -> Self {
        let query = query.into();
        Self {
            original_query: query.clone(),
            query,
            previous_attempts: Vec::new(),
            graded: Vec::new(),
            correct_documents: Vec::new(),
            attempt: 0,
            answer: None,
        }
    }
}

/// Compile the corrective-RAG graph from operator-supplied
/// primitives. Use this when you need to embed the graph as a
/// node in a larger [`StateGraph`]; for a ready-to-execute
/// agent, prefer [`create_corrective_rag_agent`].
pub fn build_corrective_rag_graph<Ret, G, R, M>(
    retriever: Arc<Ret>,
    grader: G,
    rewriter: R,
    generator: M,
    config: CragConfig,
) -> Result<CompiledGraph<CorrectiveRagState>>
where
    Ret: Retriever + ?Sized + 'static,
    G: RetrievalGrader + 'static,
    R: QueryRewriter + 'static,
    M: Runnable<Vec<Message>, Message> + 'static,
{
    let config = Arc::new(config);
    let generator = Arc::new(generator);
    let grader = Arc::new(grader);
    let rewriter = Arc::new(rewriter);

    StateGraph::<CorrectiveRagState>::new()
        .add_node(NODE_RETRIEVE, make_retriever_node(retriever, &config))
        .add_node(NODE_GRADE, make_grader_node(grader))
        .add_node(NODE_REWRITE, make_rewriter_node(rewriter))
        .add_node(NODE_GENERATE, make_generator_node(generator, &config))
        .set_entry_point(NODE_RETRIEVE)
        .add_finish_point(NODE_GENERATE)
        .add_edge(NODE_RETRIEVE, NODE_GRADE)
        .add_edge(NODE_REWRITE, NODE_RETRIEVE)
        .add_conditional_edges(
            NODE_GRADE,
            {
                let config = Arc::clone(&config);
                move |state: &CorrectiveRagState| route_after_grade(state, &config)
            },
            [(NODE_REWRITE, NODE_REWRITE), (NODE_GENERATE, NODE_GENERATE)],
        )
        .compile()
}

fn make_retriever_node<Ret>(
    retriever: Arc<Ret>,
    config: &Arc<CragConfig>,
) -> impl Runnable<CorrectiveRagState, CorrectiveRagState> + 'static
where
    Ret: Retriever + ?Sized + 'static,
{
    let config = Arc::clone(config);
    RunnableLambda::new(
        move |mut state: CorrectiveRagState, ctx: ExecutionContext| {
            let retriever = Arc::clone(&retriever);
            let top_k = config.retrieval_top_k;
            async move {
                let query = RetrievalQuery::new(state.query.clone(), top_k);
                let docs = retriever.retrieve(query, &ctx).await?;
                state.graded = docs
                    .into_iter()
                    .map(|d| (d, GradeVerdict::Ambiguous))
                    .collect();
                state.correct_documents.clear();
                Ok::<_, _>(state)
            }
        },
    )
}

fn make_grader_node<G>(
    grader: Arc<G>,
) -> impl Runnable<CorrectiveRagState, CorrectiveRagState> + 'static
where
    G: RetrievalGrader + ?Sized + 'static,
{
    RunnableLambda::new(
        move |mut state: CorrectiveRagState, ctx: ExecutionContext| {
            let grader = Arc::clone(&grader);
            async move {
                let mut verdicts = Vec::with_capacity(state.graded.len());
                for (doc, _) in std::mem::take(&mut state.graded) {
                    let verdict = grader.grade(&state.query, &doc, &ctx).await?;
                    verdicts.push((doc, verdict));
                }
                state.correct_documents = verdicts
                    .iter()
                    .filter(|(_, v)| matches!(v, GradeVerdict::Correct))
                    .map(|(d, _)| d.clone())
                    .collect();
                state.graded = verdicts;
                Ok::<_, _>(state)
            }
        },
    )
}

fn make_rewriter_node<R>(
    rewriter: Arc<R>,
) -> impl Runnable<CorrectiveRagState, CorrectiveRagState> + 'static
where
    R: QueryRewriter + ?Sized + 'static,
{
    RunnableLambda::new(
        move |mut state: CorrectiveRagState, ctx: ExecutionContext| {
            let rewriter = Arc::clone(&rewriter);
            async move {
                let new_query = rewriter
                    .rewrite(&state.original_query, &state.previous_attempts, &ctx)
                    .await?;
                state.previous_attempts.push(state.query.clone());
                state.query = new_query;
                state.attempt = state.attempt.saturating_add(1);
                Ok::<_, _>(state)
            }
        },
    )
}

fn make_generator_node<M>(
    generator: Arc<M>,
    config: &Arc<CragConfig>,
) -> impl Runnable<CorrectiveRagState, CorrectiveRagState> + 'static
where
    M: Runnable<Vec<Message>, Message> + ?Sized + 'static,
{
    let config = Arc::clone(config);
    RunnableLambda::new(
        move |mut state: CorrectiveRagState, ctx: ExecutionContext| {
            let generator = Arc::clone(&generator);
            let config = Arc::clone(&config);
            async move {
                let messages = build_generator_prompt(
                    &state.original_query,
                    &state.correct_documents,
                    &state.graded,
                    &config,
                );
                let reply = generator.invoke(messages, &ctx).await?;
                let text = extract_text(&reply);
                state.answer = Some(text);
                Ok::<_, _>(state)
            }
        },
    )
}

/// Stable agent name surfaced on every emitted
/// [`entelix_agents::AgentEvent`] and OTel `entelix.agent.run` span.
pub const CORRECTIVE_RAG_AGENT_NAME: &str = "corrective-rag";

/// Single source of truth for the CRAG graph node identifiers —
/// `add_node` / `add_edge` / `add_conditional_edges` / the router
/// closure all reference these constants so a typo surfaces at
/// compile time rather than as a silent dispatch miss.
const NODE_RETRIEVE: &str = "retrieve";
const NODE_GRADE: &str = "grade";
const NODE_REWRITE: &str = "rewrite";
const NODE_GENERATE: &str = "generate";

/// Build a ready-to-execute corrective-RAG [`Agent`]. Wraps
/// [`build_corrective_rag_graph`] in the standard `Agent<S>` shape
/// so the full lifecycle (`AgentEvent` stream, sink fan-out,
/// observer hooks, supervisor handoff) integrates uniformly with
/// every other recipe (`create_react_agent`,
/// `create_supervisor_agent`, `create_chat_agent`).
///
/// `Agent::execute` returns
/// [`AgentRunResult<CorrectiveRagState>`](entelix_agents::AgentRunResult)
/// — the standard envelope carrying the terminal state plus a
/// frozen `RunBudget` snapshot. Seed the input via
/// [`CorrectiveRagState::from_query`].
///
/// Operators embedding the CRAG graph as a node in a larger
/// `StateGraph<S>` reach for [`build_corrective_rag_graph`]
/// directly and skip the `Agent` wrapper.
pub fn create_corrective_rag_agent<Ret, G, R, M>(
    retriever: Arc<Ret>,
    grader: G,
    rewriter: R,
    generator: M,
    config: CragConfig,
) -> Result<Agent<CorrectiveRagState>>
where
    Ret: Retriever + ?Sized + 'static,
    G: RetrievalGrader + 'static,
    R: QueryRewriter + 'static,
    M: Runnable<Vec<Message>, Message> + 'static,
{
    let graph = build_corrective_rag_graph(retriever, grader, rewriter, generator, config)?;
    Agent::builder()
        .with_name(CORRECTIVE_RAG_AGENT_NAME)
        .with_runnable(graph)
        .build()
}

/// Route after a `grade` pass — decide whether to proceed to
/// `generate` or loop back through `rewrite`.
///
/// Routing rules (CRAG paper):
/// - No documents at all → `rewrite` (unless attempt cap hit).
/// - Fraction of `Correct` verdicts ≥ `min_correct_fraction` →
///   `generate` (we have enough grounded evidence).
/// - Otherwise → `rewrite` (if budget remains) or `generate`
///   (when the rewrite budget is exhausted; surrender beats
///   infinite loop).
fn route_after_grade(state: &CorrectiveRagState, config: &CragConfig) -> String {
    let total = state.graded.len();
    let correct = state.correct_documents.len();
    let fraction = if total == 0 {
        0.0_f32
    } else {
        // Cast is benign — chunk counts above ~16M are
        // pathological for an in-flight retrieval batch and the
        // operator's top_k caps it well below f32 precision loss.
        #[allow(clippy::cast_precision_loss)]
        let n = total as f32;
        #[allow(clippy::cast_precision_loss)]
        let k = correct as f32;
        k / n
    };
    let threshold = config.min_correct_fraction.clamp(0.0, 1.0);
    let budget_remaining = state.attempt < config.max_rewrite_attempts;
    if fraction >= threshold && correct > 0 {
        NODE_GENERATE.to_owned()
    } else if budget_remaining {
        NODE_REWRITE.to_owned()
    } else {
        NODE_GENERATE.to_owned()
    }
}

/// Build the user message the generator node sends to the model.
/// Three text parts: the original user query, the correct
/// documents (one block each), and a fallback context list when
/// every retrieval graded poorly so the generator at least sees
/// what the corpus surfaced.
fn build_generator_prompt(
    original_query: &str,
    correct_documents: &[RetrievedDocument],
    fallback_graded: &[(RetrievedDocument, GradeVerdict)],
    config: &CragConfig,
) -> Vec<Message> {
    let evidence: String = if correct_documents.is_empty() {
        // No `Correct` verdicts — generator runs over whatever
        // graded best. Prefer `Ambiguous` over `Incorrect`; that
        // ordering is the CRAG paper's degraded-mode behaviour.
        let mut docs: Vec<&RetrievedDocument> = fallback_graded
            .iter()
            .filter_map(|(d, v)| matches!(v, GradeVerdict::Ambiguous).then_some(d))
            .collect();
        if docs.is_empty() {
            docs = fallback_graded.iter().map(|(d, _)| d).collect();
        }
        format_documents(&docs)
    } else {
        let docs: Vec<&RetrievedDocument> = correct_documents.iter().collect();
        format_documents(&docs)
    };

    let mut messages: Vec<Message> = config
        .generator_system_prompt
        .blocks()
        .iter()
        .map(|b| Message::new(Role::System, vec![ContentPart::text(b.text.clone())]))
        .collect();
    messages.push(Message::new(
        Role::User,
        vec![
            ContentPart::text(format!("<query>\n{original_query}\n</query>")),
            ContentPart::text(format!("<documents>\n{evidence}\n</documents>")),
        ],
    ));
    messages
}

/// Render a slice of documents as one concatenated text block,
/// each delimited by a blank line so the model can tell them
/// apart without the recipe imposing JSON structure.
fn format_documents(docs: &[&RetrievedDocument]) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    for (idx, doc) in docs.iter().enumerate() {
        if idx > 0 {
            out.push_str("\n\n");
        }
        write!(&mut out, "[{idx}] {}", doc.content).expect("writing to String never fails");
    }
    out
}

/// Pull the assistant reply text out of the model's [`Message`].
/// Concatenates every [`ContentPart::Text`] part with single
/// newline separators; non-text parts are skipped.
fn extract_text(message: &Message) -> String {
    let mut buf = String::new();
    for part in &message.content {
        if let ContentPart::Text { text, .. } = part {
            if !buf.is_empty() {
                buf.push('\n');
            }
            buf.push_str(text);
        }
    }
    buf.trim().to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corrective::grader::RetrievalGrader;
    use crate::corrective::rewriter::QueryRewriter;
    use async_trait::async_trait;
    use entelix_memory::Document as RetrievedDocument;
    use std::sync::Mutex;

    /// Static retriever: returns a pre-canned doc set per query
    /// and tracks every query it saw.
    struct StaticRetriever {
        docs_by_query: std::collections::HashMap<String, Vec<RetrievedDocument>>,
        observed_queries: Mutex<Vec<String>>,
    }

    impl StaticRetriever {
        fn new() -> Self {
            Self {
                docs_by_query: std::collections::HashMap::new(),
                observed_queries: Mutex::new(Vec::new()),
            }
        }

        fn with(mut self, query: &str, docs: Vec<RetrievedDocument>) -> Self {
            self.docs_by_query.insert(query.to_owned(), docs);
            self
        }

        fn observed(&self) -> Vec<String> {
            self.observed_queries.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl Retriever for StaticRetriever {
        async fn retrieve(
            &self,
            query: RetrievalQuery,
            _ctx: &ExecutionContext,
        ) -> Result<Vec<RetrievedDocument>> {
            self.observed_queries
                .lock()
                .unwrap()
                .push(query.text.clone());
            Ok(self
                .docs_by_query
                .get(&query.text)
                .cloned()
                .unwrap_or_default())
        }
    }

    /// Verdict-scripted grader: returns pre-canned verdicts per
    /// document content.
    struct ScriptedGrader {
        verdicts: std::collections::HashMap<String, GradeVerdict>,
    }

    impl ScriptedGrader {
        fn new(map: &[(&str, GradeVerdict)]) -> Self {
            Self {
                verdicts: map.iter().map(|(k, v)| ((*k).to_owned(), *v)).collect(),
            }
        }
    }

    #[async_trait]
    impl RetrievalGrader for ScriptedGrader {
        fn name(&self) -> &'static str {
            "scripted-grader"
        }
        async fn grade(
            &self,
            _query: &str,
            doc: &RetrievedDocument,
            _ctx: &ExecutionContext,
        ) -> Result<GradeVerdict> {
            Ok(self
                .verdicts
                .get(&doc.content)
                .copied()
                .unwrap_or(GradeVerdict::Ambiguous))
        }
    }

    /// Rewriter that returns the next pre-canned string per call.
    struct ScriptedRewriter {
        replies: Mutex<Vec<String>>,
    }

    impl ScriptedRewriter {
        fn new(replies: &[&str]) -> Self {
            Self {
                replies: Mutex::new(replies.iter().map(|s| (*s).to_owned()).rev().collect()),
            }
        }
    }

    #[async_trait]
    impl QueryRewriter for ScriptedRewriter {
        fn name(&self) -> &'static str {
            "scripted-rewriter"
        }
        async fn rewrite(
            &self,
            _original: &str,
            _previous: &[String],
            _ctx: &ExecutionContext,
        ) -> Result<String> {
            Ok(self
                .replies
                .lock()
                .unwrap()
                .pop()
                .unwrap_or_else(|| "<exhausted>".to_owned()))
        }
    }

    /// Generator that records the messages it received and replies
    /// with a fixed answer. Cheap to clone — the observed-prompts
    /// log lives under an `Arc<Mutex<...>>` so the test can keep
    /// one handle for inspection while the recipe moves another
    /// into the graph.
    #[derive(Clone)]
    struct CapturingGenerator {
        observed: Arc<Mutex<Vec<Vec<Message>>>>,
        reply: String,
    }

    impl CapturingGenerator {
        fn new(reply: &str) -> Self {
            Self {
                observed: Arc::new(Mutex::new(Vec::new())),
                reply: reply.to_owned(),
            }
        }
        fn observed(&self) -> Vec<Vec<Message>> {
            self.observed.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl Runnable<Vec<Message>, Message> for CapturingGenerator {
        async fn invoke(&self, input: Vec<Message>, _ctx: &ExecutionContext) -> Result<Message> {
            self.observed.lock().unwrap().push(input);
            Ok(Message::new(
                Role::Assistant,
                vec![ContentPart::text(self.reply.clone())],
            ))
        }
    }

    fn doc(content: &str) -> RetrievedDocument {
        RetrievedDocument::new(content)
    }

    #[tokio::test]
    async fn happy_path_generates_directly_when_all_correct() {
        let retriever = Arc::new(StaticRetriever::new().with(
            "what is alpha?",
            vec![
                doc("alpha is the first letter"),
                doc("alpha is also a particle"),
            ],
        ));
        let grader = ScriptedGrader::new(&[
            ("alpha is the first letter", GradeVerdict::Correct),
            ("alpha is also a particle", GradeVerdict::Correct),
        ]);
        let rewriter = ScriptedRewriter::new(&["never used"]);
        let generator = CapturingGenerator::new("Alpha is the first letter.");

        let agent = create_corrective_rag_agent(
            Arc::clone(&retriever),
            grader,
            rewriter,
            generator.clone(),
            CragConfig::new(),
        )
        .unwrap();
        let final_state = agent
            .execute(
                CorrectiveRagState::from_query("what is alpha?"),
                &ExecutionContext::new(),
            )
            .await
            .unwrap()
            .state;

        assert_eq!(
            final_state.answer.as_deref(),
            Some("Alpha is the first letter.")
        );
        assert_eq!(
            final_state.attempt, 0,
            "no rewrite when retrieval is correct"
        );
        assert_eq!(retriever.observed(), vec!["what is alpha?".to_owned()]);
    }

    #[tokio::test]
    async fn incorrect_retrieval_triggers_rewrite_and_loops_back() {
        let retriever = Arc::new(
            StaticRetriever::new()
                .with("alpha?", vec![doc("totally off-topic")])
                .with(
                    "what is alpha letter?",
                    vec![doc("alpha is the first letter")],
                ),
        );
        let grader = ScriptedGrader::new(&[
            ("totally off-topic", GradeVerdict::Incorrect),
            ("alpha is the first letter", GradeVerdict::Correct),
        ]);
        let rewriter = ScriptedRewriter::new(&["what is alpha letter?"]);
        let generator = CapturingGenerator::new("Final.");

        let agent = create_corrective_rag_agent(
            Arc::clone(&retriever),
            grader,
            rewriter,
            generator.clone(),
            CragConfig::new(),
        )
        .unwrap();
        let final_state = agent
            .execute(
                CorrectiveRagState::from_query("alpha?"),
                &ExecutionContext::new(),
            )
            .await
            .unwrap()
            .state;

        assert_eq!(final_state.attempt, 1, "exactly one rewrite happened");
        assert_eq!(
            retriever.observed(),
            vec!["alpha?".to_owned(), "what is alpha letter?".to_owned()]
        );
        assert_eq!(final_state.answer.as_deref(), Some("Final."));
    }

    #[tokio::test]
    async fn rewrite_budget_caps_loop_and_surrenders_to_generate() {
        // Every retrieval grades Incorrect, every rewrite produces
        // a different (still-bad) query. The recipe must cap at
        // max_rewrite_attempts and generate over whatever was
        // last retrieved.
        let retriever = Arc::new(
            StaticRetriever::new()
                .with("q0", vec![doc("bad-0")])
                .with("q1", vec![doc("bad-1")])
                .with("q2", vec![doc("bad-2")])
                .with("q3", vec![doc("bad-3")]),
        );
        let grader = ScriptedGrader::new(&[
            ("bad-0", GradeVerdict::Incorrect),
            ("bad-1", GradeVerdict::Incorrect),
            ("bad-2", GradeVerdict::Incorrect),
            ("bad-3", GradeVerdict::Incorrect),
        ]);
        let rewriter = ScriptedRewriter::new(&["q1", "q2", "q3"]);
        let generator = CapturingGenerator::new("Surrendered.");

        let agent = create_corrective_rag_agent(
            Arc::clone(&retriever),
            grader,
            rewriter,
            generator.clone(),
            CragConfig::new().with_max_rewrite_attempts(2),
        )
        .unwrap();
        let final_state = agent
            .execute(
                CorrectiveRagState::from_query("q0"),
                &ExecutionContext::new(),
            )
            .await
            .unwrap()
            .state;

        assert_eq!(final_state.attempt, 2, "rewrite budget = 2");
        // Retrievals: original (q0), rewrite-1 (q1), rewrite-2 (q2).
        // Generator runs after the third retrieval — we never saw q3.
        assert_eq!(
            retriever.observed(),
            vec!["q0".to_owned(), "q1".to_owned(), "q2".to_owned()]
        );
        assert_eq!(final_state.answer.as_deref(), Some("Surrendered."));
    }

    #[tokio::test]
    async fn empty_retrieval_loops_to_rewrite_when_budget_remains() {
        let retriever = Arc::new(
            StaticRetriever::new()
                .with("q0", vec![])
                .with("q1", vec![doc("alpha is the first letter")]),
        );
        let grader = ScriptedGrader::new(&[("alpha is the first letter", GradeVerdict::Correct)]);
        let rewriter = ScriptedRewriter::new(&["q1"]);
        let generator = CapturingGenerator::new("Answered.");

        let agent = create_corrective_rag_agent(
            Arc::clone(&retriever),
            grader,
            rewriter,
            generator.clone(),
            CragConfig::new(),
        )
        .unwrap();
        let final_state = agent
            .execute(
                CorrectiveRagState::from_query("q0"),
                &ExecutionContext::new(),
            )
            .await
            .unwrap()
            .state;
        assert_eq!(final_state.attempt, 1);
        assert_eq!(final_state.answer.as_deref(), Some("Answered."));
    }

    #[tokio::test]
    async fn generator_sees_only_correct_documents_when_mixed_batch() {
        let retriever = Arc::new(StaticRetriever::new().with(
            "alpha?",
            vec![
                doc("alpha is the first letter"),
                doc("alpha is unrelated stuff"),
                doc("more about alpha letter"),
            ],
        ));
        let grader = ScriptedGrader::new(&[
            ("alpha is the first letter", GradeVerdict::Correct),
            ("alpha is unrelated stuff", GradeVerdict::Incorrect),
            ("more about alpha letter", GradeVerdict::Correct),
        ]);
        let rewriter = ScriptedRewriter::new(&["unused"]);
        let generator = CapturingGenerator::new("Answered.");

        let agent = create_corrective_rag_agent(
            Arc::clone(&retriever),
            grader,
            rewriter,
            generator.clone(),
            CragConfig::new(),
        )
        .unwrap();
        let final_state = agent
            .execute(
                CorrectiveRagState::from_query("alpha?"),
                &ExecutionContext::new(),
            )
            .await
            .unwrap()
            .state;

        assert_eq!(final_state.attempt, 0, "2/3 correct = above 0.5 → generate");
        let prompt = generator.observed();
        assert_eq!(prompt.len(), 1);
        // Last user message has documents — verify the Incorrect
        // doc is filtered out and only the two Correct ones land.
        let user_msg = prompt[0]
            .iter()
            .rfind(|m| matches!(m.role, Role::User))
            .unwrap();
        let docs_part = user_msg
            .content
            .iter()
            .find_map(|p| match p {
                ContentPart::Text { text, .. } if text.contains("documents") => Some(text.clone()),
                _ => None,
            })
            .unwrap();
        assert!(docs_part.contains("alpha is the first letter"));
        assert!(docs_part.contains("more about alpha letter"));
        assert!(!docs_part.contains("alpha is unrelated stuff"));
    }

    #[test]
    fn config_defaults_match_published_constants() {
        let cfg = CragConfig::default();
        assert!((cfg.min_correct_fraction() - DEFAULT_MIN_CORRECT_FRACTION).abs() < f32::EPSILON);
        assert_eq!(cfg.retrieval_top_k(), DEFAULT_RETRIEVAL_TOP_K);
        assert_eq!(cfg.max_rewrite_attempts(), DEFAULT_MAX_REWRITE_ATTEMPTS);
    }

    #[test]
    fn min_correct_fraction_clamped_during_routing() {
        // Out-of-range fractions don't crash routing — they
        // clamp to [0, 1] before the comparison.
        let cfg = CragConfig::new()
            .with_min_correct_fraction(2.5)
            .with_max_rewrite_attempts(0);
        let state = CorrectiveRagState {
            original_query: "q".into(),
            query: "q".into(),
            previous_attempts: vec![],
            graded: vec![(doc("d"), GradeVerdict::Correct)],
            correct_documents: vec![doc("d")],
            attempt: 0,
            answer: None,
        };
        // 1/1 = 1.0; even with threshold > 1.0 the clamp brings
        // it to 1.0 and 1.0 ≥ 1.0 so we route to generate.
        assert_eq!(route_after_grade(&state, &cfg), "generate");
    }
}
