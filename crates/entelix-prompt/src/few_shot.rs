//! `FewShotPromptTemplate` — string-shape few-shot prompting.
//!
//! Renders to `String` (so it composes ahead of `ChatPromptTemplate` or
//! a single-string `PromptTemplate`-based chain). Examples are sourced
//! from either a fixed list or an [`ExampleSelector`](crate::ExampleSelector). The
//! suffix template receives the caller's runtime vars; the prefix is a
//! static instruction; the joiner is configurable (default `"\n\n"`).
//!
//! For chat-shaped prompts where each example is a (user, assistant)
//! message pair, see [`ChatFewShotPromptTemplate`].

use std::collections::HashMap;
use std::sync::Arc;

use entelix_core::ir::Message;
use entelix_core::{ExecutionContext, Result};
use entelix_runnable::Runnable;

use crate::chat::ChatPromptTemplate;
use crate::example_selector::{Example, FixedExampleSelector, SharedExampleSelector};
use crate::template::PromptTemplate;

const DEFAULT_EXAMPLE_SEPARATOR: &str = "\n\n";

/// String-shape few-shot prompt.
///
/// Layout: `prefix + sep + (example₁ + sep + example₂ + sep + …) + sep + suffix`.
/// Each example is rendered through `example_prompt`; the suffix is
/// rendered through `suffix_prompt` with the caller's runtime variables.
#[derive(Clone)]
pub struct FewShotPromptTemplate {
    selector: SharedExampleSelector,
    example_prompt: PromptTemplate,
    prefix: String,
    suffix_prompt: PromptTemplate,
    example_separator: String,
}

impl FewShotPromptTemplate {
    /// Build with a fixed list of examples — convenience over
    /// [`Self::with_selector`].
    pub fn with_examples(
        examples: Vec<Example>,
        example_prompt: PromptTemplate,
        suffix_prompt: PromptTemplate,
    ) -> Self {
        Self::with_selector(
            Arc::new(FixedExampleSelector::new(examples)),
            example_prompt,
            suffix_prompt,
        )
    }

    /// Build with a custom [`ExampleSelector`](crate::ExampleSelector) — pass an `Arc<dyn
    /// ExampleSelector>`.
    pub fn with_selector(
        selector: SharedExampleSelector,
        example_prompt: PromptTemplate,
        suffix_prompt: PromptTemplate,
    ) -> Self {
        Self {
            selector,
            example_prompt,
            prefix: String::new(),
            suffix_prompt,
            example_separator: DEFAULT_EXAMPLE_SEPARATOR.to_owned(),
        }
    }

    /// Set the static prefix (instruction text) prepended before any
    /// examples.
    #[must_use]
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// Override the joiner placed between examples and around the
    /// prefix/suffix. Default is `"\n\n"`.
    #[must_use]
    pub fn with_example_separator(mut self, sep: impl Into<String>) -> Self {
        self.example_separator = sep.into();
        self
    }

    /// Render the few-shot prompt to a single string. `input_vars` are
    /// the suffix variables; selectors may also see them.
    pub fn render(&self, input_vars: &HashMap<String, String>) -> Result<String> {
        let examples = self.selector.select(input_vars)?;
        let mut parts: Vec<String> = Vec::with_capacity(examples.len().saturating_add(2));
        if !self.prefix.is_empty() {
            parts.push(self.prefix.clone());
        }
        for example in &examples {
            parts.push(self.example_prompt.render(example)?);
        }
        parts.push(self.suffix_prompt.render(input_vars)?);
        Ok(parts.join(&self.example_separator))
    }
}

#[async_trait::async_trait]
impl Runnable<HashMap<String, String>, String> for FewShotPromptTemplate {
    async fn invoke(
        &self,
        input: HashMap<String, String>,
        _ctx: &ExecutionContext,
    ) -> Result<String> {
        self.render(&input)
    }
}

/// Chat-shape few-shot prompt.
///
/// Each example is rendered through an `example_chat` template into a
/// `Vec<Message>` (typically a (user, assistant) pair). The suffix
/// template provides the trailing prompt — usually the system message
/// plus the new user query. All rendered messages are concatenated.
#[derive(Clone)]
pub struct ChatFewShotPromptTemplate {
    selector: SharedExampleSelector,
    example_chat: ChatPromptTemplate,
    suffix_chat: ChatPromptTemplate,
}

impl ChatFewShotPromptTemplate {
    /// Build with a fixed list of examples.
    pub fn with_examples(
        examples: Vec<Example>,
        example_chat: ChatPromptTemplate,
        suffix_chat: ChatPromptTemplate,
    ) -> Self {
        Self::with_selector(
            Arc::new(FixedExampleSelector::new(examples)),
            example_chat,
            suffix_chat,
        )
    }

    /// Build with a custom [`ExampleSelector`](crate::ExampleSelector).
    pub fn with_selector(
        selector: SharedExampleSelector,
        example_chat: ChatPromptTemplate,
        suffix_chat: ChatPromptTemplate,
    ) -> Self {
        Self {
            selector,
            example_chat,
            suffix_chat,
        }
    }

    /// Render the chat-shape few-shot prompt to a `Vec<Message>`.
    pub fn render(&self, input_vars: &crate::chat::PromptVars) -> Result<Vec<Message>> {
        let key_vars = text_only_view(input_vars);
        let examples = self.selector.select(&key_vars)?;
        let mut out: Vec<Message> = Vec::new();
        for example in &examples {
            let example_vars = example
                .iter()
                .map(|(k, v)| (k.clone(), crate::chat::PromptValue::Text(v.clone())))
                .collect::<crate::chat::PromptVars>();
            out.extend(self.example_chat.render(&example_vars)?);
        }
        out.extend(self.suffix_chat.render(input_vars)?);
        Ok(out)
    }
}

#[async_trait::async_trait]
impl Runnable<crate::chat::PromptVars, Vec<Message>> for ChatFewShotPromptTemplate {
    async fn invoke(
        &self,
        input: crate::chat::PromptVars,
        _ctx: &ExecutionContext,
    ) -> Result<Vec<Message>> {
        self.render(&input)
    }
}

fn text_only_view(vars: &crate::chat::PromptVars) -> HashMap<String, String> {
    vars.iter()
        .filter_map(|(k, v)| match v {
            crate::chat::PromptValue::Text(s) => Some((k.clone(), s.clone())),
            crate::chat::PromptValue::Messages(_) => None,
        })
        .collect()
}
