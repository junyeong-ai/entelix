//! `ChatPromptTemplate` — heterogeneous list of role-tagged templates and
//! `MessagesPlaceholder`s, rendering to `Vec<Message>`.
//!
//! Mirror of `LangChain`'s `ChatPromptTemplate.from_messages([...])` with a
//! Rust-idiomatic enum to express the heterogeneous input.

use std::collections::HashMap;

use entelix_core::ir::{ContentPart, Message, Role};
use entelix_core::{Error, ExecutionContext, Result};
use entelix_runnable::Runnable;

use crate::template::PromptTemplate;

/// One slot in a [`ChatPromptTemplate`]: either a templated message (role +
/// text template) or a placeholder for a `Vec<Message>` variable.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum ChatPromptPart {
    /// Role-tagged single-message template — renders to one `Message`.
    Templated {
        /// Role to assign to the rendered message.
        role: Role,
        /// Compiled template body.
        template: PromptTemplate,
    },
    /// Slot for a runtime-supplied conversation segment — typically the
    /// chat history. The variable's value must be a
    /// [`PromptValue::Messages`].
    Placeholder(MessagesPlaceholder),
}

impl ChatPromptPart {
    /// Build a system-role templated part.
    pub fn system(template: impl AsRef<str>) -> Result<Self> {
        Ok(Self::Templated {
            role: Role::System,
            template: PromptTemplate::new(template)?,
        })
    }

    /// Build a user-role templated part.
    pub fn user(template: impl AsRef<str>) -> Result<Self> {
        Ok(Self::Templated {
            role: Role::User,
            template: PromptTemplate::new(template)?,
        })
    }

    /// Build an assistant-role templated part.
    pub fn assistant(template: impl AsRef<str>) -> Result<Self> {
        Ok(Self::Templated {
            role: Role::Assistant,
            template: PromptTemplate::new(template)?,
        })
    }

    /// Build a placeholder for a `Vec<Message>` variable.
    pub fn placeholder(name: impl Into<String>) -> Self {
        Self::Placeholder(MessagesPlaceholder::new(name))
    }
}

/// Placeholder whose value at render time is a `Vec<Message>` (typically
/// chat history).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MessagesPlaceholder {
    name: String,
}

impl MessagesPlaceholder {
    /// Create a placeholder bound to `name`.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    /// Borrow the placeholder's variable name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// A variable's value: either text (for templated parts) or a list of
/// messages (for placeholders).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum PromptValue {
    /// Plain text — used by templated parts (`{var}`).
    Text(String),
    /// Pre-built messages — used by [`MessagesPlaceholder`] slots.
    Messages(Vec<Message>),
}

impl From<String> for PromptValue {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<&str> for PromptValue {
    fn from(s: &str) -> Self {
        Self::Text(s.to_owned())
    }
}

impl From<Vec<Message>> for PromptValue {
    fn from(v: Vec<Message>) -> Self {
        Self::Messages(v)
    }
}

/// Map of variable name → value, accepted by `ChatPromptTemplate::render`
/// and the `Runnable` impl.
pub type PromptVars = HashMap<String, PromptValue>;

/// Heterogeneous list of templated messages and placeholders. Renders to
/// `Vec<Message>` — the input type expected by [`entelix_core::ChatModel`].
#[derive(Clone, Debug)]
pub struct ChatPromptTemplate {
    parts: Vec<ChatPromptPart>,
}

impl ChatPromptTemplate {
    /// Build from an explicit list of parts.
    pub fn from_messages<I>(parts: I) -> Self
    where
        I: IntoIterator<Item = ChatPromptPart>,
    {
        Self {
            parts: parts.into_iter().collect(),
        }
    }

    /// Borrow the underlying part list.
    pub fn parts(&self) -> &[ChatPromptPart] {
        &self.parts
    }

    /// Render to `Vec<Message>`. Pulls text variables and message-list
    /// variables from `vars`. Missing variables, or variables of the wrong
    /// shape, surface as `Error::InvalidRequest`.
    pub fn render(&self, vars: &PromptVars) -> Result<Vec<Message>> {
        let mut out = Vec::with_capacity(self.parts.len());
        for part in &self.parts {
            match part {
                ChatPromptPart::Templated { role, template } => {
                    let text_vars = collect_text_vars(template.variables(), vars)?;
                    let rendered = template.render(&text_vars)?;
                    out.push(Message::new(*role, vec![ContentPart::text(rendered)]));
                }
                ChatPromptPart::Placeholder(p) => {
                    let messages = match vars.get(p.name()) {
                        Some(PromptValue::Messages(m)) => m.clone(),
                        Some(PromptValue::Text(_)) => {
                            return Err(Error::invalid_request(format!(
                                "placeholder '{}' expected Messages, got Text",
                                p.name()
                            )));
                        }
                        None => {
                            return Err(Error::invalid_request(format!(
                                "missing placeholder variable '{}'",
                                p.name()
                            )));
                        }
                    };
                    out.extend(messages);
                }
            }
        }
        Ok(out)
    }
}

#[async_trait::async_trait]
impl Runnable<PromptVars, Vec<Message>> for ChatPromptTemplate {
    async fn invoke(&self, input: PromptVars, _ctx: &ExecutionContext) -> Result<Vec<Message>> {
        self.render(&input)
    }
}

fn collect_text_vars(needed: &[String], vars: &PromptVars) -> Result<HashMap<String, String>> {
    let mut out = HashMap::with_capacity(needed.len());
    for name in needed {
        match vars.get(name) {
            Some(PromptValue::Text(s)) => {
                out.insert(name.clone(), s.clone());
            }
            Some(PromptValue::Messages(_)) => {
                return Err(Error::invalid_request(format!(
                    "variable '{name}' is Messages but template expects Text"
                )));
            }
            None => {
                return Err(Error::invalid_request(format!(
                    "missing prompt variable '{name}'"
                )));
            }
        }
    }
    Ok(out)
}
