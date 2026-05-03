//! `PromptTemplate` — Jinja2-syntax string template (minijinja-backed).
//!
//! Templates use the standard Jinja2 surface:
//!
//! - `{{ name }}` — variable substitution
//! - `{% if cond %}…{% endif %}` — conditionals
//! - `{% for item in list %}…{% endfor %}` — loops
//! - `{{ var | upper }}` / `{{ var | default("x") }}` — filters
//!
//! Missing variables surface as
//! [`Error::InvalidRequest`](entelix_core::Error::InvalidRequest) at
//! render time — the template is parsed in strict mode so silent
//! empty-string substitution never happens.

use std::collections::HashMap;
use std::sync::Arc;

use entelix_core::{Error, ExecutionContext, Result};
use entelix_runnable::Runnable;
use minijinja::{Environment, UndefinedBehavior};
use serde::Serialize;

/// Symbolic name the underlying environment uses for the user-supplied
/// template text. Hidden behind the `PromptTemplate` API.
const TEMPLATE_KEY: &str = "main";

/// A Jinja2-syntax template that renders to `String` from a variables
/// map.
///
/// Compilation cost is paid once at [`Self::new`]; rendering reuses
/// the compiled instructions. Cloning is cheap (the underlying
/// environment is `Arc`-shared).
#[derive(Clone)]
pub struct PromptTemplate {
    env: Arc<Environment<'static>>,
    /// Source text — kept for `Eq` and introspection.
    source: Arc<str>,
    /// Sorted variable names referenced by the template — populated
    /// once at construction so callers can introspect cheaply.
    variables: Vec<String>,
}

impl PromptTemplate {
    /// Compile a template string. Returns
    /// [`Error::InvalidRequest`](entelix_core::Error::InvalidRequest)
    /// on syntax errors.
    pub fn new(template: impl AsRef<str>) -> Result<Self> {
        let source: Arc<str> = Arc::from(template.as_ref().to_owned().into_boxed_str());
        let mut env = Environment::new();
        env.set_undefined_behavior(UndefinedBehavior::Strict);
        env.add_template_owned(TEMPLATE_KEY, source.as_ref().to_owned())
            .map_err(|e| Error::invalid_request(format!("template syntax: {e}")))?;
        let variables = {
            let tmpl = env
                .get_template(TEMPLATE_KEY)
                .map_err(|e| Error::invalid_request(format!("template lookup: {e}")))?;
            let mut list: Vec<String> = tmpl.undeclared_variables(true).into_iter().collect();
            list.sort();
            list
        };
        Ok(Self {
            env: Arc::new(env),
            source,
            variables,
        })
    }

    /// Names of root-level variables this template references,
    /// sorted. Nested attribute access (`{{ user.name }}`) reports
    /// the outer `user` key only — minijinja does not surface
    /// dotted paths.
    #[must_use]
    pub fn variables(&self) -> &[String] {
        &self.variables
    }

    /// Source text the template was compiled from.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Render against any [`Serialize`]-able context. Accepts a
    /// `HashMap<String, String>` (simple text vars), a
    /// `serde_json::Value` (nested JSON-shape vars supporting
    /// dotted access like `{{ user.name }}`), or any user-defined
    /// struct that derives `Serialize`. Returns
    /// [`Error::InvalidRequest`](entelix_core::Error::InvalidRequest)
    /// when any referenced variable is missing or the render fails.
    pub fn render<S: Serialize>(&self, vars: S) -> Result<String> {
        let tmpl = self
            .env
            .get_template(TEMPLATE_KEY)
            .map_err(|e| Error::invalid_request(format!("template lookup: {e}")))?;
        tmpl.render(vars)
            .map_err(|e| Error::invalid_request(format!("template render: {e}")))
    }
}

impl std::fmt::Debug for PromptTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PromptTemplate")
            .field("source", &self.source.as_ref())
            .field("variables", &self.variables)
            .finish_non_exhaustive()
    }
}

impl PartialEq for PromptTemplate {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source
    }
}

impl Eq for PromptTemplate {}

#[async_trait::async_trait]
impl Runnable<HashMap<String, String>, String> for PromptTemplate {
    async fn invoke(
        &self,
        input: HashMap<String, String>,
        _ctx: &ExecutionContext,
    ) -> Result<String> {
        self.render(&input)
    }
}

/// Pipe arbitrary JSON-shaped state through a template — the typical
/// composition path inside a `StateGraph` node where the working state
/// is already a `serde_json::Value`. Supports nested attribute access
/// (`{{ user.name }}`) and Jinja2 loops over arrays — capabilities the
/// `HashMap<String, String>` impl cannot express.
#[async_trait::async_trait]
impl Runnable<serde_json::Value, String> for PromptTemplate {
    async fn invoke(&self, input: serde_json::Value, _ctx: &ExecutionContext) -> Result<String> {
        self.render(&input)
    }
}
