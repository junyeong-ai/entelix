//! Example-selection strategies for [`crate::FewShotPromptTemplate`].
//!
//! Selectors convert the caller's input variables into a list of examples
//! (each example is a `HashMap<String, String>` mapping example-prompt
//! variable names to their values). Two concrete impls ship in 1.0:
//!
//! - [`FixedExampleSelector`] — returns the same static example list every
//!   call.
//! - [`LengthBasedExampleSelector`] — drops trailing examples until the
//!   rendered length fits a configurable character cap.
//!
//! A semantic selector (embedding-similarity) is deferred to 1.1 alongside
//! concrete `Embedder` impls.

use std::collections::HashMap;
use std::sync::Arc;

use entelix_core::Result;

use crate::template::PromptTemplate;

/// One few-shot example: a map of example-prompt variable name → value.
pub type Example = HashMap<String, String>;

/// Strategy that picks which examples to inject into a few-shot prompt.
///
/// The trait is async-free; selection is expected to be a fast in-process
/// computation (filter, length cap, embedding lookup once vector stores
/// arrive). Implementors that need IO can return a precomputed list.
pub trait ExampleSelector: Send + Sync {
    /// Select examples for an invocation. `input_vars` are the suffix
    /// variables the caller will pass at render time — selectors may use
    /// them to score relevance; `FixedExampleSelector` ignores them.
    fn select(&self, input_vars: &HashMap<String, String>) -> Result<Vec<Example>>;
}

/// Returns the configured examples verbatim, regardless of input.
#[derive(Clone, Debug)]
pub struct FixedExampleSelector {
    examples: Vec<Example>,
}

impl FixedExampleSelector {
    /// Build a selector from a static example list.
    pub const fn new(examples: Vec<Example>) -> Self {
        Self { examples }
    }

    /// Borrow the underlying example list.
    pub fn examples(&self) -> &[Example] {
        &self.examples
    }
}

impl ExampleSelector for FixedExampleSelector {
    fn select(&self, _input_vars: &HashMap<String, String>) -> Result<Vec<Example>> {
        Ok(self.examples.clone())
    }
}

/// Selector that caps the rendered example total at a character budget.
///
/// Drops trailing examples until the rendered character length fits.
/// The cap measures raw UTF-8 character count over the concatenated
/// rendered examples joined by `\n` — coarse but deterministic and
/// dependency-free.
#[derive(Clone, Debug)]
pub struct LengthBasedExampleSelector {
    examples: Vec<Example>,
    example_prompt: PromptTemplate,
    max_chars: usize,
}

impl LengthBasedExampleSelector {
    /// Build a length-based selector. `example_prompt` is the same
    /// template used by the parent `FewShotPromptTemplate`; rendered
    /// length is what counts.
    pub const fn new(
        examples: Vec<Example>,
        example_prompt: PromptTemplate,
        max_chars: usize,
    ) -> Self {
        Self {
            examples,
            example_prompt,
            max_chars,
        }
    }

    /// Borrow the example pool.
    pub fn examples(&self) -> &[Example] {
        &self.examples
    }
}

impl ExampleSelector for LengthBasedExampleSelector {
    fn select(&self, _input_vars: &HashMap<String, String>) -> Result<Vec<Example>> {
        let mut total = 0usize;
        let mut kept: Vec<Example> = Vec::new();
        for example in &self.examples {
            let rendered = self.example_prompt.render(example)?;
            let len = rendered.chars().count();
            if total.saturating_add(len) > self.max_chars {
                break;
            }
            total = total.saturating_add(len);
            kept.push(example.clone());
        }
        Ok(kept)
    }
}

/// Convenience type-alias for selectors handed around as trait objects.
pub type SharedExampleSelector = Arc<dyn ExampleSelector>;
