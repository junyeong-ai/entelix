//! Output parsers — `Runnable`s that turn model output (`Message`) into
//! strongly-typed Rust values.
//!
//! Surface:
//! - [`JsonOutputParser<T>`] — direct serde-JSON deserializer.

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
