//! # entelix-prompt
//!
//! Prompt primitives — `PromptTemplate`, `ChatPromptTemplate`,
//! `MessagesPlaceholder`, `FewShotPromptTemplate`,
//! `ChatFewShotPromptTemplate`, `ExampleSelector`. All implement
//! `Runnable` so they compose with codecs, parsers, tools and graphs
//! through `.pipe()`.
//!
//! Surface: `PromptTemplate`, `ChatPromptTemplate`,
//! `ChatPromptPart`, `MessagesPlaceholder`, `PromptValue`, `PromptVars`,
//! `FewShotPromptTemplate`, `ChatFewShotPromptTemplate`,
//! `ExampleSelector` + `FixedExampleSelector` +
//! `LengthBasedExampleSelector`.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_root_url = "https://docs.rs/entelix-prompt/1.0.0-rc.2")]
#![deny(missing_docs)]

mod chat;
mod example_selector;
mod few_shot;
mod template;

pub use chat::{ChatPromptPart, ChatPromptTemplate, MessagesPlaceholder, PromptValue, PromptVars};
pub use example_selector::{
    Example, ExampleSelector, FixedExampleSelector, LengthBasedExampleSelector,
    SharedExampleSelector,
};
pub use few_shot::{ChatFewShotPromptTemplate, FewShotPromptTemplate};
pub use template::PromptTemplate;
