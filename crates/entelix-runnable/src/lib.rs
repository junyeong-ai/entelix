//! # entelix-runnable
//!
//! Composition contract for entelix (invariant 7). Defines
//! `Runnable<I, O>` and the universal `.pipe()` connector —
//! `Sequence`, `Parallel`, `Router`, `Lambda`, `Passthrough`, plus the
//! type-erased `AnyRunnable` for dynamic dispatch (F12 mitigation).
//!
//! ## Composition surface
//!
//! Every adapter on `RunnableExt` returns a concrete `Runnable<I, O>`
//! type — chains stay zero-cost in the steady state, with boxing only
//! at the explicit `erase()` boundary. The standard adapters:
//!
//! - `pipe(next)` — sequence two runnables
//! - `with_retry(policy)` — retry transient errors
//! - `with_fallbacks(others)` — ordered fallbacks on transient errors
//! - `map(fn)` — pure synchronous output transform
//! - `with_config(fn)` — branch-local `ExecutionContext` mutation
//! - `with_timeout(duration)` — wall-clock deadline override
//! - `stream_with(input, mode, ctx)` — convenience streaming entry

#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_root_url = "https://docs.rs/entelix-runnable/0.5.1")]
#![deny(missing_docs)]

mod adapter;
mod any_runnable;
mod chat;
mod configured;
mod ext;
mod fallback;
mod lambda;
mod mapping;
mod parallel;
mod parser;
mod passthrough;
mod retrying;
mod router;
mod runnable;
mod sequence;
pub mod stream;
mod structured;
mod timed;

pub use adapter::ToolToRunnableAdapter;
pub use any_runnable::{AnyRunnable, AnyRunnableHandle, erase};
pub use configured::Configured;
pub use ext::RunnableExt;
pub use fallback::Fallback;
pub use lambda::RunnableLambda;
pub use mapping::Mapping;
pub use parallel::RunnableParallel;
pub use parser::JsonOutputParser;
pub use passthrough::RunnablePassthrough;
pub use retrying::Retrying;
pub use router::RunnableRouter;
pub use runnable::Runnable;
pub use sequence::RunnableSequence;
pub use stream::{BoxStream, DebugEvent, RunnableEvent, StreamChunk, StreamMode};
pub use structured::{ChatModelExt, StructuredOutputAdapter};
pub use timed::Timed;
