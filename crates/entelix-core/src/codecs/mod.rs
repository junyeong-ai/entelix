//! Provider codecs (invariant 4).
//!
//! Each codec is a stateless encoder/decoder that operates on
//! `ModelRequest` / `ModelResponse` and produces / consumes a JSON
//! wire body — Anthropic Messages, OpenAI Chat, OpenAI Responses,
//! Gemini, Bedrock Converse. Stateful concerns (auth, conn pool,
//! retry) belong in `Transport`.
//!
//! Codecs MUST emit a `ModelWarning::LossyEncode` when the IR carries a
//! field they cannot preserve on the wire (invariant 6). Silent loss is a
//! bug.

mod anthropic;
mod bedrock_converse;
mod codec;
mod gemini;
mod openai_chat;
mod openai_responses;

pub use anthropic::AnthropicMessagesCodec;
pub use bedrock_converse::BedrockConverseCodec;
pub use codec::{BoxByteStream, BoxDeltaStream, Codec, EncodedRequest, extract_openai_rate_limit};
pub use gemini::GeminiCodec;
pub use openai_chat::OpenAiChatCodec;
pub use openai_responses::OpenAiResponsesCodec;
