//! AWS Bedrock — `BedrockTransport`, SigV4 signing, IAM credential
//! chain, and the `vnd.amazon.eventstream` binary frame decoder used
//! by `:converse-stream` responses.

mod credential;
pub mod event_stream;
mod signer;
mod transport;

pub use credential::BedrockCredentialProvider;
pub use event_stream::{
    EventStreamDecoder, EventStreamFrame, EventStreamHeader, EventStreamHeaderValue,
    EventStreamParseError, encode_frame,
};
pub use signer::BedrockSigner;
pub use transport::{BedrockAuth, BedrockTransport, BedrockTransportBuilder};
