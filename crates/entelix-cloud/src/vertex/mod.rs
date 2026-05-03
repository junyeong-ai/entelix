//! Google Cloud Vertex AI — `VertexTransport` plus the `gcp_auth`
//! credential adapter that drives [`crate::refresh::CachedTokenProvider`].

mod credential;
mod transport;

pub use credential::{VERTEX_SCOPE, VertexCredentialProvider};
pub use transport::{VertexTransport, VertexTransportBuilder};
