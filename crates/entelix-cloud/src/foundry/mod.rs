//! Microsoft Azure AI Foundry — `FoundryTransport` plus the
//! `azure_identity`-backed credential adapter that drives
//! [`crate::refresh::CachedTokenProvider`].

mod credential;
mod transport;

pub use credential::{FOUNDRY_SCOPE, FoundryCredentialProvider};
pub use transport::{FoundryAuth, FoundryTransport, FoundryTransportBuilder};
