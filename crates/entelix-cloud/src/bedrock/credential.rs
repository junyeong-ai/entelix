//! AWS credential resolution chain.
//!
//! [`BedrockCredentialProvider`] wraps the standard AWS SDK
//! `aws-config` default chain (env vars → SSO → IRSA / Workload
//! Identity → IMDS) so the same chain that powers `aws s3` powers
//! Bedrock requests. Operators get IRSA out of the box on EKS and
//! `~/.aws/credentials` profile resolution on workstations.

use std::sync::Arc;

use aws_credential_types::Credentials;
use aws_credential_types::provider::ProvideCredentials;

use crate::CloudError;

/// Chain-aware credential provider for [`crate::bedrock::BedrockTransport`].
///
/// Internally an `Arc<dyn ProvideCredentials>` so the provider can be
/// shared across many transport instances without re-running the
/// resolver chain.
#[derive(Clone)]
pub struct BedrockCredentialProvider {
    inner: Arc<dyn ProvideCredentials>,
}

impl BedrockCredentialProvider {
    /// Wrap any AWS SDK `ProvideCredentials` impl. Tests use this to
    /// inject a static `Credentials` value; production typically goes
    /// through [`Self::default_chain`].
    pub fn from_provider(inner: Arc<dyn ProvideCredentials>) -> Self {
        Self { inner }
    }

    /// Resolve via the AWS SDK default credential chain — this is
    /// what most callers want.
    pub async fn default_chain() -> Self {
        let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
        let provider = config
            .credentials_provider()
            .expect("aws_config::load_defaults always installs a credentials provider");
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Resolve current credentials from the chain. The returned value
    /// is a snapshot — re-call before each signing if you want to
    /// honour mid-flight rotation.
    pub async fn resolve(&self) -> Result<Credentials, CloudError> {
        self.inner
            .provide_credentials()
            .await
            .map_err(CloudError::credential)
    }
}
