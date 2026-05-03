//! SigV4 request signer for AWS Bedrock.
//!
//! Wraps `aws-sigv4`'s low-level signing API into a single
//! `sign_request` call that takes a method/URL/headers/body and
//! returns the signed `HeaderMap` (with `authorization`,
//! `x-amz-date`, optional `x-amz-security-token` injected).
//!
//! The signer is stateless — credentials are passed per call so
//! callers can refresh the credential snapshot before each
//! invocation without rebuilding the signer.

use aws_credential_types::Credentials;
use aws_sigv4::http_request::{SignableBody, SignableRequest, SigningSettings, sign};
use aws_sigv4::sign::v4;

use crate::CloudError;

const SERVICE_NAME: &str = "bedrock";

/// Stateless SigV4 signer for Bedrock requests.
#[derive(Clone, Debug)]
pub struct BedrockSigner {
    region: String,
}

impl BedrockSigner {
    /// Build a signer for the given AWS region (`us-east-1`,
    /// `eu-west-1`, etc.).
    pub fn new(region: impl Into<String>) -> Self {
        Self {
            region: region.into(),
        }
    }

    /// Borrow the configured region.
    pub fn region(&self) -> &str {
        &self.region
    }

    /// Sign a request. Returns header `(name, value)` pairs that the
    /// caller appends to the outgoing request — the signer never
    /// owns the underlying HTTP client.
    pub fn sign_request(
        &self,
        creds: &Credentials,
        method: &str,
        url: &str,
        headers: &[(String, String)],
        body: &[u8],
    ) -> Result<Vec<(String, String)>, CloudError> {
        let identity = creds.clone().into();
        let signing_settings = SigningSettings::default();
        let signing_params = v4::SigningParams::builder()
            .identity(&identity)
            .region(&self.region)
            .name(SERVICE_NAME)
            .time(std::time::SystemTime::now())
            .settings(signing_settings)
            .build()
            .map_err(|e| CloudError::Signing {
                message: format!("build signing params: {e}"),
                source: Some(Box::new(e)),
            })?
            .into();

        let header_pairs: Vec<(&str, &str)> = headers
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_str()))
            .collect();
        let signable = SignableRequest::new(
            method,
            url,
            header_pairs.into_iter(),
            SignableBody::Bytes(body),
        )
        .map_err(|e| CloudError::Signing {
            message: format!("build signable request: {e}"),
            source: Some(Box::new(e)),
        })?;

        let (instructions, _signature) = sign(signable, &signing_params)
            .map_err(|e| CloudError::Signing {
                message: format!("sign: {e}"),
                source: Some(Box::new(e)),
            })?
            .into_parts();

        let mut out = Vec::new();
        let (sigv4_headers, _) = instructions.into_parts();
        for header in sigv4_headers {
            out.push((header.name().to_owned(), header.value().to_owned()));
        }
        Ok(out)
    }
}
