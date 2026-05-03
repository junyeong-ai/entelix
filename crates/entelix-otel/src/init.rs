//! Convenience helpers for wiring `tracing` + `tracing-opentelemetry`
//! + `opentelemetry-otlp`. Gated behind the `otlp` cargo feature so
//! teams running their own observability bootstrap can skip the
//! exporter dep tree entirely.
//!
//! ## One-call setup
//!
//! ```ignore
//! use entelix_otel::init::{OtlpConfig, init_otlp};
//!
//! // Hold the handle for the lifetime of the process; spans flush
//! // when it drops.
//! let _otel = init_otlp(OtlpConfig::local())?;
//! ```
//!
//! [`init_otlp`] returns an [`OtlpHandle`] that owns the
//! [`SdkTracerProvider`] and flushes outstanding spans on drop —
//! callers do not need to remember to `shutdown` manually. Override
//! the endpoint / service name / log filter via the [`OtlpConfig`]
//! `with_*` builder chain when defaults are insufficient.

use std::sync::Arc;

use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::trace::SdkTracerProvider;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use entelix_core::error::Error;

use crate::semconv;

/// Configuration for the bundled tracing+OTLP bootstrap.
#[derive(Clone, Debug)]
pub struct OtlpConfig {
    /// OTLP collector endpoint (e.g. `http://localhost:4318`).
    pub endpoint: String,
    /// `service.name` resource attribute.
    pub service_name: String,
    /// Tracer scope name (`opentelemetry::global::tracer(name)` will
    /// receive this).
    pub tracer_name: String,
    /// `RUST_LOG`-style filter; `None` falls back to the env var or
    /// `info`.
    pub log_filter: Option<String>,
}

impl OtlpConfig {
    /// Sensible defaults — `http://localhost:4318`, service name
    /// `entelix`, tracer scope `entelix`.
    #[must_use]
    pub fn local() -> Self {
        Self {
            endpoint: "http://localhost:4318".into(),
            service_name: "entelix".into(),
            tracer_name: "entelix".into(),
            log_filter: None,
        }
    }

    /// Read endpoint + service name + log filter from the standard
    /// OTLP environment variables (`OTEL_EXPORTER_OTLP_ENDPOINT`,
    /// `OTEL_SERVICE_NAME`, `RUST_LOG`). Anything unset falls back to
    /// the same values [`Self::local`] uses.
    #[must_use]
    pub fn from_env() -> Self {
        let mut cfg = Self::local();
        if let Ok(endpoint) = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT") {
            cfg.endpoint = endpoint;
        }
        if let Ok(service) = std::env::var("OTEL_SERVICE_NAME") {
            cfg.service_name = service;
        }
        if let Ok(filter) = std::env::var("RUST_LOG") {
            cfg.log_filter = Some(filter);
        }
        cfg
    }

    /// Override the OTLP endpoint.
    #[must_use]
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Override the `service.name` resource attribute.
    #[must_use]
    pub fn with_service_name(mut self, name: impl Into<String>) -> Self {
        self.service_name = name.into();
        self
    }

    /// Override the tracer scope name.
    #[must_use]
    pub fn with_tracer_name(mut self, name: impl Into<String>) -> Self {
        self.tracer_name = name.into();
        self
    }

    /// Set the `RUST_LOG`-style filter.
    #[must_use]
    pub fn with_log_filter(mut self, filter: impl Into<String>) -> Self {
        self.log_filter = Some(filter.into());
        self
    }
}

/// RAII handle returned by [`init_otlp`]. Holds the
/// [`SdkTracerProvider`] alive for the lifetime of the process and
/// flushes outstanding spans when dropped — operators do not need
/// to remember to call `shutdown` explicitly. Cloning is cheap (the
/// inner provider is shared via `Arc`).
#[derive(Clone)]
pub struct OtlpHandle {
    provider: Arc<SdkTracerProvider>,
}

impl OtlpHandle {
    /// Borrow the underlying `SdkTracerProvider` — useful when the
    /// caller wants to register additional tracer scopes alongside
    /// the entelix bootstrap.
    #[must_use]
    pub fn provider(&self) -> &SdkTracerProvider {
        &self.provider
    }
}

impl std::fmt::Debug for OtlpHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OtlpHandle").finish_non_exhaustive()
    }
}

impl Drop for OtlpHandle {
    fn drop(&mut self) {
        // The handle is the only strong reference outside the global
        // tracer provider; on drop, flush whatever is still buffered.
        // Errors are surfaced via `tracing::warn!` rather than a
        // panic — Drop must not unwind.
        if Arc::strong_count(&self.provider) <= 1
            && let Err(e) = self.provider.shutdown()
        {
            tracing::warn!(
                target: "entelix_otel::init",
                error = ?e,
                "OTLP tracer provider shutdown failed"
            );
        }
    }
}

/// Wire `tracing` + `tracing-opentelemetry` + `opentelemetry-otlp`
/// in a single call.
///
/// Returns an [`OtlpHandle`] that flushes the exporter on drop —
/// hold it for the lifetime of the process. The returned handle is
/// cloneable (`Arc`-backed); cloning does not duplicate the
/// shutdown.
pub fn init_otlp(config: &OtlpConfig) -> Result<OtlpHandle, Error> {
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .with_endpoint(format!("{}/v1/traces", config.endpoint))
        .build()
        .map_err(|e| Error::config(format!("OTLP span exporter: {e}")))?;

    let resource = Resource::builder()
        .with_service_name(config.service_name.clone())
        .with_attribute(opentelemetry::KeyValue::new(
            "telemetry.sdk.name",
            "entelix-otel",
        ))
        .with_attribute(opentelemetry::KeyValue::new(
            "telemetry.sdk.language",
            "rust",
        ))
        .build();

    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(resource)
        .build();

    let tracer = provider.tracer(config.tracer_name.clone());

    let env_filter = match &config.log_filter {
        Some(f) => EnvFilter::new(f),
        None => EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
    };

    tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_opentelemetry::layer().with_tracer(tracer))
        .with(tracing_subscriber::fmt::layer())
        .try_init()
        .map_err(|e| Error::config(format!("set tracing subscriber: {e}")))?;

    opentelemetry::global::set_tracer_provider(provider.clone());

    tracing::debug!(
        target: "entelix_otel::init",
        service = %config.service_name,
        endpoint = %config.endpoint,
        sdk = %semconv::SYSTEM,
        "OTLP subscriber initialized"
    );

    Ok(OtlpHandle {
        provider: Arc::new(provider),
    })
}
