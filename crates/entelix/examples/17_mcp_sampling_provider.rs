//! `17_mcp_sampling_provider` — MCP `sampling/createMessage`
//! server-initiated requests dispatched through an
//! `entelix_core::ChatModel<C, T>` via the
//! `entelix-mcp-chatmodel` companion adapter.
//!
//! Build: `cargo build --example 17_mcp_sampling_provider -p entelix`
//! Run:   `cargo run   --example 17_mcp_sampling_provider -p entelix`
//!
//! Demonstrates the operator-side wiring of MCP sampling
//! (ADR-0054 trait surface, ADR-0060 ChatModel bridge):
//!
//! 1. A stub `Codec` + `Transport` stand in for a real LLM. The
//!    stub echoes the last user message as the assistant reply so
//!    the example runs deterministically without any API key.
//! 2. `ChatModel::new(codec, transport, model)` builds the chat
//!    surface; `ChatModelSamplingProvider::new(chat)` wraps it
//!    behind the MCP `SamplingProvider` trait.
//! 3. We construct a `SamplingRequest` mimicking what an MCP
//!    server would push (system_prompt + temperature + max_tokens
//!    overrides) and call `provider.sample(request)` directly —
//!    the same code path that `McpServerConfig::with_sampling_provider`
//!    drives in production when the server's background SSE
//!    listener delivers a `sampling/createMessage` request.
//! 4. The response carries the model identifier echoed back, the
//!    `endTurn` stop reason, and the assistant content the model
//!    produced.
//!
//! No LLM dependency — runs deterministically in CI like
//! `03_state_graph` and `13_mcp_tools`.

#![allow(
    clippy::print_stdout,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::doc_markdown
)]

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use bytes::Bytes;
use entelix::codecs::{Codec, EncodedRequest};
use entelix::ir::{
    Capabilities, ContentPart, ModelRequest, ModelResponse, ModelWarning, StopReason, Usage,
};
use entelix::transports::{Transport, TransportResponse};
use entelix::{
    ChatModel, ChatModelSamplingProvider, ExecutionContext, SamplingContent, SamplingMessage,
    SamplingProvider, SamplingRequest,
};

/// Stub codec — captures the encode-time `ModelRequest` so the
/// decode side can echo the last user text back as an assistant
/// reply. Lets the example show per-request override application
/// (system_prompt / temperature / max_tokens) without a real LLM.
#[derive(Debug, Default)]
struct EchoCodec {
    captured: Arc<Mutex<Option<ModelRequest>>>,
}

impl Codec for EchoCodec {
    fn name(&self) -> &'static str {
        "echo"
    }
    fn capabilities(&self, _model: &str) -> Capabilities {
        Capabilities::default()
    }
    fn encode(&self, request: &ModelRequest) -> entelix_core::Result<EncodedRequest> {
        *self.captured.lock().unwrap() = Some(request.clone());
        Ok(EncodedRequest::post_json(
            "/echo",
            Bytes::from_static(b"{}"),
        ))
    }
    fn decode(
        &self,
        _body: &[u8],
        warnings_in: Vec<ModelWarning>,
    ) -> entelix_core::Result<ModelResponse> {
        let req = self.captured.lock().unwrap().clone().unwrap();
        let last_user = req
            .messages
            .last()
            .and_then(|m| {
                m.content.iter().find_map(|c| match c {
                    ContentPart::Text { text, .. } => Some(text.clone()),
                    _ => None,
                })
            })
            .unwrap_or_default();
        Ok(ModelResponse {
            id: "echo-1".into(),
            model: req.model,
            stop_reason: StopReason::EndTurn,
            content: vec![ContentPart::Text {
                text: format!("(echo) {last_user}"),
                cache_control: None,
            }],
            usage: Usage::default(),
            rate_limit: None,
            warnings: warnings_in,
        })
    }
}

#[derive(Debug)]
struct NoopTransport;

#[async_trait]
impl Transport for NoopTransport {
    fn name(&self) -> &'static str {
        "noop"
    }
    async fn send(
        &self,
        _request: EncodedRequest,
        _ctx: &ExecutionContext,
    ) -> entelix_core::Result<TransportResponse> {
        Ok(TransportResponse {
            status: 200,
            headers: http::HeaderMap::new(),
            body: Bytes::from_static(b"{}"),
        })
    }
}

#[tokio::main]
async fn main() -> entelix_core::Result<()> {
    // ── Build the chat model. In production this is the same chat
    //    model the agent uses for its own reasoning — sampling
    //    surface and agent surface share one model + one layer
    //    stack (PolicyLayer, OtelLayer, retry middleware).
    let chat =
        ChatModel::new(EchoCodec::default(), NoopTransport, "demo-model").with_max_tokens(2048);

    // ── Wrap it as an MCP `SamplingProvider`. The companion
    //    adapter handles IR conversion both ways and surfaces the
    //    per-request overrides the MCP server pushes.
    let provider = Arc::new(ChatModelSamplingProvider::new(chat));

    // In production this provider would be wired via
    //   `McpServerConfig::http(url).with_sampling_provider(provider)`
    // and called from the MCP background SSE listener whenever the
    // server emits `sampling/createMessage`. For demo purposes we
    // call it directly — same code path, no MCP transport required.

    println!("=== request ===");
    let request = SamplingRequest {
        messages: vec![SamplingMessage {
            role: "user".into(),
            content: SamplingContent::Text {
                text: "what's the most concise answer?".into(),
            },
        }],
        model_preferences: None,
        // Per-request override — the wrapped chat model's default
        // (no system) is shadowed by this prompt for THIS dispatch
        // only. Concurrent dispatches see their own overrides.
        system_prompt: Some("be terse".into()),
        include_context: None,
        // f64 → f32 inside the adapter (vendor-bounded loss).
        temperature: Some(0.2),
        // Override the chat-model default (2048) for this call.
        max_tokens: Some(64),
        stop_sequences: vec!["STOP".into()],
        metadata: None,
    };
    println!(
        "  system_prompt: {:?}",
        request.system_prompt.as_deref().unwrap_or("(none)")
    );
    println!("  temperature:   {:?}", request.temperature.unwrap());
    println!("  max_tokens:    {:?}", request.max_tokens.unwrap());
    println!(
        "  user text:     {:?}",
        match &request.messages[0].content {
            SamplingContent::Text { text } => text,
            _ => "(non-text)",
        }
    );

    let response = provider
        .sample(request)
        .await
        .map_err(|e| entelix_core::Error::config(format!("sampling failed: {e}")))?;

    println!();
    println!("=== response ===");
    println!("  model:       {}", response.model);
    println!("  stop_reason: {}", response.stop_reason);
    println!("  role:        {}", response.role);
    let assistant_text = match &response.content {
        SamplingContent::Text { text } => text.clone(),
        SamplingContent::Image { mime_type, .. } => format!("(image {mime_type})"),
        SamplingContent::Audio { mime_type, .. } => format!("(audio {mime_type})"),
        _ => "(other content)".into(),
    };
    println!("  content:     {assistant_text}");

    Ok(())
}
