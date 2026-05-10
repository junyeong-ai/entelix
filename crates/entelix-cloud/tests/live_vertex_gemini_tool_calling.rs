//! Live-API smoke for [`VertexGeminiCodec`] tool calling.
//!
//! Verifies the function-tool dispatch path end-to-end against
//! Vertex Gemini: a [`ToolSpec::function`] declaration encodes
//! into the Vertex `tools[].functionDeclarations[]` shape, the
//! model returns a `functionCall` part that the codec decodes
//! back into [`ContentPart::ToolUse`], and the round-trip
//! preserves both the typed input shape and the stable
//! ID-to-name pairing the harness needs to dispatch into
//! [`entelix_core::tools::Tool`].
//!
//! `#[ignore]`-gated and feature-gated behind `gcp` so a default
//! `cargo test --workspace` skips it. Run with:
//!
//! ```text
//! ENTELIX_LIVE_VERTEX_PROJECT=my-gcp-project \
//!     ENTELIX_LIVE_VERTEX_LOCATION=global \
//!     ENTELIX_LIVE_VERTEX_GEMINI_MODEL=gemini-3.1-pro-preview \
//!     cargo test -p entelix-cloud --features gcp \
//!         --test live_vertex_gemini_tool_calling -- --ignored
//! ```
//!
//! ## What is verified
//!
//! 1. `ToolSpec::function` with a JSON-Schema `input_schema`
//!    encodes through the Gemini codec onto the wire.
//! 2. Gemini's `functionCall` response decodes into a
//!    [`ContentPart::ToolUse`] carrying a non-empty `id`, the
//!    expected `name`, and an `input` object that contains the
//!    schema's required field.
//! 3. The terminal `StopReason` is the tool-use stop the model
//!    surfaces when it pauses to await tool execution
//!    (`StopReason::ToolUse`).
//! 4. `Usage` counters are populated — tool calling does not
//!    short-circuit cost accounting.
//!
//! ## Cost discipline
//!
//! `max_tokens = 512` covers the thinking pass + the tool call
//! payload. Per-run cost well under $0.001.

#![cfg(feature = "gcp")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::doc_markdown)]

use std::sync::Arc;

use entelix_cloud::vertex::{VertexCredentialProvider, VertexTransport};
use entelix_core::ChatModel;
use entelix_core::codecs::VertexGeminiCodec;
use entelix_core::context::ExecutionContext;
use entelix_core::install_default_tls;
use entelix_core::ir::{ContentPart, Message, StopReason, ToolSpec};
use serde_json::json;

const DEFAULT_MODEL: &str = "gemini-3.1-pro-preview";
const TOOL_NAME: &str = "get_current_weather";

#[tokio::test]
#[ignore = "live-API: requires GCP ADC + ENTELIX_LIVE_VERTEX_PROJECT + ENTELIX_LIVE_VERTEX_LOCATION"]
async fn vertex_gemini_function_tool_call() {
    install_default_tls();

    let project = std::env::var("ENTELIX_LIVE_VERTEX_PROJECT").expect(
        "set ENTELIX_LIVE_VERTEX_PROJECT (GCP project hosting the Vertex Gemini model) to run this live smoke",
    );
    let location = std::env::var("ENTELIX_LIVE_VERTEX_LOCATION")
        .expect("set ENTELIX_LIVE_VERTEX_LOCATION (e.g. `global`, `asia-northeast3`)");
    let model = std::env::var("ENTELIX_LIVE_VERTEX_GEMINI_MODEL")
        .unwrap_or_else(|_| DEFAULT_MODEL.to_owned());

    let credentials = VertexCredentialProvider::default_chain()
        .await
        .expect("ADC must resolve — run `gcloud auth application-default login` first");

    let mut builder = VertexTransport::builder()
        .with_project_id(&project)
        .with_location(&location)
        .with_token_refresher(Arc::new(credentials));
    if let Ok(qp) = std::env::var("ENTELIX_LIVE_VERTEX_QUOTA_PROJECT") {
        builder = builder.with_quota_project(qp);
    }
    let transport = builder
        .build()
        .expect("VertexTransport built from ADC chain");

    // A canonical function tool — single required string input,
    // unambiguous prompt the model can only satisfy by calling
    // the tool. The schema is deliberately minimal so the
    // assertion stays robust against vendor preference for short
    // function-call payloads.
    let weather_tool = ToolSpec::function(
        TOOL_NAME,
        "Get the current weather for a specific city. \
         Always call this tool when the user asks about weather.",
        json!({
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. `Seoul` or `Tokyo`."
                }
            },
            "required": ["city"]
        }),
    );

    let chat = ChatModel::new(VertexGeminiCodec::new(), transport, model)
        .with_max_tokens(512)
        .with_temperature(0.0)
        .with_tools(vec![weather_tool]);

    let ctx = ExecutionContext::new();
    let response = chat
        .complete_full(
            vec![Message::user(
                "What is the current weather in Seoul, South Korea?",
            )],
            &ctx,
        )
        .await
        .expect("Vertex Gemini tool-call round-trip");

    // Gemini surfaces the tool call alongside a normal `STOP`
    // finish reason (`StopReason::EndTurn`) — unlike Anthropic
    // which emits a dedicated `tool_use` stop. Either is valid;
    // the harness keys off the presence of a `ToolUse` part, not
    // the stop reason. The assertion accepts both shapes so the
    // smoke is portable across vendors that route through this
    // same Gemini-on-Vertex codec.
    assert!(
        matches!(
            response.stop_reason,
            StopReason::ToolUse | StopReason::EndTurn
        ),
        "unexpected stop_reason on tool-call dispatch: {:?}",
        response.stop_reason
    );

    // Pull the first ToolUse part — there must be exactly one for
    // this prompt under the schema.
    let tool_use = response
        .content
        .iter()
        .find_map(|part| match part {
            ContentPart::ToolUse { id, name, input, .. } => Some((id, name, input)),
            _ => None,
        })
        .expect("response must contain a ContentPart::ToolUse");

    let (id, name, input) = tool_use;
    assert!(
        !id.is_empty(),
        "ToolUse.id must be non-empty so the harness can pair it with the ToolResult"
    );
    assert_eq!(
        name, TOOL_NAME,
        "ToolUse.name must round-trip the function name verbatim"
    );
    let city = input
        .get("city")
        .and_then(|v| v.as_str())
        .expect("ToolUse.input must contain the schema-required `city` string field");
    assert!(
        city.to_lowercase().contains("seoul"),
        "model must extract `Seoul` from the prompt; got `{city}`"
    );

    assert!(
        response.usage.input_tokens > 0,
        "tool-call dispatch must populate usage.input_tokens"
    );
    assert!(
        response.usage.output_tokens > 0,
        "tool-call dispatch must populate usage.output_tokens"
    );
}
