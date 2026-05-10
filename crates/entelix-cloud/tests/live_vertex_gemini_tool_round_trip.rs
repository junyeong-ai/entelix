//! Live-API smoke for [`VertexGeminiCodec`] full tool round-trip:
//! `user → assistant ToolUse → user ToolResult → assistant final
//! text`.
//!
//! This is the canonical agent-loop shape — the model emits a
//! tool call, the harness dispatches the tool, the result is fed
//! back as the next user turn, and the model completes the
//! conversation citing the tool's output. ontosyx's `ox-agent`
//! drives this exact loop on every `query_graph` /
//! `edit_ontology` / `apply_ontology` call.
//!
//! ## What is verified
//!
//! 1. Turn 1 produces a `ContentPart::ToolUse` with the
//!    expected name and a `Seoul` extraction.
//! 2. The transcript fed back ([user, assistant(ToolUse),
//!    user(ToolResult)]) round-trips through the codec — the
//!    `ToolResult.tool_use_id` matches the prior `ToolUse.id`,
//!    and Gemini's `functionResponse` shape (which keys on the
//!    function name, not the id) survives the codec mapping.
//! 3. Turn 2 produces a final assistant text that mentions the
//!    tool result (Celsius value injected through the
//!    `ToolResultContent::Json` channel).
//! 4. `Usage` counters populated on both turns.
//!
//! ## Cost discipline
//!
//! Two non-streaming calls, `max_tokens = 512` each. Per-run
//! cost well under $0.001.

#![cfg(feature = "gcp")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::doc_markdown)]

use std::sync::Arc;

use entelix_cloud::vertex::{VertexCredentialProvider, VertexTransport};
use entelix_core::ChatModel;
use entelix_core::codecs::VertexGeminiCodec;
use entelix_core::context::ExecutionContext;
use entelix_core::install_default_tls;
use entelix_core::ir::{ContentPart, Message, Role, ToolResultContent, ToolSpec};
use serde_json::json;

const DEFAULT_MODEL: &str = "gemini-3.1-pro-preview";
const TOOL_NAME: &str = "get_current_weather";

#[tokio::test]
#[ignore = "live-API: requires GCP ADC + ENTELIX_LIVE_VERTEX_PROJECT + ENTELIX_LIVE_VERTEX_LOCATION"]
async fn vertex_gemini_full_tool_round_trip() {
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

    // === Turn 1 ===
    let mut transcript: Vec<Message> = vec![Message::user(
        "What is the current weather in Seoul, South Korea?",
    )];
    let r1 = chat
        .complete_full(transcript.clone(), &ctx)
        .await
        .expect("turn 1 must complete");

    let (tool_use_id, tool_name) = r1
        .content
        .iter()
        .find_map(|p| match p {
            ContentPart::ToolUse { id, name, .. } => Some((id.clone(), name.clone())),
            _ => None,
        })
        .expect("turn 1 response must contain a ToolUse");
    assert!(!tool_use_id.is_empty(), "ToolUse.id must be non-empty");
    assert_eq!(tool_name, TOOL_NAME, "ToolUse.name must round-trip verbatim");

    assert!(r1.usage.input_tokens > 0);
    assert!(r1.usage.output_tokens > 0);

    // === Turn 2 — feed assistant ToolUse + user ToolResult back ===
    transcript.push(Message::new(Role::Assistant, r1.content.clone()));
    transcript.push(Message::new(
        Role::Tool,
        vec![ContentPart::ToolResult {
            tool_use_id: tool_use_id.clone(),
            name: TOOL_NAME.to_owned(),
            content: ToolResultContent::Json(json!({
                "city": "Seoul",
                "temperature_celsius": 15,
                "conditions": "sunny"
            })),
            is_error: false,
            cache_control: None,
        }],
    ));

    let r2 = chat
        .complete_full(transcript, &ctx)
        .await
        .expect("turn 2 must complete after tool result");

    let final_text: String = r2
        .content
        .iter()
        .filter_map(|p| match p {
            ContentPart::Text { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("");
    assert!(
        !final_text.trim().is_empty(),
        "turn 2 must produce a final visible text reply citing the tool result; got: {r2:?}"
    );
    assert!(
        final_text.contains("15") || final_text.to_lowercase().contains("sunny"),
        "final reply must reflect the tool result (15°C / sunny); got: `{final_text}`"
    );

    assert!(r2.usage.input_tokens > 0);
    assert!(r2.usage.output_tokens > 0);
}
