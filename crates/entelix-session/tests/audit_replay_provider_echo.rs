//! Audit-replay preserves vendor opaque round-trip tokens.
//!
//! `SessionGraph::current_branch_messages` reconstructs `Vec<Message>`
//! from the event log — invariant 1 (session-as-event-SSoT) means a
//! fresh process replaying a thread MUST emit the same wire bytes the
//! model originally signed. `provider_echoes` carrying Anthropic
//! `signature`, Gemini `thought_signature`, OpenAI Responses
//! `encrypted_content` etc. must survive the replay verbatim.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::doc_markdown)]

use chrono::Utc;
use entelix_core::ir::{
    ContentPart, ProviderEchoSnapshot, Role, ToolResultContent, find_provider_echo,
};
use entelix_session::{GraphEvent, SessionGraph};

const ANTHROPIC_SIGNATURE: &str = "WaUjzkypQ2mUEVM36O2TxuC06KN8xyfbJwyem2dw3UR";
const GEMINI_THOUGHT_SIGNATURE: &str = "EhsMUkVHSU9OOmFzaWEtbm9ydGhlYXN0Mw";

#[test]
fn replay_preserves_thinking_provider_echoes_through_assistant_message_event() {
    let mut graph = SessionGraph::new("t-replay-thinking");
    graph.append(GraphEvent::UserMessage {
        content: vec![ContentPart::text("question")],
        timestamp: Utc::now(),
    });
    graph.append(GraphEvent::AssistantMessage {
        content: vec![ContentPart::Thinking {
            text: "internal reasoning".into(),
            cache_control: None,
            provider_echoes: vec![ProviderEchoSnapshot::for_provider(
                "anthropic-messages",
                "signature",
                ANTHROPIC_SIGNATURE,
            )],
        }],
        usage: None,
        timestamp: Utc::now(),
    });

    let messages = graph.current_branch_messages();
    let assistant = messages
        .iter()
        .find(|m| matches!(m.role, Role::Assistant))
        .expect("replayed assistant message must exist");
    let part = assistant.content.first().expect("part must exist");
    assert_eq!(
        find_provider_echo(part.provider_echoes(), "anthropic-messages")
            .and_then(|e| e.payload_str("signature")),
        Some(ANTHROPIC_SIGNATURE),
    );
}

#[test]
fn replay_preserves_tool_use_provider_echoes_through_assistant_message_event() {
    // Gemini 3.x attaches `thought_signature` to `functionCall` parts;
    // missing this on the next turn → HTTP 400. Audit replay must
    // round-trip the carrier alongside the tool_use itself so a
    // wake-from-checkpoint produces the same signed wire bytes.
    let mut graph = SessionGraph::new("t-replay-tool-use");
    graph.append(GraphEvent::AssistantMessage {
        content: vec![ContentPart::ToolUse {
            id: "get_weather#0".into(),
            name: "get_weather".into(),
            input: serde_json::json!({"city": "Seoul"}),
            provider_echoes: vec![ProviderEchoSnapshot::for_provider(
                "gemini",
                "thought_signature",
                GEMINI_THOUGHT_SIGNATURE,
            )],
        }],
        usage: None,
        timestamp: Utc::now(),
    });
    graph.append(GraphEvent::ToolResult {
        tool_use_id: "get_weather#0".into(),
        name: "get_weather".into(),
        content: ToolResultContent::Json(serde_json::json!({"temp": 15})),
        is_error: false,
        timestamp: Utc::now(),
    });

    let messages = graph.current_branch_messages();
    let assistant = messages
        .iter()
        .find(|m| matches!(m.role, Role::Assistant))
        .unwrap();
    let tool_use = assistant.content.first().expect("tool_use must exist");
    assert_eq!(
        find_provider_echo(tool_use.provider_echoes(), "gemini")
            .and_then(|e| e.payload_str("thought_signature")),
        Some(GEMINI_THOUGHT_SIGNATURE),
    );
}
