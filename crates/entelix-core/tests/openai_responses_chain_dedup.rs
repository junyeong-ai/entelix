//! Regression — when `ModelRequest::continued_from` carries an
//! `OpenAI` Responses `previous_response_id`, the codec must encode
//! only the messages appended after the last assistant turn. The
//! prior turn is represented server-side via the chain pointer;
//! re-sending it as input items duplicates context and cost.

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use entelix_core::codecs::{Codec, OpenAiResponsesCodec};
use entelix_core::ir::{
    ContentPart, Message, ModelRequest, ModelResponse, ProviderEchoSnapshot, Role, StopReason,
    Usage,
};
use serde_json::Value;

fn user_text(text: &str) -> Message {
    Message::new(Role::User, vec![ContentPart::text(text)])
}

fn assistant_text(text: &str) -> Message {
    Message::new(Role::Assistant, vec![ContentPart::text(text)])
}

fn prior_response(content_text: &str, response_id: &str) -> ModelResponse {
    ModelResponse {
        id: response_id.to_owned(),
        model: "gpt-4o".to_owned(),
        stop_reason: StopReason::EndTurn,
        content: vec![ContentPart::text(content_text)],
        usage: Usage::default(),
        rate_limit: None,
        warnings: Vec::new(),
        provider_echoes: vec![ProviderEchoSnapshot::for_provider(
            "openai-responses",
            "response_id",
            response_id,
        )],
    }
}

fn body_input_items(request: &ModelRequest) -> Vec<Value> {
    let encoded = OpenAiResponsesCodec::new().encode(request).unwrap();
    let body: Value = serde_json::from_slice(&encoded.body).unwrap();
    body.get("input").unwrap().as_array().unwrap().clone()
}

#[test]
fn chain_mode_skips_prior_assistant_and_pre_assistant_messages() {
    // Build the next request the way `continue_turn` does — push the
    // prior assistant turn, chain the response_id, push the new user
    // turn — then confirm the encoded `input` array carries only the
    // new user turn. The server-side previous_response_id pointer
    // owns everything up to (and including) the assistant turn.
    let prior = prior_response("prior assistant reply", "resp_abc");
    let request = ModelRequest {
        model: "gpt-4o".to_owned(),
        messages: vec![user_text("first user turn")],
        ..ModelRequest::default()
    }
    .continue_turn(&prior, user_text("next user turn"));

    let input_items = body_input_items(&request);
    assert_eq!(
        input_items.len(),
        1,
        "chain-mode encode should ship only the messages after the last assistant turn"
    );
    let only = &input_items[0];
    assert_eq!(only.get("role").and_then(Value::as_str), Some("user"));
    let content = only.get("content").unwrap().as_array().unwrap();
    let combined: String = content
        .iter()
        .filter_map(|c| c.get("text").and_then(Value::as_str))
        .collect();
    assert!(
        combined.contains("next user turn"),
        "only the new user content should ride the wire, got: {combined}"
    );
    assert!(
        !combined.contains("first user turn"),
        "pre-chain user turn must not be re-sent, got: {combined}"
    );
    assert!(
        !combined.contains("prior assistant reply"),
        "prior assistant turn must not be re-sent, got: {combined}"
    );
}

#[test]
fn non_chain_mode_carries_full_transcript() {
    // Without continued_from, the codec must encode every transcript
    // entry — the request has no server-side state to lean on.
    let request = ModelRequest {
        model: "gpt-4o".to_owned(),
        messages: vec![
            user_text("first user turn"),
            assistant_text("assistant turn"),
            user_text("next user turn"),
        ],
        ..ModelRequest::default()
    };
    let input_items = body_input_items(&request);
    // Two user items + one assistant item == 3 entries.
    assert_eq!(input_items.len(), 3, "full transcript should be encoded");
}

#[test]
fn chain_mode_with_tool_result_after_assistant_keeps_tool_message() {
    // Operator dispatched a tool round-trip — the tool result lives
    // after the assistant turn that called it. Chain-mode must
    // carry the tool result forward (it post-dates the chain anchor).
    let prior = prior_response("calling tool", "resp_xyz");
    let request = ModelRequest {
        model: "gpt-4o".to_owned(),
        messages: vec![user_text("ask")],
        ..ModelRequest::default()
    }
    .continue_turn(
        &prior,
        Message::new(
            Role::Tool,
            vec![ContentPart::ToolResult {
                tool_use_id: "call-1".to_owned(),
                name: "calc".to_owned(),
                content: entelix_core::ir::ToolResultContent::Text("42".to_owned()),
                is_error: false,
                cache_control: None,
                provider_echoes: Vec::new(),
            }],
        ),
    );
    let input_items = body_input_items(&request);
    assert_eq!(
        input_items.len(),
        1,
        "tool-result-only follow-up after the assistant turn should encode as one item"
    );
    let item = &input_items[0];
    assert_eq!(
        item.get("type").and_then(Value::as_str),
        Some("function_call_output"),
        "tool result encodes as function_call_output, got: {item}"
    );
}

#[test]
fn chain_mode_with_no_assistant_in_transcript_encodes_everything() {
    // Defensive: operator handed a `previous_response_id` but the
    // local transcript has no assistant message — the codec falls
    // through to full-transcript encoding rather than silently
    // dropping legitimate user content.
    let request = ModelRequest {
        model: "gpt-4o".to_owned(),
        messages: vec![user_text("only user")],
        continued_from: vec![ProviderEchoSnapshot::for_provider(
            "openai-responses",
            "response_id",
            "resp_no_assistant",
        )],
        ..ModelRequest::default()
    };
    let input_items = body_input_items(&request);
    assert_eq!(input_items.len(), 1, "user message must still encode");
}
