//! Regression — [`ModelRequest::continue_turn`] chains a prior turn's
//! assistant message, provider echoes, and the next user message into
//! the canonical multi-turn shape.
//!
//! The helper exists because forgetting any of the three pieces is a
//! silent failure for at least one vendor:
//!
//! - Anthropic extended thinking — drops `signature` echoes and the
//!   next call rejects with 401.
//! - Gemini thought signatures — drops continuation and reasoning
//!   state regresses silently.
//! - `OpenAI` Responses — drops `previous_response_id` and the chain
//!   restarts from scratch.
//!
//! The tests below pin the IR-level shape; codec-side round-trip is
//! covered by `tests/provider_echo_round_trip.rs`.

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use entelix_core::ir::{
    ContentPart, Message, ModelRequest, ModelResponse, ProviderEchoSnapshot, Role, StopReason,
    Usage,
};

fn assistant_text(text: &str) -> ContentPart {
    ContentPart::Text {
        text: text.to_owned(),
        cache_control: None,
        provider_echoes: Vec::new(),
    }
}

fn prior_response(content: Vec<ContentPart>, echoes: Vec<ProviderEchoSnapshot>) -> ModelResponse {
    ModelResponse {
        id: "resp-1".to_owned(),
        model: "claude-opus-4-7".to_owned(),
        stop_reason: StopReason::EndTurn,
        content,
        usage: Usage::default(),
        rate_limit: None,
        warnings: Vec::new(),
        provider_echoes: echoes,
    }
}

#[test]
fn appends_assistant_then_user_in_order() {
    let req = ModelRequest {
        model: "claude-opus-4-7".to_owned(),
        messages: vec![Message::user("first user turn")],
        ..ModelRequest::default()
    };
    let prior = prior_response(vec![assistant_text("first model reply")], Vec::new());
    let next = req.continue_turn(&prior, Message::user("second user turn"));

    assert_eq!(next.messages.len(), 3, "user → assistant → user");
    assert_eq!(next.messages[0].role, Role::User);
    assert_eq!(next.messages[1].role, Role::Assistant);
    assert_eq!(next.messages[2].role, Role::User);

    match &next.messages[1].content[0] {
        ContentPart::Text { text, .. } => assert_eq!(text, "first model reply"),
        other => panic!("expected text in assistant turn, got {other:?}"),
    }
    match &next.messages[2].content[0] {
        ContentPart::Text { text, .. } => assert_eq!(text, "second user turn"),
        other => panic!("expected text in next user turn, got {other:?}"),
    }
}

#[test]
fn chains_provider_echoes_from_response_to_request() {
    let anthropic_sig =
        ProviderEchoSnapshot::for_provider("anthropic-messages", "signature", "WaUjzkypQ2yIBQs=");
    let req = ModelRequest::default();
    let prior = prior_response(vec![assistant_text("ok")], vec![anthropic_sig.clone()]);

    let next = req.continue_turn(&prior, Message::user("continue"));

    assert_eq!(next.continued_from.len(), 1);
    assert_eq!(next.continued_from[0], anthropic_sig);
}

#[test]
fn carries_multiple_provider_echoes_unchanged() {
    let anthropic = ProviderEchoSnapshot::for_provider("anthropic-messages", "signature", "a");
    let openai =
        ProviderEchoSnapshot::for_provider("openai-responses", "previous_response_id", "resp_abc");
    let prior = prior_response(
        vec![assistant_text("multi-vendor")],
        vec![anthropic.clone(), openai.clone()],
    );

    let next = ModelRequest::default().continue_turn(&prior, Message::user("more"));
    assert_eq!(next.continued_from, vec![anthropic, openai]);
}

#[test]
fn replaces_existing_continued_from() {
    // A request that was already a continuation must adopt the new
    // turn's echoes — not concatenate stale ones from the prior turn.
    let stale = ProviderEchoSnapshot::for_provider("anthropic-messages", "signature", "stale");
    let fresh = ProviderEchoSnapshot::for_provider("anthropic-messages", "signature", "fresh");
    let req = ModelRequest {
        continued_from: vec![stale],
        ..ModelRequest::default()
    };
    let prior = prior_response(vec![assistant_text("turn")], vec![fresh.clone()]);

    let next = req.continue_turn(&prior, Message::user("again"));
    assert_eq!(next.continued_from, vec![fresh]);
}

#[test]
fn preserves_model_system_tools_and_sampling_knobs() {
    use std::sync::Arc;
    let req = ModelRequest {
        model: "gpt-4o".to_owned(),
        temperature: Some(0.7),
        max_tokens: Some(1024),
        top_p: Some(0.9),
        stop_sequences: vec!["END".to_owned()],
        tools: Arc::from(Vec::<entelix_core::ir::ToolSpec>::new()),
        ..ModelRequest::default()
    };
    let prior = prior_response(vec![assistant_text("ok")], Vec::new());
    let next = req.continue_turn(&prior, Message::user("again"));

    assert_eq!(next.model, "gpt-4o");
    assert_eq!(next.temperature, Some(0.7));
    assert_eq!(next.max_tokens, Some(1024));
    assert_eq!(next.top_p, Some(0.9));
    assert_eq!(next.stop_sequences, vec!["END".to_owned()]);
}

#[test]
fn empty_echoes_yields_empty_continued_from() {
    let prior = prior_response(vec![assistant_text("trivial")], Vec::new());
    let next = ModelRequest::default().continue_turn(&prior, Message::user("more"));
    assert!(next.continued_from.is_empty());
}

#[test]
fn carries_non_text_assistant_content_into_next_turn() {
    // A `ToolUse` block in the prior assistant turn must survive the
    // round-trip even though `continue_turn` doesn't synthesise the
    // matching tool result — agent loops append `Message::Tool`
    // results separately before calling `continue_turn` with the next
    // user turn.
    let prior_content = vec![
        assistant_text("calling tool"),
        ContentPart::ToolUse {
            id: "call-1".to_owned(),
            name: "calc".to_owned(),
            input: serde_json::json!({"a": 1, "b": 2}),
            provider_echoes: Vec::new(),
        },
    ];
    let prior = prior_response(prior_content, Vec::new());

    // Operator dispatched the tool and appended the result separately.
    let mut req = ModelRequest {
        messages: vec![
            Message::user("compute"),
            // continue_turn appends the assistant turn — followed by
            // the tool result the operator already pushed.
        ],
        ..ModelRequest::default()
    };
    let result_msg = Message::new(
        Role::Tool,
        vec![ContentPart::ToolResult {
            tool_use_id: "call-1".to_owned(),
            name: "calc".to_owned(),
            content: entelix_core::ir::ToolResultContent::Text("3".to_owned()),
            is_error: false,
            cache_control: None,
            provider_echoes: Vec::new(),
        }],
    );

    req = req.continue_turn(&prior, result_msg);
    // user → assistant(text+tool_use) → tool(result)
    assert_eq!(req.messages.len(), 3);
    assert!(matches!(
        req.messages[1].content[1],
        ContentPart::ToolUse { .. }
    ));
    assert_eq!(req.messages[2].role, Role::Tool);
}
