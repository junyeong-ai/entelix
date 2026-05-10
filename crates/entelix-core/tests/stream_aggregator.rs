//! `StreamAggregator` tests.

#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::expect_used)]

use entelix_core::ir::{ContentPart, ModelWarning, StopReason, Usage};
use entelix_core::stream::{StreamAggregator, StreamDelta};
use entelix_core::{Error, Result};

#[test]
fn happy_path_text_only_response() -> Result<()> {
    let mut agg = StreamAggregator::new();
    agg.push(StreamDelta::Start {
        id: "msg_01".into(),
        model: "claude-opus-4-7".into(),
    })?;
    agg.push(StreamDelta::TextDelta {
        text: "Hello".into(),
        provider_echoes: Vec::new(),
    })?;
    agg.push(StreamDelta::TextDelta {
        text: ", world".into(),
        provider_echoes: Vec::new(),
    })?;
    agg.push(StreamDelta::Usage(Usage::new(5, 2)))?;
    agg.push(StreamDelta::Stop {
        stop_reason: StopReason::EndTurn,
    })?;

    let resp = agg.finalize()?;
    assert_eq!(resp.id, "msg_01");
    assert_eq!(resp.model, "claude-opus-4-7");
    assert_eq!(resp.content.len(), 1);
    match &resp.content[0] {
        ContentPart::Text { text, .. } => assert_eq!(text, "Hello, world"),
        other => panic!("expected text, got {other:?}"),
    }
    assert_eq!(resp.usage.output_tokens, 2);
    assert!(matches!(resp.stop_reason, StopReason::EndTurn));
    Ok(())
}

#[test]
fn tool_use_block_buffers_then_parses_input() -> Result<()> {
    let mut agg = StreamAggregator::new();
    agg.push(StreamDelta::Start {
        id: "m".into(),
        model: "x".into(),
    })?;
    agg.push(StreamDelta::TextDelta {
        text: "Sure, calling tool".into(),
        provider_echoes: Vec::new(),
    })?;
    agg.push(StreamDelta::ToolUseStart {
        id: "toolu_99".into(),
        name: "calculator".into(),
        provider_echoes: Vec::new(),
    })?;
    agg.push(StreamDelta::ToolUseInputDelta {
        partial_json: "{\"expr\":".into(),
    })?;
    agg.push(StreamDelta::ToolUseInputDelta {
        partial_json: "\"2+2\"}".into(),
    })?;
    agg.push(StreamDelta::ToolUseStop)?;
    agg.push(StreamDelta::Stop {
        stop_reason: StopReason::ToolUse,
    })?;

    let resp = agg.finalize()?;
    // Text first, then tool_use — order preserved.
    assert_eq!(resp.content.len(), 2);
    assert!(matches!(resp.content[0], ContentPart::Text { .. }));
    match &resp.content[1] {
        ContentPart::ToolUse {
            id, name, input, ..
        } => {
            assert_eq!(id, "toolu_99");
            assert_eq!(name, "calculator");
            assert_eq!(input["expr"], "2+2");
        }
        other => panic!("expected tool_use, got {other:?}"),
    }
    Ok(())
}

#[test]
fn empty_tool_input_defaults_to_empty_object() -> Result<()> {
    let mut agg = StreamAggregator::new();
    agg.push(StreamDelta::Start {
        id: "m".into(),
        model: "x".into(),
    })?;
    agg.push(StreamDelta::ToolUseStart {
        id: "t".into(),
        name: "noop".into(),
        provider_echoes: Vec::new(),
    })?;
    agg.push(StreamDelta::ToolUseStop)?;
    agg.push(StreamDelta::Stop {
        stop_reason: StopReason::ToolUse,
    })?;
    let resp = agg.finalize()?;
    match &resp.content[0] {
        ContentPart::ToolUse { input, .. } => {
            assert_eq!(input, &serde_json::json!({}));
        }
        other => panic!("got {other:?}"),
    }
    Ok(())
}

#[test]
fn multiple_tool_blocks_preserve_order() -> Result<()> {
    let mut agg = StreamAggregator::new();
    agg.push(StreamDelta::Start {
        id: "m".into(),
        model: "x".into(),
    })?;
    agg.push(StreamDelta::ToolUseStart {
        id: "a".into(),
        name: "one".into(),
        provider_echoes: Vec::new(),
    })?;
    agg.push(StreamDelta::ToolUseStop)?;
    agg.push(StreamDelta::ToolUseStart {
        id: "b".into(),
        name: "two".into(),
        provider_echoes: Vec::new(),
    })?;
    agg.push(StreamDelta::ToolUseStop)?;
    agg.push(StreamDelta::Stop {
        stop_reason: StopReason::ToolUse,
    })?;

    let resp = agg.finalize()?;
    assert_eq!(resp.content.len(), 2);
    let names: Vec<_> = resp
        .content
        .iter()
        .filter_map(|p| match p {
            ContentPart::ToolUse { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(names, vec!["one", "two"]);
    Ok(())
}

#[test]
fn warnings_collected_in_order() -> Result<()> {
    let mut agg = StreamAggregator::new();
    agg.push(StreamDelta::Start {
        id: "m".into(),
        model: "x".into(),
    })?;
    agg.push(StreamDelta::Warning(ModelWarning::LossyEncode {
        field: "a".into(),
        detail: "first".into(),
    }))?;
    agg.push(StreamDelta::Warning(ModelWarning::UnknownStopReason {
        raw: "weird".into(),
    }))?;
    agg.push(StreamDelta::TextDelta {
        text: "ok".into(),
        provider_echoes: Vec::new(),
    })?;
    // Emit a Usage delta — without it, finalize would attach a
    // synthetic `LossyEncode` warning about missing usage and this
    // test would conflate two semantics.
    agg.push(StreamDelta::Usage(Usage::new(1, 1)))?;
    agg.push(StreamDelta::Stop {
        stop_reason: StopReason::EndTurn,
    })?;

    let resp = agg.finalize()?;
    assert_eq!(resp.warnings.len(), 2);
    assert!(matches!(resp.warnings[0], ModelWarning::LossyEncode { .. }));
    assert!(matches!(
        resp.warnings[1],
        ModelWarning::UnknownStopReason { .. }
    ));
    Ok(())
}

#[test]
fn finalize_without_usage_attaches_lossy_warning() {
    // A streaming response that closes without a Usage delta is a
    // silent cost-accounting failure. Aggregator must surface a
    // `LossyEncode { field: "usage", … }` so operators see the
    // miss in observability instead of debugging a suspiciously
    // cheap billing month.
    let mut agg = StreamAggregator::new();
    agg.push(StreamDelta::Start {
        id: "m".into(),
        model: "x".into(),
    })
    .unwrap();
    agg.push(StreamDelta::TextDelta {
        text: "ok".into(),
        provider_echoes: Vec::new(),
    })
    .unwrap();
    agg.push(StreamDelta::Stop {
        stop_reason: StopReason::EndTurn,
    })
    .unwrap();

    let resp = agg.finalize().unwrap();
    assert_eq!(resp.usage, Usage::default());
    let usage_warning = resp
        .warnings
        .iter()
        .find(|w| matches!(w, ModelWarning::LossyEncode { field, .. } if field == "usage"));
    assert!(
        usage_warning.is_some(),
        "expected LossyEncode warning for missing usage, got {:?}",
        resp.warnings
    );
}

#[test]
fn finalize_without_stop_returns_invalid_request() {
    let mut agg = StreamAggregator::new();
    agg.push(StreamDelta::Start {
        id: "m".into(),
        model: "x".into(),
    })
    .unwrap();
    agg.push(StreamDelta::TextDelta {
        text: "ok".into(),
        provider_echoes: Vec::new(),
    })
    .unwrap();
    let err = agg.finalize().unwrap_err();
    assert!(matches!(err, Error::InvalidRequest(_)));
}

#[test]
fn finalize_with_open_tool_block_returns_invalid_request() {
    let mut agg = StreamAggregator::new();
    agg.push(StreamDelta::Start {
        id: "m".into(),
        model: "x".into(),
    })
    .unwrap();
    agg.push(StreamDelta::ToolUseStart {
        id: "t".into(),
        name: "noop".into(),
        provider_echoes: Vec::new(),
    })
    .unwrap();
    agg.push(StreamDelta::Stop {
        stop_reason: StopReason::EndTurn,
    })
    .unwrap();
    let err = agg.finalize().unwrap_err();
    assert!(matches!(err, Error::InvalidRequest(_)));
}

#[test]
fn malformed_tool_input_json_returns_actionable_invalid_request() {
    // The aggregator surfaces malformed tool-use arguments as
    // [`Error::InvalidRequest`] carrying the tool name + id + the
    // wrapped serde-json diagnostic, instead of a bare
    // [`Error::Serde`]. In a multi-tool agent run, knowing *which*
    // tool call's arguments failed to parse is the difference
    // between a 30-second triage and a 30-minute log dive.
    let mut agg = StreamAggregator::new();
    agg.push(StreamDelta::Start {
        id: "m".into(),
        model: "x".into(),
    })
    .unwrap();
    agg.push(StreamDelta::ToolUseStart {
        id: "t".into(),
        name: "noop".into(),
        provider_echoes: Vec::new(),
    })
    .unwrap();
    agg.push(StreamDelta::ToolUseInputDelta {
        partial_json: "not json".into(),
    })
    .unwrap();
    let err = agg.push(StreamDelta::ToolUseStop).unwrap_err();
    match err {
        Error::InvalidRequest(msg) => {
            assert!(msg.contains("noop"), "missing tool name in: {msg}");
            assert!(msg.contains("id=t"), "missing tool id in: {msg}");
            assert!(
                msg.contains("not valid JSON"),
                "missing diagnostic in: {msg}"
            );
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
}

#[test]
fn duplicate_start_returns_invalid_request() {
    let mut agg = StreamAggregator::new();
    agg.push(StreamDelta::Start {
        id: "a".into(),
        model: "x".into(),
    })
    .unwrap();
    let err = agg
        .push(StreamDelta::Start {
            id: "b".into(),
            model: "y".into(),
        })
        .unwrap_err();
    assert!(matches!(err, Error::InvalidRequest(_)));
}

#[test]
fn text_during_tool_block_returns_invalid_request() {
    let mut agg = StreamAggregator::new();
    agg.push(StreamDelta::Start {
        id: "m".into(),
        model: "x".into(),
    })
    .unwrap();
    agg.push(StreamDelta::ToolUseStart {
        id: "t".into(),
        name: "x".into(),
        provider_echoes: Vec::new(),
    })
    .unwrap();
    let err = agg
        .push(StreamDelta::TextDelta {
            text: "oops".into(),
            provider_echoes: Vec::new(),
        })
        .unwrap_err();
    assert!(matches!(err, Error::InvalidRequest(_)));
}

#[test]
fn is_finished_flips_on_stop() {
    let mut agg = StreamAggregator::new();
    assert!(!agg.is_finished());
    agg.push(StreamDelta::Start {
        id: "m".into(),
        model: "x".into(),
    })
    .unwrap();
    assert!(!agg.is_finished());
    agg.push(StreamDelta::Stop {
        stop_reason: StopReason::EndTurn,
    })
    .unwrap();
    assert!(agg.is_finished());
}

#[test]
fn duplicate_stop_delta_with_different_reason_is_rejected() {
    // A misbehaving provider sending two terminal `Stop` events
    // would silently change the observable termination cause —
    // e.g. a model first reports `EndTurn` then re-emits as
    // `MaxTokens`. The aggregator must fail closed so the
    // semantic flip is detected by the caller, not silently
    // baked into the final ModelResponse.
    let mut agg = StreamAggregator::new();
    agg.push(StreamDelta::Start {
        id: "msg_01".into(),
        model: "claude-opus-4-7".into(),
    })
    .unwrap();
    agg.push(StreamDelta::TextDelta {
        text: "ok".into(),
        provider_echoes: Vec::new(),
    })
    .unwrap();
    agg.push(StreamDelta::Stop {
        stop_reason: StopReason::EndTurn,
    })
    .unwrap();
    let err = agg
        .push(StreamDelta::Stop {
            stop_reason: StopReason::MaxTokens,
        })
        .expect_err("second Stop must be rejected");
    match err {
        Error::InvalidRequest(msg) => {
            assert!(
                msg.contains("duplicate Stop") || msg.contains("terminal state"),
                "expected duplicate-stop diagnostic, got: {msg}"
            );
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
}
