//! `ListSkillsTool` / `ActivateSkillTool` / `ReadSkillResourceTool`
//! behaviour tests — the three LLM-facing tools that drive
//! progressive disclosure.

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use std::sync::Arc;

use entelix_core::AgentContext;
use entelix_core::skills::SkillRegistry;
use entelix_core::tools::Tool;
use entelix_tools::{
    ActivateSkillTool, InMemorySkill, ListSkillsTool, ReadSkillResourceTool, StaticResource,
};
use serde_json::{Value, json};

fn registry() -> SkillRegistry {
    let echo = InMemorySkill::builder("echo")
        .with_description("Repeat the user's last message verbatim.")
        .with_version("1.0.0")
        .with_instructions("Long instructions for the echo skill.")
        .with_text_resource("examples/basic.md", "U: hi\nA: hi")
        .with_resource(
            "icon.png",
            Arc::new(StaticResource::binary(
                "image/png",
                vec![0x89, 0x50, 0x4e, 0x47],
            )),
        )
        .build()
        .unwrap();
    let sum = InMemorySkill::builder("sum")
        .with_description("Sum a list of integers.")
        .with_instructions("Long instructions for the sum skill.")
        .build()
        .unwrap();
    SkillRegistry::new()
        .register(Arc::new(echo))
        .unwrap()
        .register(Arc::new(sum))
        .unwrap()
}

#[tokio::test]
async fn list_skills_returns_metadata_only() {
    let tool = ListSkillsTool::new(registry());
    let out = tool
        .execute(json!({}), &AgentContext::default())
        .await
        .unwrap();
    let arr = out.get("skills").and_then(Value::as_array).unwrap();
    assert_eq!(arr.len(), 2);
    // Stable sort by name.
    assert_eq!(arr[0]["name"], "echo");
    assert_eq!(
        arr[0]["description"],
        "Repeat the user's last message verbatim."
    );
    assert_eq!(arr[0]["version"], "1.0.0");
    assert_eq!(arr[1]["name"], "sum");
    // Skills with no version omit the field.
    assert!(arr[1].get("version").is_none());
    // Listing must NOT contain instructions or resource bodies.
    let serialized = serde_json::to_string(&out).unwrap();
    assert!(!serialized.contains("Long instructions"));
}

#[tokio::test]
async fn activate_skill_returns_instructions_and_resource_keys() {
    let tool = ActivateSkillTool::new(registry());
    let out = tool
        .execute(json!({"name": "echo"}), &AgentContext::default())
        .await
        .unwrap();
    assert!(
        out.get("instructions")
            .and_then(Value::as_str)
            .unwrap()
            .contains("Long instructions for the echo")
    );
    let keys = out.get("resources").and_then(Value::as_array).unwrap();
    let mut keys_str: Vec<&str> = keys.iter().filter_map(Value::as_str).collect();
    keys_str.sort_unstable();
    assert_eq!(keys_str, vec!["examples/basic.md", "icon.png"]);
}

#[tokio::test]
async fn activate_unknown_skill_returns_config_error() {
    let tool = ActivateSkillTool::new(registry());
    let err = tool
        .execute(json!({"name": "ghost"}), &AgentContext::default())
        .await
        .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("ghost"), "unexpected error: {msg}");
}

#[tokio::test]
async fn read_skill_resource_text_returns_full_text() {
    let tool = ReadSkillResourceTool::new(registry());
    let out = tool
        .execute(
            json!({"skill": "echo", "key": "examples/basic.md"}),
            &AgentContext::default(),
        )
        .await
        .unwrap();
    assert_eq!(
        out.get("text").and_then(Value::as_str).unwrap(),
        "U: hi\nA: hi"
    );
}

#[tokio::test]
async fn read_skill_resource_binary_returns_metadata_only() {
    let tool = ReadSkillResourceTool::new(registry());
    let out = tool
        .execute(
            json!({"skill": "echo", "key": "icon.png"}),
            &AgentContext::default(),
        )
        .await
        .unwrap();
    // No `text` field — that would defeat progressive disclosure.
    assert!(out.get("text").is_none());
    assert_eq!(out["mime_type"], "image/png");
    assert_eq!(out["size_bytes"], 4);
    let sha = out["sha256"].as_str().unwrap();
    assert_eq!(sha.len(), 64);
    assert!(
        sha.chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
    );
}

#[tokio::test]
async fn read_skill_resource_missing_key_errors() {
    let tool = ReadSkillResourceTool::new(registry());
    let err = tool
        .execute(
            json!({"skill": "echo", "key": "missing"}),
            &AgentContext::default(),
        )
        .await
        .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("missing"), "unexpected error: {msg}");
}
