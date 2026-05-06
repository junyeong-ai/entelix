//! Forcing function for ADR-0027's progressive-disclosure budget.
//!
//! For one skill with bulky T2 instructions and a bulky T3 resource,
//! assert each tier's LLM-facing payload size grows only when the
//! corresponding tier is invoked:
//!
//! - **T1** (`list_skills`) returns `name + description + version` only —
//!   the body of `instructions` and any resource payload must NOT
//!   appear in the output bytes.
//! - **T2** (`activate_skill`) returns `instructions` and resource
//!   *keys*. Resource bodies must NOT appear.
//! - **T3** (`read_skill_resource`) is the only path that surfaces a
//!   resource's text body. Binary resources surface as metadata only,
//!   never as a base64 payload.

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use std::sync::Arc;

use entelix_core::AgentContext;
use entelix_core::skills::SkillRegistry;
use entelix_core::tools::Tool;
use entelix_tools::{
    ActivateSkillTool, InMemorySkill, ListSkillsTool, ReadSkillResourceTool, StaticResource,
};
use serde_json::json;

const BULKY_INSTRUCTIONS: &str =
    "INSTRUCTIONS_BODY_MARKER_LONG_LONG_LONG_LONG_LONG_LONG_LONG_LONG_LONG";
const BULKY_TEXT_RESOURCE: &str =
    "RESOURCE_BODY_MARKER_LONG_LONG_LONG_LONG_LONG_LONG_LONG_LONG_LONG_LONG";

fn registry() -> SkillRegistry {
    let bulky = InMemorySkill::builder("bulky")
        .with_description("A skill with bulky instructions and resources.")
        .with_version("1.0.0")
        .with_instructions(BULKY_INSTRUCTIONS)
        .with_text_resource("notes.md", BULKY_TEXT_RESOURCE)
        .with_resource(
            "asset.png",
            Arc::new(StaticResource::binary(
                "image/png",
                vec![0xff; 4096], // 4 KB
            )),
        )
        .build()
        .unwrap();
    SkillRegistry::new().register(Arc::new(bulky)).unwrap()
}

#[tokio::test]
async fn t1_listing_excludes_instructions_and_resource_bodies() {
    let tool = ListSkillsTool::new(registry());
    let out = tool
        .execute(json!({}), &AgentContext::default())
        .await
        .unwrap();
    let serialized = serde_json::to_string(&out).unwrap();
    assert!(
        !serialized.contains(BULKY_INSTRUCTIONS),
        "T1 listing must not embed the instructions body"
    );
    assert!(
        !serialized.contains(BULKY_TEXT_RESOURCE),
        "T1 listing must not embed any resource body"
    );
    // 4 KB binary asset must not leak as base64 either.
    assert!(
        !serialized.contains("/////////"),
        "T1 listing must not embed binary bytes (in any encoding)"
    );
}

#[tokio::test]
async fn t2_activation_includes_instructions_excludes_resource_bodies() {
    let tool = ActivateSkillTool::new(registry());
    let out = tool
        .execute(json!({"name": "bulky"}), &AgentContext::default())
        .await
        .unwrap();
    let serialized = serde_json::to_string(&out).unwrap();
    // T2 carries instructions.
    assert!(
        serialized.contains(BULKY_INSTRUCTIONS),
        "T2 activation must include the instructions body"
    );
    // T2 must NOT carry resource bodies — keys only.
    assert!(
        !serialized.contains(BULKY_TEXT_RESOURCE),
        "T2 activation must list resource keys only, not bodies"
    );
}

#[tokio::test]
async fn t3_text_read_returns_full_body() {
    let tool = ReadSkillResourceTool::new(registry());
    let out = tool
        .execute(
            json!({"skill": "bulky", "key": "notes.md"}),
            &AgentContext::default(),
        )
        .await
        .unwrap();
    assert_eq!(
        out.get("text").and_then(serde_json::Value::as_str).unwrap(),
        BULKY_TEXT_RESOURCE
    );
}

#[tokio::test]
async fn t3_binary_read_returns_metadata_only_never_bytes() {
    let tool = ReadSkillResourceTool::new(registry());
    let out = tool
        .execute(
            json!({"skill": "bulky", "key": "asset.png"}),
            &AgentContext::default(),
        )
        .await
        .unwrap();
    let serialized = serde_json::to_string(&out).unwrap();
    // The 4 KB of `0xff` would be ~5.5 KB of base64 starting with `/`
    // characters — assert that pattern never appears anywhere in the
    // tool result.
    assert!(
        !serialized.contains("/////////"),
        "binary T3 read must surface metadata only — bytes leaked: {serialized}"
    );
    assert_eq!(out["mime_type"], "image/png");
    assert_eq!(out["size_bytes"], 4096);
    assert!(out["sha256"].as_str().unwrap().len() == 64);
}
