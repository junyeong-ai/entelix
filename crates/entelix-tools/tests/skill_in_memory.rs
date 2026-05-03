//! `InMemorySkill` round-trip tests.

#![allow(clippy::unwrap_used)]

use entelix_core::context::ExecutionContext;
use entelix_core::skills::{Skill, SkillResourceContent};
use entelix_tools::InMemorySkill;

#[tokio::test]
async fn in_memory_skill_load_returns_instructions_and_resource_keys() {
    let skill = InMemorySkill::builder("echo")
        .with_description("Echo skill — repeats the user's last message")
        .with_version("1.0.0")
        .with_instructions("When asked to echo, repeat verbatim.")
        .with_text_resource("examples/basic.md", "User: hi\nAssistant: hi")
        .with_text_resource("reference/api.md", "echo(text) -> text")
        .build()
            .unwrap();

    assert_eq!(skill.name(), "echo");
    assert_eq!(
        skill.description(),
        "Echo skill — repeats the user's last message"
    );
    assert_eq!(skill.version(), Some("1.0.0"));

    let loaded = skill.load(&ExecutionContext::new()).await.unwrap();
    assert!(loaded.instructions.contains("repeat verbatim"));
    let mut keys = loaded.resource_keys();
    keys.sort_unstable();
    assert_eq!(keys, vec!["examples/basic.md", "reference/api.md"]);
}

#[tokio::test]
async fn in_memory_skill_resources_resolve_to_text() {
    let skill = InMemorySkill::builder("greet")
        .with_description("greeter")
        .with_instructions("Greet warmly.")
        .with_text_resource("hello.txt", "Hello, world!")
        .build()
            .unwrap();
    let loaded = skill.load(&ExecutionContext::new()).await.unwrap();
    let res = loaded.resources.get("hello.txt").unwrap();
    let content = res.read(&ExecutionContext::new()).await.unwrap();
    assert!(matches!(content, SkillResourceContent::Text(ref t) if t == "Hello, world!"));
}

#[tokio::test]
async fn in_memory_skill_with_no_resources_returns_empty_map() {
    let skill = InMemorySkill::builder("plain")
        .with_description("plain")
        .with_instructions("Just instructions, no resources.")
        .build()
            .unwrap();
    let loaded = skill.load(&ExecutionContext::new()).await.unwrap();
    assert!(loaded.resources.is_empty());
    assert!(loaded.resource_keys().is_empty());
}

#[tokio::test]
async fn binary_resource_round_trips_via_static_resource() {
    use std::sync::Arc;

    use entelix_core::skills::SkillResource;
    use entelix_tools::StaticResource;

    let r = Arc::new(StaticResource::binary(
        "image/png",
        b"\x89PNG\r\n\x1a\n".to_vec(),
    )) as Arc<dyn SkillResource>;
    let content = r.read(&ExecutionContext::new()).await.unwrap();
    match content {
        SkillResourceContent::Binary { mime_type, bytes } => {
            assert_eq!(mime_type, "image/png");
            assert_eq!(bytes.len(), 8);
        }
        other => panic!("expected binary, got {other:?}"),
    }
}
