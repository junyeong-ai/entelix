//! `FewShotPromptTemplate` + `ExampleSelector` tests.

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use std::collections::HashMap;

use entelix_core::ExecutionContext;
use entelix_prompt::{
    ChatFewShotPromptTemplate, ChatPromptPart, ChatPromptTemplate, Example, ExampleSelector,
    FewShotPromptTemplate, FixedExampleSelector, LengthBasedExampleSelector, PromptTemplate,
    PromptValue, PromptVars,
};
use entelix_runnable::Runnable;

fn ex(q: &str, a: &str) -> Example {
    HashMap::from([
        ("q".to_owned(), q.to_owned()),
        ("a".to_owned(), a.to_owned()),
    ])
}

fn input_vars(input: &str) -> HashMap<String, String> {
    HashMap::from([("input".to_owned(), input.to_owned())])
}

#[tokio::test]
async fn fewshot_renders_prefix_examples_and_suffix() -> entelix_core::Result<()> {
    let example_prompt = PromptTemplate::new("Q: {{ q }}\nA: {{ a }}")?;
    let suffix_prompt = PromptTemplate::new("Q: {{ input }}\nA:")?;
    let prompt = FewShotPromptTemplate::with_examples(
        vec![ex("2+2?", "4"), ex("3+3?", "6")],
        example_prompt,
        suffix_prompt,
    )
    .with_prefix("Solve the problems below.")
    .with_example_separator("\n---\n");

    let out = prompt
        .invoke(input_vars("4+4?"), &ExecutionContext::new())
        .await?;
    let expected =
        "Solve the problems below.\n---\nQ: 2+2?\nA: 4\n---\nQ: 3+3?\nA: 6\n---\nQ: 4+4?\nA:";
    assert_eq!(out, expected);
    Ok(())
}

#[tokio::test]
async fn fewshot_zero_examples_renders_only_prefix_and_suffix() -> entelix_core::Result<()> {
    let example_prompt = PromptTemplate::new("Q: {{ q }} A: {{ a }}")?;
    let suffix_prompt = PromptTemplate::new("Q: {{ input }}")?;
    let prompt = FewShotPromptTemplate::with_examples(vec![], example_prompt, suffix_prompt)
        .with_prefix("INSTR");
    let out = prompt
        .invoke(input_vars("hi"), &ExecutionContext::new())
        .await?;
    assert_eq!(out, "INSTR\n\nQ: hi");
    Ok(())
}

#[test]
fn fixed_selector_returns_examples_verbatim() -> entelix_core::Result<()> {
    let selector = FixedExampleSelector::new(vec![ex("a", "b")]);
    let chosen = selector.select(&input_vars("ignored"))?;
    assert_eq!(chosen.len(), 1);
    assert_eq!(chosen[0].get("q").map(String::as_str), Some("a"));
    Ok(())
}

#[test]
fn length_based_selector_drops_overflow_examples() -> entelix_core::Result<()> {
    let example_prompt = PromptTemplate::new("Q: {{ q }} A: {{ a }}")?;
    // Each "Q: X A: Y" rendering is 9 chars; cap at 20 keeps 2 examples.
    let selector = LengthBasedExampleSelector::new(
        vec![ex("X", "Y"), ex("X", "Y"), ex("X", "Y")],
        example_prompt,
        20,
    );
    let chosen = selector.select(&HashMap::new())?;
    assert_eq!(chosen.len(), 2);
    Ok(())
}

#[test]
fn length_based_selector_zero_cap_returns_empty() -> entelix_core::Result<()> {
    let example_prompt = PromptTemplate::new("Q: {{ q }}")?;
    let selector = LengthBasedExampleSelector::new(vec![ex("X", "Y")], example_prompt, 0);
    let chosen = selector.select(&HashMap::new())?;
    assert!(chosen.is_empty());
    Ok(())
}

#[tokio::test]
async fn chat_fewshot_emits_user_assistant_pairs_then_suffix() -> entelix_core::Result<()> {
    let example_chat = ChatPromptTemplate::from_messages([
        ChatPromptPart::user("{{ q }}")?,
        ChatPromptPart::assistant("{{ a }}")?,
    ]);
    let suffix_chat = ChatPromptTemplate::from_messages([
        ChatPromptPart::system("Be concise.")?,
        ChatPromptPart::user("{{ input }}")?,
    ]);
    let prompt = ChatFewShotPromptTemplate::with_examples(
        vec![ex("hi?", "hello"), ex("bye?", "ciao")],
        example_chat,
        suffix_chat,
    );

    let mut vars: PromptVars = HashMap::new();
    vars.insert("input".to_owned(), PromptValue::Text("hola?".into()));
    let messages = prompt.invoke(vars, &ExecutionContext::new()).await?;
    // 2 examples × 2 messages = 4, plus suffix system + user = 6.
    assert_eq!(messages.len(), 6);
    Ok(())
}
