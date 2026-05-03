//! `PromptTemplate` parse + render tests.

#![allow(
    clippy::unwrap_used,
    clippy::needless_borrows_for_generic_args,
    clippy::approx_constant
)]

use std::collections::HashMap;

use entelix_core::{Error, ExecutionContext, Result};
use entelix_prompt::PromptTemplate;
use entelix_runnable::Runnable;

fn vars(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect()
}

#[test]
fn render_substitutes_single_variable() -> Result<()> {
    let t = PromptTemplate::new("Hello, {{ name }}!")?;
    assert_eq!(t.render(&vars(&[("name", "world")]))?, "Hello, world!");
    Ok(())
}

#[test]
fn render_substitutes_multiple_variables_in_order() -> Result<()> {
    let t = PromptTemplate::new("{{ greet }}, {{ name }}! You are {{ role }}.")?;
    let out = t.render(&vars(&[
        ("greet", "Hi"),
        ("name", "Ada"),
        ("role", "admin"),
    ]))?;
    assert_eq!(out, "Hi, Ada! You are admin.");
    Ok(())
}

#[test]
fn render_handles_literal_braces_outside_jinja_delimiters() -> Result<()> {
    // Single braces are literal text — they are not template syntax,
    // so they round-trip without any escape ceremony.
    let t = PromptTemplate::new("Use { and } for braces; var={{ x }}")?;
    assert_eq!(
        t.render(&vars(&[("x", "1")]))?,
        "Use { and } for braces; var=1"
    );
    Ok(())
}

#[test]
fn variables_returns_sorted_unique_names() -> Result<()> {
    let t = PromptTemplate::new("{{ b }} and {{ a }} and {{ b }} again")?;
    assert_eq!(t.variables(), &["a".to_owned(), "b".to_owned()]);
    Ok(())
}

#[test]
fn missing_variable_returns_invalid_request() -> Result<()> {
    let t = PromptTemplate::new("Hello, {{ name }}!")?;
    let err = t.render(&vars(&[])).unwrap_err();
    assert!(matches!(err, Error::InvalidRequest(_)));
    Ok(())
}

#[test]
fn unterminated_template_block_rejected_at_parse() {
    assert!(matches!(
        PromptTemplate::new("hello {{ name").unwrap_err(),
        Error::InvalidRequest(_)
    ));
    assert!(matches!(
        PromptTemplate::new("{% if cond %}body").unwrap_err(),
        Error::InvalidRequest(_)
    ));
}

#[test]
fn jinja_conditionals_render_correctly() -> Result<()> {
    let t =
        PromptTemplate::new("{% if role == \"admin\" %}admin view{% else %}user view{% endif %}")?;
    assert_eq!(t.render(&vars(&[("role", "admin")]))?, "admin view");
    assert_eq!(t.render(&vars(&[("role", "guest")]))?, "user view");
    Ok(())
}

#[test]
fn jinja_filters_compose() -> Result<()> {
    let t = PromptTemplate::new("{{ name | upper }}!")?;
    assert_eq!(t.render(&vars(&[("name", "ada")]))?, "ADA!");
    Ok(())
}

#[tokio::test]
async fn implements_runnable() -> Result<()> {
    let t = PromptTemplate::new("Hello, {{ name }}!")?;
    let ctx = ExecutionContext::new();
    let out = t.invoke(vars(&[("name", "Runnable")]), &ctx).await?;
    assert_eq!(out, "Hello, Runnable!");
    Ok(())
}

// ── serde_json::Value Runnable impl ────────────────────────────────────────

#[test]
fn render_with_value_supports_dotted_attribute_access() -> Result<()> {
    let t = PromptTemplate::new("Hello, {{ user.name }}! Email: {{ user.email }}")?;
    let vars = serde_json::json!({
        "user": {"name": "Ada", "email": "ada@example.com"}
    });
    assert_eq!(t.render(&vars)?, "Hello, Ada! Email: ada@example.com");
    Ok(())
}

#[test]
fn render_with_value_supports_array_iteration() -> Result<()> {
    let t = PromptTemplate::new("{% for item in items %}- {{ item }}\n{% endfor %}")?;
    let vars = serde_json::json!({"items": ["alpha", "beta", "gamma"]});
    assert_eq!(t.render(&vars)?, "- alpha\n- beta\n- gamma\n");
    Ok(())
}

#[test]
fn render_with_value_handles_numeric_types() -> Result<()> {
    let t = PromptTemplate::new("count={{ n }} pi={{ pi }}")?;
    let vars = serde_json::json!({"n": 42, "pi": 3.14});
    assert_eq!(t.render(&vars)?, "count=42 pi=3.14");
    Ok(())
}

#[tokio::test]
async fn runnable_value_invoke_renders_nested() -> Result<()> {
    let t = PromptTemplate::new("Hi {{ user.name }}")?;
    let ctx = ExecutionContext::new();
    let input = serde_json::json!({"user": {"name": "Ada"}});
    let out: String = t.invoke(input, &ctx).await?;
    assert_eq!(out, "Hi Ada");
    Ok(())
}

#[tokio::test]
async fn runnable_value_missing_var_returns_invalid_request() -> Result<()> {
    let t = PromptTemplate::new("Hi {{ name }}")?;
    let ctx = ExecutionContext::new();
    let err = t.invoke(serde_json::json!({}), &ctx).await.unwrap_err();
    assert!(matches!(err, Error::InvalidRequest(_)));
    Ok(())
}
