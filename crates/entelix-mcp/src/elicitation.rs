//! `ElicitationProvider` + request/response shapes — client-side
//! answer to the server-initiated `elicitation/create` request
//! (MCP 2025-03 §"Elicitation").
//!
//! Elicitation lets an MCP server ask the client (the agent
//! harness) for typed input mid-session — typically a missing
//! piece of configuration or a confirmation the server needs
//! before continuing. The server names the shape it expects
//! (`requested_schema`) and a human-readable prompt (`message`);
//! the client consults a wired [`ElicitationProvider`] and
//! returns either accepted content, an explicit decline, or a
//! cancel.
//!
//! ## Why a trait, not a closure
//!
//! Real elicitation handlers reach across the agent's execution
//! environment: a CLI prompt, a UI form, a stored answer
//! cache, a policy that auto-declines unknown servers. Each of
//! those wants its own state — a closure surface would force
//! every handler into `Box<dyn Fn(...)>` plumbing the operator
//! has to `Arc<Mutex>`-wrap by hand. The trait shape stays
//! clean: implementors hold their own state, the manager calls
//! `elicit(...)` per server-initiated request.
//!
//! ## No `ExecutionContext` parameter
//!
//! Mirrors [`crate::RootsProvider`]: server-initiated requests
//! arrive on a background SSE listener, not in the middle of a
//! client-driven call. Threading an `ExecutionContext` through
//! the listener would force the listener to invent one (which
//! request's context?) — that choice has no honest answer. The
//! signature stays context-free.
//!
//! ## Action enum, not `Result<Option<Value>>`
//!
//! The MCP spec distinguishes three outcomes — accept, decline,
//! cancel. `Result<Option<Value>>` would collapse decline + cancel
//! into `Ok(None)`, losing operator-meaningful information.
//! [`ElicitationResponse`] keeps the three cases explicit and
//! serialises to the wire shape the spec defines.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::McpResult;

/// Elicitation request as it arrives from the server. Both
/// fields ride straight from the JSON-RPC `params` block.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct ElicitationRequest {
    /// Human-readable prompt the server wants the client to
    /// surface (CLI prompt text, UI label, log line).
    pub message: String,
    /// JSON Schema describing the shape of the expected
    /// response. Operators that auto-respond should validate
    /// the candidate value against this schema before returning
    /// `Accept`; the SDK does not enforce shape conformance for
    /// the trait surface so providers stay free to do the
    /// validation in whichever JSON-Schema engine they prefer.
    #[serde(rename = "requestedSchema")]
    pub requested_schema: Value,
}

/// Operator's answer to an elicitation. Three explicit cases
/// — the spec distinguishes them, callers should too.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ElicitationResponse {
    /// Operator approved and supplied content matching
    /// `requested_schema`. `content` rides back as the
    /// `result.content` field on the wire.
    Accept(Value),
    /// Operator explicitly refused (e.g. policy prohibits
    /// auto-answering this prompt). Server SHOULD respect the
    /// refusal and not retry.
    Decline,
    /// Operator dismissed the prompt without answering. Server
    /// MAY retry under different conditions.
    Cancel,
}

impl Serialize for ElicitationResponse {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;
        match self {
            Self::Accept(content) => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("action", "accept")?;
                map.serialize_entry("content", content)?;
                map.end()
            }
            Self::Decline => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("action", "decline")?;
                map.end()
            }
            Self::Cancel => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("action", "cancel")?;
                map.end()
            }
        }
    }
}

/// Async source-of-truth for elicitation answers. Mirrors the
/// `*Provider` taxonomy — async, single-purpose,
/// replaceable.
///
/// Operators wire one provider per server through
/// [`crate::McpServerConfig::with_elicitation_provider`].
/// Servers that don't request elicitation get the provider for
/// free as dormant state (no calls into it).
#[async_trait]
pub trait ElicitationProvider: Send + Sync + 'static + std::fmt::Debug {
    /// Resolve one server-initiated elicitation. Implementors
    /// surface the request to the operator — interactively (CLI
    /// / UI), via a stored cache (looked up by message /
    /// schema), or by an auto-decline policy — and return the
    /// chosen action.
    async fn elicit(&self, request: ElicitationRequest) -> McpResult<ElicitationResponse>;
}

/// In-memory [`ElicitationProvider`] returning a fixed response.
///
/// Useful for tests and for deployments that want to
/// deterministically auto-decline (or auto-accept a single
/// known prompt).
#[derive(Clone, Debug)]
pub struct StaticElicitationProvider {
    response: ElicitationResponse,
}

impl StaticElicitationProvider {
    /// Always-accept variant returning the supplied `content`.
    /// Use when the deployment knows the answer at boot —
    /// typically for auto-confirming a prompt the server is
    /// known to issue.
    #[must_use]
    pub fn accept(content: Value) -> Self {
        Self {
            response: ElicitationResponse::Accept(content),
        }
    }

    /// Always-decline variant. Use when the deployment policy
    /// is "the server may not interrupt for input".
    #[must_use]
    pub fn decline() -> Self {
        Self {
            response: ElicitationResponse::Decline,
        }
    }

    /// Always-cancel variant. Use when the deployment policy
    /// is "the server may try again later".
    #[must_use]
    pub fn cancel() -> Self {
        Self {
            response: ElicitationResponse::Cancel,
        }
    }
}

#[async_trait]
impl ElicitationProvider for StaticElicitationProvider {
    async fn elicit(&self, _request: ElicitationRequest) -> McpResult<ElicitationResponse> {
        Ok(self.response.clone())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn accept_serializes_with_content() {
        let r = ElicitationResponse::Accept(json!({"api_key": "abc"}));
        let s = serde_json::to_value(&r).unwrap();
        assert_eq!(
            s,
            json!({"action": "accept", "content": {"api_key": "abc"}})
        );
    }

    #[test]
    fn decline_serializes_action_only() {
        let r = ElicitationResponse::Decline;
        let s = serde_json::to_value(&r).unwrap();
        assert_eq!(s, json!({"action": "decline"}));
    }

    #[test]
    fn cancel_serializes_action_only() {
        let r = ElicitationResponse::Cancel;
        let s = serde_json::to_value(&r).unwrap();
        assert_eq!(s, json!({"action": "cancel"}));
    }

    #[test]
    fn request_deserializes_from_wire_shape() {
        let raw = json!({
            "message": "Please provide your API key",
            "requestedSchema": {
                "type": "object",
                "properties": {"api_key": {"type": "string"}}
            }
        });
        let parsed: ElicitationRequest = serde_json::from_value(raw).unwrap();
        assert_eq!(parsed.message, "Please provide your API key");
        assert_eq!(parsed.requested_schema["type"], "object");
    }

    #[tokio::test]
    async fn static_accept_returns_supplied_content() {
        let provider = StaticElicitationProvider::accept(json!({"v": 1}));
        let resp = provider
            .elicit(ElicitationRequest {
                message: "?".into(),
                requested_schema: json!({}),
            })
            .await
            .unwrap();
        assert_eq!(resp, ElicitationResponse::Accept(json!({"v": 1})));
    }

    #[tokio::test]
    async fn static_decline_returns_decline() {
        let provider = StaticElicitationProvider::decline();
        let resp = provider
            .elicit(ElicitationRequest {
                message: "?".into(),
                requested_schema: json!({}),
            })
            .await
            .unwrap();
        assert_eq!(resp, ElicitationResponse::Decline);
    }
}
