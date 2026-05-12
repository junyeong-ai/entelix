//! `AgentEvent<S>` — runtime events the agent emits during a turn.
//!
//! ## Two surfaces, one type
//!
//! - **LLM-facing**: state inside `Complete { state }` round-trips
//!   into the next graph turn when an agent is composed inside a
//!   larger graph. Sinks render the same value for observability.
//! - **Observability-facing**: every variant carries a `run_id` for
//!   correlation; `OTel` sinks stamp it onto `entelix.run_id` span
//!   attributes without the agent itself reading the field.
//!
//! ## Lifecycle contract
//!
//! Every run emits `Started{run_id}` and exactly one of
//! `Complete{run_id, ...}` or `Failed{run_id, ...}` with the same
//! `run_id`. Tool variants (`ToolStart` / `ToolComplete` /
//! `ToolError`) are interleaved between the book-ends as the
//! agent's inner graph dispatches tools.
//!
//! `#[non_exhaustive]` keeps adding variants forward-compatible —
//! consumer `match` arms always need a fallback.
//!
//! ## Relationship to [`entelix_session::GraphEvent`]
//!
//! `AgentEvent<S>` is the **runtime-side superset** of the durable
//! audit log entry [`entelix_session::GraphEvent`]:
//!
//! - **Runtime-only variants** — `Started`, `Complete`, `Failed`,
//!   plus the `tool_version` / `duration_ms` metric fields on the
//!   tool variants — exist for telemetry and per-run correlation.
//!   They have no audit projection.
//! - **Audit-projecting variants** — `ToolStart` / `ToolComplete` /
//!   `ToolError` — map onto `GraphEvent::ToolCall` /
//!   `GraphEvent::ToolResult` via [`AgentEvent::to_graph_event`].
//!
//! The projection is the single source of truth: an operator
//! wiring both an `AgentEventSink` (for telemetry) and a
//! `SessionGraph` (for durable audit) routes tool emissions
//! through this method rather than constructing `GraphEvent`
//! independently — the two channels record the same fact through
//! one construction path.

use chrono::{DateTime, Utc};
use serde_json::Value;

use entelix_core::ErrorEnvelope;
use entelix_core::RenderedForLlm;
use entelix_core::TenantId;
use entelix_core::UsageSnapshot;
use entelix_core::ir::ToolResultContent;
use entelix_session::GraphEvent;

/// Runtime events emitted by the agent during a single
/// `execute` / `execute_stream` call.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum AgentEvent<S> {
    /// Run opened. Sinks use this to mark span beginnings, allocate
    /// per-run state, and emit "session opened" telemetry.
    Started {
        /// Per-run correlation id (UUID v7). Stable for the
        /// duration of the run; matches the id on every subsequent
        /// event for this same call.
        run_id: String,
        /// Tenant scope this event belongs to (invariant 11 —
        /// every emit site stamps `ctx.tenant_id().clone()`). Audit /
        /// billing / replay consumers key off this field directly
        /// instead of correlating through a separate `run_id` →
        /// `tenant_id` lookup.
        tenant_id: TenantId,
        /// Run id of the calling agent when this run was dispatched
        /// from a parent (sub-agent fan-out, supervisor handoff).
        /// `None` for top-level runs. LangSmith-style trace-tree
        /// consumers reconstruct the hierarchy from
        /// `(run_id, parent_run_id)` edges across these events.
        parent_run_id: Option<String>,
        /// Agent identifier configured on `AgentBuilder::name(...)`.
        agent: String,
    },

    /// One tool dispatch began. Emitted by
    /// [`crate::agent::tool_event_layer::ToolEventLayer`] when wired
    /// into the tool registry. Absent when the layer is not wired
    /// (the agent runtime itself does not generate tool events).
    ToolStart {
        /// Run correlation id.
        run_id: String,
        /// Tenant scope this event belongs to (see `Started`).
        tenant_id: TenantId,
        /// Stable tool-use id matching the originating
        /// `ContentPart::ToolUse`.
        tool_use_id: String,
        /// Tool name being dispatched.
        tool: String,
        /// Tool version (`Tool::version()`) when the tool advertises
        /// one — useful for distinguishing behaviour changes between
        /// otherwise-identically-named tool revisions.
        tool_version: Option<String>,
        /// Tool input (already JSON-validated by the tool's schema).
        input: Value,
    },

    /// One tool dispatch finished successfully.
    ToolComplete {
        /// Run correlation id.
        run_id: String,
        /// Tenant scope this event belongs to (see `Started`).
        tenant_id: TenantId,
        /// Stable tool-use id matching the corresponding `ToolStart`.
        tool_use_id: String,
        /// Tool name (echoed for sink convenience).
        tool: String,
        /// Tool version echoed from the matching `ToolStart` so sinks
        /// can correlate completion telemetry without retaining
        /// per-`tool_use_id` state.
        tool_version: Option<String>,
        /// Wall-clock duration measured by the layer.
        duration_ms: u64,
        /// JSON output the tool produced. Sinks that persist tool
        /// audit logs read this directly; PII redaction happens at
        /// the policy layer before this event is emitted, so the
        /// payload is safe for storage.
        output: Value,
    },

    /// One tool dispatch failed.
    ToolError {
        /// Run correlation id.
        run_id: String,
        /// Tenant scope this event belongs to (see `Started`).
        tenant_id: TenantId,
        /// Stable tool-use id matching the corresponding `ToolStart`.
        tool_use_id: String,
        /// Tool name (echoed for sink convenience).
        tool: String,
        /// Tool version echoed from the matching `ToolStart` so sinks
        /// see the same provenance on the failure path as on success.
        tool_version: Option<String>,
        /// Operator-facing error message (`Display` form, includes
        /// vendor status, source chain). Sinks, OTel, and log
        /// destinations consume this.
        error: String,
        /// LLM-facing error message wrapped in a sealed
        /// [`RenderedForLlm`] carrier. The carrier's constructor is
        /// `pub(crate)` to `entelix-core`, so the only path from a
        /// raw `String` to this field is
        /// [`entelix_core::LlmRenderable::for_llm`] — emit sites
        /// cannot fabricate model-facing content. The audit-log
        /// projection ([`Self::to_graph_event`]) extracts the inner
        /// rendering into `GraphEvent::ToolResult` so replay
        /// reconstructs the model's view without re-leaking
        /// operator content (invariant #16).
        error_for_llm: RenderedForLlm<String>,
        /// Typed wire shape produced by
        /// [`entelix_core::Error::envelope`]. Bundles `wire_code`
        /// (i18n key / metric label), `wire_class` (responsibility
        /// split), `retry_after_secs` (vendor `Retry-After` hint),
        /// and `provider_status` (raw HTTP status) so sinks, audit
        /// replay, SSE adapters, and FE rate-limit timers all read
        /// one `Copy` value instead of pattern-matching the inner
        /// error variant. Patch-version-stable.
        envelope: ErrorEnvelope,
        /// Wall-clock duration measured by the layer.
        duration_ms: u64,
    },

    /// Run terminated with the inner runnable's error. The matching
    /// `Started{run_id}` is always present in the same stream.
    /// Caller-facing streams additionally surface the typed error
    /// via `Result::Err`; sinks see only this event.
    Failed {
        /// Run correlation id.
        run_id: String,
        /// Tenant scope this event belongs to (see `Started`).
        tenant_id: TenantId,
        /// Lean error message (`Display` form).
        error: String,
        /// Typed wire shape produced by
        /// [`entelix_core::Error::envelope`] — see `ToolError` for
        /// the field roster. Replay / audit / metric / SSE consumers
        /// route off this field instead of parsing `error` prose.
        envelope: ErrorEnvelope,
    },

    /// Run terminated successfully with the agent's terminal state.
    Complete {
        /// Run correlation id.
        run_id: String,
        /// Tenant scope this event belongs to (see `Started`).
        tenant_id: TenantId,
        /// Final state returned by the inner runnable.
        state: S,
        /// Frozen [`UsageSnapshot`] of the [`entelix_core::RunBudget`]
        /// counters at the moment the inner runnable returned.
        /// `None` when no budget was attached to the
        /// [`entelix_core::ExecutionContext`]. Mirrors the
        /// `usage` field on
        /// [`crate::AgentRunResult`] so streaming and one-shot
        /// surfaces observe the same terminal artifact.
        usage: Option<UsageSnapshot>,
    },

    /// HITL approver decided to permit one tool dispatch. Emitted by
    /// [`crate::agent::ApprovalLayer`] before the matching `ToolStart`
    /// fires. Only present when an `Approver` is wired (default
    /// agents skip approval and never emit this variant).
    ToolCallApproved {
        /// Run correlation id.
        run_id: String,
        /// Tenant scope this event belongs to (see `Started`).
        tenant_id: TenantId,
        /// Stable tool-use id matching the originating
        /// `ContentPart::ToolUse`. Pairs with the subsequent
        /// `ToolStart` / `ToolComplete` / `ToolError`.
        tool_use_id: String,
        /// Tool name being approved.
        tool: String,
    },

    /// HITL approver decided to reject one tool dispatch. The
    /// matching `ToolStart` does NOT fire — denial short-circuits
    /// the dispatch path. The agent observes the rejection as
    /// `Error::InvalidRequest` carrying the same reason.
    ToolCallDenied {
        /// Run correlation id.
        run_id: String,
        /// Tenant scope this event belongs to (see `Started`).
        tenant_id: TenantId,
        /// Stable tool-use id of the rejected dispatch.
        tool_use_id: String,
        /// Tool name being denied.
        tool: String,
        /// Approver-supplied rationale.
        reason: String,
    },
}

impl<S> AgentEvent<S> {
    /// Project this runtime event onto the durable audit-log shape
    /// `GraphEvent`. Returns `None` when the variant has no audit
    /// projection — `Started`, `Complete`, `Failed` are runtime-only
    /// lifecycle markers that do not belong in the per-thread audit
    /// trail.
    ///
    /// The `timestamp` argument is supplied by the caller (typically
    /// `Utc::now()` at emit time) so this method stays pure: a single
    /// runtime event projected at two different points in time
    /// produces two distinct (but otherwise equal) `GraphEvent`s.
    ///
    /// Lossy projection notes — `run_id`, `tool_version`, and
    /// `duration_ms` are dropped because the audit log keys
    /// correlation by `tool_use_id` + `timestamp` and is not the
    /// home for runtime metrics. Operators who need run-level
    /// correlation in audit do it at the sink layer (e.g. by
    /// stamping a thread tag prior to append).
    ///
    /// `ToolError` is mapped onto a `GraphEvent::ToolResult` with
    /// `is_error: true` and the error message carried as text
    /// content — preserving the same correlation key
    /// (`tool_use_id`) so a session replay can pair the failed
    /// dispatch back with the originating `ToolCall`.
    pub fn to_graph_event(&self, timestamp: DateTime<Utc>) -> Option<GraphEvent> {
        match self {
            // Lifecycle / approval markers are runtime-only — the
            // audit log records the actual `ToolCall` / `ToolResult`
            // pair, not the surrounding gate decisions.
            Self::Started { .. }
            | Self::Complete { .. }
            | Self::Failed { .. }
            | Self::ToolCallApproved { .. }
            | Self::ToolCallDenied { .. } => None,
            Self::ToolStart {
                tool_use_id,
                tool,
                input,
                ..
            } => Some(GraphEvent::ToolCall {
                id: tool_use_id.clone(),
                name: tool.clone(),
                input: input.clone(),
                timestamp,
            }),
            Self::ToolComplete {
                tool_use_id,
                tool,
                output,
                ..
            } => Some(GraphEvent::ToolResult {
                tool_use_id: tool_use_id.clone(),
                name: tool.clone(),
                content: ToolResultContent::Json(output.clone()),
                is_error: false,
                timestamp,
            }),
            Self::ToolError {
                tool_use_id,
                tool,
                error_for_llm,
                ..
            } => Some(GraphEvent::ToolResult {
                tool_use_id: tool_use_id.clone(),
                name: tool.clone(),
                // Audit log carries the LLM-facing rendering — replay
                // and resume paths reconstruct conversation history
                // from `GraphEvent::ToolResult`, so the content here
                // becomes the model's view (invariant #16). The full
                // operator-facing `error` continues to flow through
                // the event sink and OTel.
                content: ToolResultContent::Text(error_for_llm.as_inner().clone()),
                is_error: true,
                timestamp,
            }),
        }
    }

    /// Erase the agent-state type parameter, replacing
    /// [`Self::Complete::state`] with the unit value. Every other
    /// variant rebuilds with identical field values — they carry no
    /// state. Enables a single audit / SSE / OTel sink (typed
    /// [`AgentEventSink<()>`](crate::agent::AgentEventSink)) to fan
    /// in from heterogeneous agents (`Agent<ReActState>`,
    /// `Agent<SupervisorState>`, …) through the
    /// [`StateErasureSink`](crate::agent::StateErasureSink) adapter.
    ///
    /// Operators consuming the post-erasure event tree retain access
    /// to every header field (`run_id`, `tenant_id`, `parent_run_id`)
    /// and every per-variant payload (tool inputs / outputs, error
    /// envelope, usage snapshot) — only the agent's terminal state
    /// is dropped, which is the field a state-agnostic sink could
    /// not type-erase anyway.
    #[allow(clippy::too_many_lines)]
    // 1-to-1 exhaustive variant rebuild — splitting hurts readability and the line count is structural, not accidental.
    #[must_use]
    pub fn erase_state(self) -> AgentEvent<()> {
        match self {
            Self::Started {
                run_id,
                tenant_id,
                parent_run_id,
                agent,
            } => AgentEvent::Started {
                run_id,
                tenant_id,
                parent_run_id,
                agent,
            },
            Self::ToolStart {
                run_id,
                tenant_id,
                tool_use_id,
                tool,
                tool_version,
                input,
            } => AgentEvent::ToolStart {
                run_id,
                tenant_id,
                tool_use_id,
                tool,
                tool_version,
                input,
            },
            Self::ToolComplete {
                run_id,
                tenant_id,
                tool_use_id,
                tool,
                tool_version,
                duration_ms,
                output,
            } => AgentEvent::ToolComplete {
                run_id,
                tenant_id,
                tool_use_id,
                tool,
                tool_version,
                duration_ms,
                output,
            },
            Self::ToolError {
                run_id,
                tenant_id,
                tool_use_id,
                tool,
                tool_version,
                error,
                error_for_llm,
                envelope,
                duration_ms,
            } => AgentEvent::ToolError {
                run_id,
                tenant_id,
                tool_use_id,
                tool,
                tool_version,
                error,
                error_for_llm,
                envelope,
                duration_ms,
            },
            Self::Failed {
                run_id,
                tenant_id,
                error,
                envelope,
            } => AgentEvent::Failed {
                run_id,
                tenant_id,
                error,
                envelope,
            },
            Self::Complete {
                run_id,
                tenant_id,
                state: _,
                usage,
            } => AgentEvent::Complete {
                run_id,
                tenant_id,
                state: (),
                usage,
            },
            Self::ToolCallApproved {
                run_id,
                tenant_id,
                tool_use_id,
                tool,
            } => AgentEvent::ToolCallApproved {
                run_id,
                tenant_id,
                tool_use_id,
                tool,
            },
            Self::ToolCallDenied {
                run_id,
                tenant_id,
                tool_use_id,
                tool,
                reason,
            } => AgentEvent::ToolCallDenied {
                run_id,
                tenant_id,
                tool_use_id,
                tool,
                reason,
            },
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ts() -> DateTime<Utc> {
        chrono::DateTime::parse_from_rfc3339("2026-04-29T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc)
    }

    #[test]
    fn lifecycle_variants_have_no_audit_projection() {
        let tenant = TenantId::new("t-test");
        let started: AgentEvent<u32> = AgentEvent::Started {
            run_id: "r1".into(),
            tenant_id: tenant.clone(),
            parent_run_id: None,
            agent: "a".into(),
        };
        let complete: AgentEvent<u32> = AgentEvent::Complete {
            run_id: "r1".into(),
            tenant_id: tenant.clone(),
            state: 7,
            usage: None,
        };
        let failed: AgentEvent<u32> = AgentEvent::Failed {
            run_id: "r1".into(),
            tenant_id: tenant,
            error: "boom".into(),
            envelope: entelix_core::Error::config("boom").envelope(),
        };
        assert!(started.to_graph_event(ts()).is_none());
        assert!(complete.to_graph_event(ts()).is_none());
        assert!(failed.to_graph_event(ts()).is_none());
    }

    #[test]
    fn tool_start_projects_to_graph_event_tool_call() {
        let event: AgentEvent<u32> = AgentEvent::ToolStart {
            run_id: "r1".into(),
            tenant_id: TenantId::new("t-test"),
            tool_use_id: "tu-1".into(),
            tool: "double".into(),
            tool_version: Some("1.2.0".into()),
            input: json!({"n": 21}),
        };
        let projected = event.to_graph_event(ts()).unwrap();
        match projected {
            GraphEvent::ToolCall {
                id,
                name,
                input,
                timestamp,
            } => {
                assert_eq!(id, "tu-1");
                assert_eq!(name, "double");
                assert_eq!(input, json!({"n": 21}));
                assert_eq!(timestamp, ts());
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn tool_complete_projects_to_successful_tool_result() {
        let event: AgentEvent<u32> = AgentEvent::ToolComplete {
            run_id: "r1".into(),
            tenant_id: TenantId::new("t-test"),
            tool_use_id: "tu-1".into(),
            tool: "double".into(),
            tool_version: Some("1.2.0".into()),
            duration_ms: 42,
            output: json!({"doubled": 42}),
        };
        let projected = event.to_graph_event(ts()).unwrap();
        match projected {
            GraphEvent::ToolResult {
                tool_use_id,
                name,
                content,
                is_error,
                timestamp,
            } => {
                assert_eq!(tool_use_id, "tu-1");
                assert_eq!(name, "double");
                assert!(!is_error, "successful tool dispatch must not flag is_error");
                assert_eq!(timestamp, ts());
                match content {
                    ToolResultContent::Json(v) => assert_eq!(v, json!({"doubled": 42})),
                    other => panic!("expected Json content, got {other:?}"),
                }
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn tool_error_projects_to_error_flagged_tool_result_using_llm_facing_text() {
        use entelix_core::{Error, LlmRenderable};
        // The carrier `RenderedForLlm<String>` is sealed to
        // `entelix-core` — there is no way to fabricate one from a
        // raw `String` here. The only path to populate
        // `error_for_llm` is `LlmRenderable::for_llm` on a value
        // that implements the trait. `Error::provider_http(503,
        // ...).for_llm()` produces the canonical "upstream model
        // error" rendering through the same code path the
        // production tool-event layer uses, so the test exercises
        // the real boundary instead of stubbing past it.
        let source = Error::provider_http(503, "vendor down");
        let envelope = source.envelope();
        let llm_facing = source.for_llm();
        let event: AgentEvent<u32> = AgentEvent::ToolError {
            run_id: "r1".into(),
            tenant_id: TenantId::new("t-test"),
            tool_use_id: "tu-1".into(),
            tool: "double".into(),
            tool_version: None,
            // Operator-facing text — full Display, includes vendor
            // status / source chain. The audit projection MUST NOT
            // surface this to the model channel.
            error: "provider returned 503: vendor down".into(),
            // LLM-facing rendering — short, actionable, no vendor
            // identifiers. The audit projection picks this.
            error_for_llm: llm_facing,
            envelope,
            duration_ms: 7,
        };
        let projected = event.to_graph_event(ts()).unwrap();
        match projected {
            GraphEvent::ToolResult {
                tool_use_id,
                name,
                content,
                is_error,
                ..
            } => {
                assert_eq!(tool_use_id, "tu-1");
                assert_eq!(name, "double");
                assert!(is_error, "ToolError must surface as is_error: true");
                match content {
                    ToolResultContent::Text(s) => {
                        assert_eq!(s, "upstream model error");
                        assert!(
                            !s.contains("provider returned"),
                            "audit log content must use the LLM-facing rendering, not the operator-facing one: {s}"
                        );
                        assert!(
                            !s.contains("503"),
                            "audit log must not leak vendor status code: {s}"
                        );
                    }
                    other => panic!("expected Text content for error, got {other:?}"),
                }
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn projection_is_deterministic_across_calls() {
        // Same event projected with the same timestamp produces the
        // same GraphEvent — required for replay coherence (two
        // operators running the same projection at the same wall
        // clock get the same audit row).
        let event: AgentEvent<u32> = AgentEvent::ToolStart {
            run_id: "r1".into(),
            tenant_id: TenantId::new("t-test"),
            tool_use_id: "tu-1".into(),
            tool: "double".into(),
            tool_version: None,
            input: json!({"n": 21}),
        };
        let a = event.to_graph_event(ts()).unwrap();
        let b = event.to_graph_event(ts()).unwrap();
        assert_eq!(a, b);
    }
}
