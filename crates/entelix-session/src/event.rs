//! `GraphEvent` — the audit-trail unit appended to a `SessionGraph`.
//!
//! Every event is timestamped and serializable, so a persisted log can be
//! replayed verbatim by a fresh process (Anthropic-style `wake(thread_id)`).
//! Events are **strictly additive** — once written, never mutated.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use entelix_core::ir::{ContentPart, ModelWarning, ToolResultContent, Usage};
use entelix_core::rate_limit::RateLimitSnapshot;

/// One audit-log entry.
///
/// Aggregating these (oldest-to-newest) reconstructs the full conversation
/// trace for a thread. Branches and checkpoints are recorded inline so a
/// single linear scan is enough for replay.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum GraphEvent {
    /// User-authored input.
    UserMessage {
        /// Multi-part content (text, image, `tool_result`).
        content: Vec<ContentPart>,
        /// Wall-clock time the event was appended.
        timestamp: DateTime<Utc>,
    },
    /// Assistant reply (after stream aggregation).
    AssistantMessage {
        /// Multi-part content (text, `tool_use`).
        content: Vec<ContentPart>,
        /// Token accounting if reported by the provider.
        usage: Option<Usage>,
        /// Wall-clock time the event was appended.
        timestamp: DateTime<Utc>,
    },
    /// A tool was dispatched by the assistant.
    ToolCall {
        /// Stable tool-use id matching a future `ToolResult`.
        id: String,
        /// Registered tool name.
        name: String,
        /// Tool input as JSON.
        input: serde_json::Value,
        /// Wall-clock time the event was appended.
        timestamp: DateTime<Utc>,
    },
    /// The dispatched tool returned.
    ToolResult {
        /// `ToolCall::id` this result resolves.
        tool_use_id: String,
        /// `ToolCall::name` this result resolves — required by
        /// codecs whose wire format keys correlation by name
        /// (Gemini's `functionResponse`) rather than id.
        name: String,
        /// Result payload.
        content: ToolResultContent,
        /// True if the tool reported an error.
        is_error: bool,
        /// Wall-clock time the event was appended.
        timestamp: DateTime<Utc>,
    },
    /// A branch was forked off this session at the indicated event index.
    /// The new branch's thread id is recorded alongside.
    BranchCreated {
        /// Identifier of the forked sub-session.
        branch_id: String,
        /// Index in `events` (0-based) the branch diverged at.
        parent_event: usize,
        /// Wall-clock time the event was appended.
        timestamp: DateTime<Utc>,
    },
    /// Marker tying this position in the audit log to a `Checkpointer`
    /// snapshot. Cross-tier reference for crash recovery flows that pair
    /// `SessionGraph` (Tier 2) with `StateGraph` checkpoints (Tier 1).
    CheckpointMarker {
        /// Stringified `entelix_graph::CheckpointId`.
        checkpoint_id: String,
        /// Thread the checkpoint was written under (typically same as
        /// the session's thread).
        thread_id: String,
        /// Wall-clock time the event was appended.
        timestamp: DateTime<Utc>,
    },
    /// Codec / runtime advisory captured into the audit trail.
    Warning {
        /// Underlying advisory.
        warning: ModelWarning,
        /// Wall-clock time the event was appended.
        timestamp: DateTime<Utc>,
    },
    /// Streaming thinking-content fragment captured into the audit
    /// trail. Multiple consecutive deltas with the same `signature`
    /// belong to the same logical thinking block; aggregators fold
    /// them into a single `ContentPart::Thinking` when reconstructing
    /// a finalized message. Recording deltas individually keeps the
    /// audit log faithful to the wire — a replay that needs only the
    /// final block can fold the deltas, while a replay that needs
    /// per-token timing has the data.
    ThinkingDelta {
        /// Token text appended to the in-progress thinking block.
        text: String,
        /// Vendor signature for redaction-resistant replay
        /// (Anthropic supplies a discrete `signature_delta` event
        /// or a single `signature` on the block-start event).
        #[serde(default)]
        signature: Option<String>,
        /// Wall-clock time the event was appended.
        timestamp: DateTime<Utc>,
    },
    /// Provider rate-limit snapshot at this position in the
    /// conversation. Operators reading the audit log can correlate a
    /// later throttling failure with the snapshot that warned them.
    /// Recorded inline rather than on a separate metric channel so
    /// the audit trail is self-contained for compliance review.
    RateLimit {
        /// Snapshot the codec extracted from response headers.
        snapshot: RateLimitSnapshot,
        /// Wall-clock time the event was appended.
        timestamp: DateTime<Utc>,
    },
    /// HITL pause point — the runtime asked the host application for
    /// input. The matching resume signal lands in
    /// `entelix_graph::Command` outside the audit log; this event
    /// records that the pause happened and what was visible to the
    /// human at the time.
    Interrupt {
        /// Operator-supplied payload describing the pause point.
        /// Free-form JSON so the agent recipe owns the schema; the
        /// audit log just persists it.
        payload: serde_json::Value,
        /// Wall-clock time the event was appended.
        timestamp: DateTime<Utc>,
    },
    /// The run was cancelled — either via cancellation token or via
    /// a deadline elapsing. Recording the reason inline lets a
    /// replay reconstruct partial-run audit traces faithfully.
    Cancelled {
        /// Lean reason string. Human-readable; not parsed downstream.
        reason: String,
        /// Wall-clock time the event was appended.
        timestamp: DateTime<Utc>,
    },
    /// A sub-agent was dispatched from the parent's run. The parent
    /// `run_id` (recorded on the surrounding `AgentEvent::Started`)
    /// scopes the audit trail; this event ties the parent's
    /// position to the child's `sub_thread_id` so a replay can walk
    /// from parent to child without keying on heuristic timing.
    /// Managed-agent shape — every `Subagent::execute`
    /// call surfaces here as the canonical "brain passes hand"
    /// audit boundary.
    SubAgentInvoked {
        /// Stable identifier the parent uses to refer to the
        /// sub-agent (typically the `Subagent`'s configured name).
        agent_id: String,
        /// Thread the sub-agent ran under. Same as the parent's
        /// thread when the sub-agent shares state; a fresh value
        /// when the sub-agent runs in its own scope.
        sub_thread_id: String,
        /// Wall-clock time the event was appended.
        timestamp: DateTime<Utc>,
    },
    /// A supervisor recipe handed control between named agents.
    /// Distinct from `SubAgentInvoked` — supervisor handoffs route
    /// inside one logical conversation, while sub-agent invocations
    /// open a child run.
    AgentHandoff {
        /// Agent name that finished this turn (`None` on the first
        /// supervisor turn where no agent has spoken yet).
        from: Option<String>,
        /// Agent name the supervisor routed to next.
        to: String,
        /// Wall-clock time the event was appended.
        timestamp: DateTime<Utc>,
    },
    /// A run resumed from a prior checkpoint — either via
    /// `wake(thread_id)` after a crash or via `Command::Resume` from
    /// a HITL pause. Pairs with the `CheckpointMarker` whose id is
    /// referenced so a single linear replay stays coherent across
    /// the suspend / resume seam.
    Resumed {
        /// `CheckpointMarker::checkpoint_id` the resume hydrated
        /// from. Empty string when the resume happened from a fresh
        /// state (operator built the resume payload by hand).
        from_checkpoint: String,
        /// Wall-clock time the event was appended.
        timestamp: DateTime<Utc>,
    },
    /// A long-term memory tier returned hits to the agent. Records
    /// which tier was queried (`semantic` / `entity` / `graph` /
    /// caller-defined), the namespace key (operator identifier for
    /// the slice queried), and the number of hits returned. The
    /// hits themselves stay outside the audit log — the model-facing
    /// content already lands in `AssistantMessage` / `ToolResult`,
    /// and storing the full retrieved corpus inline would balloon
    /// the audit trail.
    MemoryRecall {
        /// Memory tier identifier (typically `"semantic"`,
        /// `"entity"`, `"graph"`, or an operator-supplied label).
        tier: String,
        /// Rendered namespace key the query targeted.
        namespace_key: String,
        /// Number of records returned to the agent.
        hits: usize,
        /// Wall-clock time the event was appended.
        timestamp: DateTime<Utc>,
    },
    /// An [`entelix_core::RunBudget`] axis hit its cap and
    /// short-circuited the run with
    /// `entelix_core::Error::UsageLimitExceeded`. Compliance and
    /// billing audits replay this to attribute breaches per-tenant
    /// per-run; the operator-facing `Error` continues to flow
    /// through the typed dispatch return as well, so the audit
    /// channel's role here is the durable record, not the only
    /// breach signal.
    UsageLimitExceeded {
        /// Axis that breached — `"requests"`, `"input_tokens"`,
        /// `"output_tokens"`, `"total_tokens"`, or `"tool_calls"`.
        /// Stable wire string matching
        /// `entelix_core::run_budget::UsageLimitAxis`'s `Display`
        /// rendering; dashboards key off these without depending
        /// on the typed enum.
        axis: String,
        /// Cap that was set on the breached axis.
        limit: u64,
        /// Counter value at the moment the cap was hit.
        observed: u64,
        /// Wall-clock time the event was appended.
        timestamp: DateTime<Utc>,
    },
    /// A failure surfaced from the model / tool / graph runtime.
    /// Errors that the agent recovers from internally are still
    /// recorded so post-mortems see the full picture.
    Error {
        /// Coarse classification matching `entelix_core::Error`
        /// variants (`"provider"`, `"invalid_request"`, `"config"`,
        /// `"auth"`, `"interrupted"`, `"cancelled"`, `"serde"`,
        /// `"transport"`). Stable wire strings — dashboards key off
        /// these without the SDK leaking internal error layout.
        class: String,
        /// Human-readable summary (`Display` form).
        message: String,
        /// Wall-clock time the event was appended.
        timestamp: DateTime<Utc>,
    },
}

impl GraphEvent {
    /// Borrow the timestamp of any event variant.
    pub const fn timestamp(&self) -> &DateTime<Utc> {
        match self {
            Self::UserMessage { timestamp, .. }
            | Self::AssistantMessage { timestamp, .. }
            | Self::ToolCall { timestamp, .. }
            | Self::ToolResult { timestamp, .. }
            | Self::BranchCreated { timestamp, .. }
            | Self::CheckpointMarker { timestamp, .. }
            | Self::Warning { timestamp, .. }
            | Self::ThinkingDelta { timestamp, .. }
            | Self::RateLimit { timestamp, .. }
            | Self::Interrupt { timestamp, .. }
            | Self::Cancelled { timestamp, .. }
            | Self::SubAgentInvoked { timestamp, .. }
            | Self::AgentHandoff { timestamp, .. }
            | Self::Resumed { timestamp, .. }
            | Self::MemoryRecall { timestamp, .. }
            | Self::UsageLimitExceeded { timestamp, .. }
            | Self::Error { timestamp, .. } => timestamp,
        }
    }
}
