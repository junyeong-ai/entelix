//! `EventBus` — ephemeral broadcast pub-sub for SDK observability.
//!
//! Per F1 mitigation, this channel is **distinct from** the durable audit
//! log (`entelix-session::SessionGraph`). `EventBus` is for hooks,
//! observability sinks, and live-tail debug consoles — fire-and-forget,
//! drop-on-overflow. `SessionGraph` is the source of truth for replay.
//!
//! Built on `tokio::sync::broadcast`: lagging subscribers see
//! `RecvError::Lagged(n)` and the next available message; senders never
//! block.

use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::ir::ModelWarning;

/// Default channel capacity (bounded buffer per subscriber).
pub const DEFAULT_CAPACITY: usize = 256;

/// One ephemeral runtime event.
///
/// Add variants conservatively — the bus is wide; flood-prone events
/// belong on dedicated typed buses.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Event {
    /// A model request is about to be encoded and sent.
    RequestStarted {
        /// Codec name (e.g. `"anthropic-messages"`).
        codec: String,
        /// Transport name (e.g. `"direct"`, `"bedrock"`).
        transport: String,
        /// Vendor model identifier.
        model: String,
    },
    /// A model response was decoded.
    RequestCompleted {
        /// Vendor model identifier echoed in the response.
        model: String,
        /// Tokens consumed from the prompt.
        input_tokens: u32,
        /// Tokens produced as output.
        output_tokens: u32,
    },
    /// A non-fatal codec / transport advisory was raised.
    Warning(ModelWarning),
    /// A tool was dispatched.
    ToolDispatched {
        /// Registered tool name.
        name: String,
    },
    /// A tool finished, possibly with an error.
    ToolCompleted {
        /// Registered tool name.
        name: String,
        /// True if the tool returned `Err`.
        error: bool,
    },
}

/// Non-blocking pub-sub channel for ephemeral events.
///
/// Cheap to clone — internal `broadcast::Sender` is reference-counted.
/// Cloning shares the same channel; subscribers see all events from the
/// point they subscribe.
#[derive(Clone)]
pub struct EventBus {
    sender: broadcast::Sender<Event>,
}

impl EventBus {
    /// Build a bus with the supplied per-subscriber buffer capacity.
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self { sender }
    }

    /// Publish an event. Returns the number of subscribers reached
    /// (`0` is fine — the bus does not require any).
    pub fn publish(&self, event: Event) -> usize {
        // `send` returns `Err` only when no receivers exist — equivalent to 0 here.
        self.sender.send(event).unwrap_or_default()
    }

    /// Subscribe to future events. The receiver sees only events
    /// published after this call.
    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.sender.subscribe()
    }

    /// Currently-connected subscriber count.
    pub fn receiver_count(&self) -> usize {
        self.sender.receiver_count()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new(DEFAULT_CAPACITY)
    }
}
