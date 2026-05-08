//! `McpClient` trait + `HttpMcpClient` streamable-http impl.
//!
//! ## Streamable HTTP transport (MCP 2025-03-26)
//!
//! Modern MCP transport: a single endpoint that carries every
//! direction of the conversation:
//!
//! - **Client → Server requests** ride POST. The response is a
//!   single JSON envelope (stateless servers) or an SSE stream
//!   (streamable servers — every event is one JSON-RPC message,
//!   matched to the request by `id`).
//! - **Server → Client requests** ride a long-lived `GET /` SSE
//!   the client opens once after `initialize`. The dispatcher in
//!   this module matches on `method` and routes:
//!     - `roots/list` → [`crate::RootsProvider`] (when configured)
//!     - any other method → JSON-RPC `-32601 Method not found`.
//! - **Notifications** (either direction) are POSTs whose body has
//!   no `id`. We send `notifications/initialized` and
//!   `notifications/roots/list_changed`; the server may emit any
//!   notifications it likes through the SSE listener.
//!
//! Sticky session: when the server returns an `Mcp-Session-Id`
//! header on the `initialize` response, every subsequent client
//! request echoes it. Servers that omit the header signal stateless
//! mode — the client respects that, skips the listener, and
//! `RootsProvider` (if configured) stays dormant.
//!
//! ## Drop semantics
//!
//! The listener is a `tokio::spawn` task; on drop the client
//! cancels its [`tokio_util::sync::CancellationToken`] and aborts
//! the `JoinHandle`. No detached state survives.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use futures::StreamExt;
use parking_lot::{Mutex, RwLock};
use secrecy::ExposeSecret;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use tokio::sync::{Semaphore, TryAcquireError};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use crate::completion::{McpCompletionArgument, McpCompletionReference, McpCompletionResult};
use crate::error::{McpError, McpResult, ResourceBoundKind};
use crate::fsm::{McpClientState, StateCell};
use crate::prompt::{McpPrompt, McpPromptInvocation};
use crate::protocol::{
    ClientCapabilities, ClientInfo, CompleteParams, CompleteResult, ElicitationCapability,
    InitializeParams, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, JsonRpcServerRequest,
    PROTOCOL_VERSION, PromptsGetParams, PromptsListResult, ResourcesListResult,
    ResourcesReadParams, ResourcesReadResult, RootsCapability, RootsListResult, SamplingCapability,
    ToolContent, ToolsCallParams, ToolsCallResult, ToolsListResult,
};
use crate::resource::{McpResource, McpResourceContent};
use crate::server_config::McpServerConfig;
use crate::sse::{find_double_newline, parse_sse_data};
use crate::tool_definition::McpToolDefinition;

const SESSION_ID_HEADER: &str = "mcp-session-id";

/// Backend-agnostic MCP client surface.
#[async_trait]
pub trait McpClient: Send + Sync {
    /// Drive the client through `initialize` →
    /// `notifications/initialized` → `tools/list`. Idempotent: if the
    /// underlying state is already [`McpClientState::Ready`] this
    /// returns the cached tool list.
    async fn initialize(&self) -> McpResult<Vec<McpToolDefinition>>;

    /// Invoke `tools/call`. The returned `Value` is the merged text
    /// content from the MCP response (multi-part responses join with
    /// newlines).
    async fn call_tool(&self, name: &str, arguments: Value) -> McpResult<Value>;

    /// List the resources advertised by the server. Default impl
    /// reports `Ok(vec![])` so test mocks that don't care about
    /// resources stay terse — production [`HttpMcpClient`] always
    /// overrides.
    async fn list_resources(&self) -> McpResult<Vec<McpResource>> {
        Ok(Vec::new())
    }

    /// Fetch one resource's content blocks. Default impl errors with
    /// the JSON-RPC method-not-found code so a mock that didn't opt
    /// in surfaces the gap visibly rather than returning an
    /// ambiguous empty list.
    async fn read_resource(&self, _uri: &str) -> McpResult<Vec<McpResourceContent>> {
        Err(McpError::JsonRpc {
            code: -32601,
            message: "client did not implement resources/read".into(),
        })
    }

    /// List the prompts advertised by the server.
    async fn list_prompts(&self) -> McpResult<Vec<McpPrompt>> {
        Ok(Vec::new())
    }

    /// Bind a prompt's arguments and fetch the resulting transcript.
    async fn prompt(
        &self,
        _name: &str,
        _arguments: BTreeMap<String, String>,
    ) -> McpResult<McpPromptInvocation> {
        Err(McpError::JsonRpc {
            code: -32601,
            message: "client did not implement prompts/get".into(),
        })
    }

    /// Ask the server to complete a partial argument value. Targets
    /// either a prompt argument or a resource-template placeholder.
    async fn complete(
        &self,
        _reference: McpCompletionReference,
        _argument: McpCompletionArgument,
    ) -> McpResult<McpCompletionResult> {
        Err(McpError::JsonRpc {
            code: -32601,
            message: "client did not implement completion/complete".into(),
        })
    }

    /// Notify the server that the client's roots changed. Triggers
    /// the server to re-issue `roots/list` if it cares; otherwise
    /// the notification is a no-op on the server side. The default
    /// impl is a no-op so test mocks ignore the channel.
    async fn notify_roots_changed(&self) -> McpResult<()> {
        Ok(())
    }

    /// Current FSM position — primarily for diagnostics and tests.
    fn state(&self) -> McpClientState;
}

/// Production `McpClient` — JSON-RPC 2.0 over MCP streamable-http.
pub struct HttpMcpClient {
    config: McpServerConfig,
    client: reqwest::Client,
    next_id: AtomicU64,
    state: StateCell,
    cached_tools: RwLock<Option<Vec<McpToolDefinition>>>,
    /// Sticky session id, populated from the server's
    /// `Mcp-Session-Id` response header. `None` ⇒ stateless server.
    session_id: RwLock<Option<String>>,
    /// Background SSE listener task — `Some` once initialized in
    /// streamable mode, `None` in stateless mode.
    listener: Mutex<Option<JoinHandle<()>>>,
    /// Cancellation signal honoured by the listener loop. Cancelled
    /// from `Drop` so the loop returns promptly.
    listener_cancel: CancellationToken,
    /// Concurrency gate for in-flight server-initiated dispatches.
    /// Sized at [`McpServerConfig::listener_concurrency`]; saturation
    /// causes new server requests to be dropped (with `tracing::warn!`)
    /// rather than queued, bounding executor memory under flood.
    server_request_permits: Arc<Semaphore>,
}

impl HttpMcpClient {
    /// Build from a [`McpServerConfig`].
    pub fn new(config: McpServerConfig) -> McpResult<Self> {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| McpError::Config(format!("HTTP client: {e}")))?;
        let permits = Arc::new(Semaphore::new(config.listener_concurrency()));
        Ok(Self {
            config,
            client,
            next_id: AtomicU64::new(1),
            state: StateCell::new(),
            cached_tools: RwLock::new(None),
            session_id: RwLock::new(None),
            listener: Mutex::new(None),
            listener_cancel: CancellationToken::new(),
            server_request_permits: permits,
        })
    }

    fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    fn current_session_id(&self) -> Option<String> {
        self.session_id.read().clone()
    }

    /// Apply per-request boilerplate that every outbound POST / GET
    /// must carry: bearer auth, optional decorator headers, sticky
    /// session id. Centralised so the dispatcher and listener
    /// agree byte-for-byte on what the server sees.
    fn apply_common_headers(
        &self,
        mut req: reqwest::RequestBuilder,
    ) -> McpResult<reqwest::RequestBuilder> {
        if let Some(token) = &self.config.bearer {
            req = req.header("authorization", format!("Bearer {}", token.expose_secret()));
        }
        if let Some(decorator) = self.config.request_decorator() {
            let extras = run_request_decorator(decorator)?;
            for (name, value) in &extras {
                req = req.header(name, value);
            }
        }
        if let Some(sid) = self.current_session_id() {
            req = req.header(SESSION_ID_HEADER, sid);
        }
        Ok(req)
    }

    async fn rpc_call<P, R>(&self, method: &str, params: P) -> McpResult<R>
    where
        P: Serialize,
        R: DeserializeOwned,
    {
        let request_id = self.next_id();
        let body = JsonRpcRequest::new(request_id, method, params);
        let mut request = self
            .client
            .post(self.config.url())
            .header("accept", "application/json, text/event-stream")
            .json(&body);
        request = self.apply_common_headers(request)?;
        let response = request.send().await.map_err(McpError::network)?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(McpError::network_status(format!(
                "HTTP {status}: {}",
                truncate_for_error(&body)
            )));
        }

        // Capture sticky session id when the server sets one. The
        // header lands on the initialize response; later responses
        // either echo it (no-op write) or omit it (no-op read).
        if let Some(sid) = response
            .headers()
            .get(SESSION_ID_HEADER)
            .and_then(|v| v.to_str().ok())
        {
            *self.session_id.write() = Some(sid.to_owned());
        }

        let is_event_stream = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|ct| ct.starts_with("text/event-stream"));

        if is_event_stream {
            self.consume_sse_response(response, request_id).await
        } else {
            let envelope: JsonRpcResponse = response.json().await.map_err(McpError::malformed)?;
            extract_result(envelope)
        }
    }

    async fn rpc_notify<P>(&self, method: &str, params: P) -> McpResult<()>
    where
        P: Serialize,
    {
        let body = JsonRpcNotification::new(method, params);
        let mut request = self.client.post(self.config.url()).json(&body);
        request = self.apply_common_headers(request)?;
        let _ = request.send().await.map_err(McpError::network)?;
        Ok(())
    }

    /// Drain an SSE-framed POST response until EOF, capturing the
    /// JSON-RPC response that matches `expected_id` while
    /// dispatching any server-initiated request that rides the
    /// same stream.
    ///
    /// Returning early on the first matching response would silently
    /// drop subsequent server-initiated frames the spec allows
    /// inside a POST-response SSE — the streamable-http design
    /// expects client + server to share one connection until the
    /// server closes it. Real MCP servers terminate the stream
    /// promptly after the matching response, so the EOF wait is
    /// bounded.
    async fn consume_sse_response<R: DeserializeOwned>(
        &self,
        response: reqwest::Response,
        expected_id: u64,
    ) -> McpResult<R> {
        let mut bytes = response.bytes_stream();
        let mut buf: Vec<u8> = Vec::new();
        let mut matched: Option<JsonRpcResponse> = None;
        let cap = self.config.max_frame_bytes();
        loop {
            match bytes.next().await {
                Some(Ok(chunk)) => buf.extend_from_slice(&chunk),
                Some(Err(e)) => return Err(McpError::network(e)),
                None => break,
            }
            if buf.len() > cap {
                return Err(McpError::ResourceBounded {
                    kind: ResourceBoundKind::FrameSize,
                    message: format!(
                        "MCP SSE frame exceeded {cap} bytes without `\\n\\n` terminator; closing connection"
                    ),
                });
            }
            while let Some(pos) = find_double_newline(&buf) {
                let frame: Vec<u8> = buf.drain(..pos.saturating_add(2)).collect();
                let Ok(frame_str) = std::str::from_utf8(&frame) else {
                    continue;
                };
                let Some(payload) = parse_sse_data(frame_str) else {
                    continue;
                };
                let value: Value = serde_json::from_str(&payload).map_err(McpError::from)?;
                match classify_inbound(&value, expected_id) {
                    Inbound::MatchingResponse if matched.is_none() => {
                        matched = Some(serde_json::from_value(value).map_err(McpError::from)?);
                    }
                    Inbound::MatchingResponse => {
                        debug!(
                            "duplicate response for id {expected_id} on the same SSE — ignoring"
                        );
                    }
                    Inbound::ServerRequest(request) => {
                        self.spawn_handle_server_request(request);
                    }
                    Inbound::ServerNotification => {
                        // No-op for now — future variants (logging,
                        // resource-list-changed) hook here.
                    }
                    Inbound::Other => {
                        debug!(?value, "unrecognised SSE payload — ignoring");
                    }
                }
            }
        }
        let envelope = matched.ok_or_else(|| {
            McpError::malformed_msg(format!(
                "SSE stream ended without a response matching id {expected_id}"
            ))
        })?;
        extract_result(envelope)
    }

    fn spawn_handle_server_request(&self, request: JsonRpcServerRequest) {
        let client = self.client.clone();
        let config = self.config.clone();
        let session_id = self.current_session_id();
        let cancel = self.listener_cancel.clone();
        let permits = Arc::clone(&self.server_request_permits);
        spawn_bounded_dispatch(permits, client, config, session_id, cancel, request);
    }

    fn ensure_listener(&self) {
        let session_id = match self.current_session_id() {
            Some(sid) => sid,
            None => return, // stateless server — no listener
        };
        let mut guard = self.listener.lock();
        if guard.is_some() {
            return;
        }
        let handle = spawn_listener(
            self.client.clone(),
            self.config.clone(),
            session_id,
            self.listener_cancel.clone(),
            Arc::clone(&self.server_request_permits),
        );
        *guard = Some(handle);
    }
}

impl Drop for HttpMcpClient {
    fn drop(&mut self) {
        self.listener_cancel.cancel();
        if let Some(handle) = self.listener.lock().take() {
            handle.abort();
        }
    }
}

#[async_trait]
impl McpClient for HttpMcpClient {
    async fn initialize(&self) -> McpResult<Vec<McpToolDefinition>> {
        if matches!(self.state.load(), McpClientState::Ready)
            && let Some(cached) = self.cached_tools.read().clone()
        {
            return Ok(cached);
        }

        self.state.advance(McpClientState::Spawn);
        self.state.advance(McpClientState::Handshake);
        self.state.advance(McpClientState::InitializeProtocol);

        let capabilities = ClientCapabilities {
            roots: self
                .config
                .roots_provider()
                .map(|_| RootsCapability { list_changed: true }),
            elicitation: self
                .config
                .elicitation_provider()
                .map(|_| ElicitationCapability {}),
            sampling: self
                .config
                .sampling_provider()
                .map(|_| SamplingCapability {}),
        };

        let _: Value = self
            .rpc_call(
                "initialize",
                InitializeParams {
                    protocol_version: PROTOCOL_VERSION,
                    capabilities,
                    client_info: ClientInfo {
                        name: "entelix-mcp",
                        version: env!("CARGO_PKG_VERSION"),
                    },
                },
            )
            .await
            .inspect_err(|_| {
                self.state.advance(McpClientState::Failed);
            })?;

        self.state.advance(McpClientState::NegotiateCapabilities);
        self.rpc_notify("notifications/initialized", json!({}))
            .await
            .inspect_err(|_| {
                self.state.advance(McpClientState::Failed);
            })?;

        // Open the long-lived `GET /` listener so the server can
        // start issuing server-initiated requests (`roots/list`,
        // future sampling/elicitation). Stateless servers (no
        // session id) skip the listener entirely.
        self.ensure_listener();

        self.state.advance(McpClientState::ListTools);
        let tools_result: ToolsListResult = self
            .rpc_call("tools/list", json!({}))
            .await
            .inspect_err(|_| {
                self.state.advance(McpClientState::Failed);
            })?;
        let tools = tools_result.tools;
        // ListResources / ListPrompts intentionally skipped — the
        // FSM advances through them so existing diagnostics stay
        // unchanged; the actual fetches are demand-driven via
        // `list_resources` / `list_prompts`.
        self.state.advance(McpClientState::ListResources);
        self.state.advance(McpClientState::ListPrompts);
        self.state.advance(McpClientState::CacheWarmup);
        self.state.advance(McpClientState::Ready);
        *self.cached_tools.write() = Some(tools.clone());
        Ok(tools)
    }

    async fn call_tool(&self, name: &str, arguments: Value) -> McpResult<Value> {
        if !matches!(self.state.load(), McpClientState::Ready) {
            self.initialize().await?;
        }
        let result: ToolsCallResult = self
            .rpc_call("tools/call", ToolsCallParams { name, arguments })
            .await?;
        if result.is_error {
            return Err(McpError::JsonRpc {
                code: -32000,
                message: format!("MCP tool '{name}' reported is_error: true"),
            });
        }
        let mut text_parts = Vec::new();
        for c in result.content {
            if let ToolContent::Text { text } = c {
                text_parts.push(text);
            }
        }
        Ok(Value::String(text_parts.join("\n")))
    }

    async fn list_resources(&self) -> McpResult<Vec<McpResource>> {
        if !matches!(self.state.load(), McpClientState::Ready) {
            self.initialize().await?;
        }
        let result: ResourcesListResult = self.rpc_call("resources/list", json!({})).await?;
        Ok(result.resources)
    }

    async fn read_resource(&self, uri: &str) -> McpResult<Vec<McpResourceContent>> {
        if !matches!(self.state.load(), McpClientState::Ready) {
            self.initialize().await?;
        }
        let result: ResourcesReadResult = self
            .rpc_call("resources/read", ResourcesReadParams { uri })
            .await?;
        Ok(result.contents)
    }

    async fn list_prompts(&self) -> McpResult<Vec<McpPrompt>> {
        if !matches!(self.state.load(), McpClientState::Ready) {
            self.initialize().await?;
        }
        let result: PromptsListResult = self.rpc_call("prompts/list", json!({})).await?;
        Ok(result.prompts)
    }

    async fn prompt(
        &self,
        name: &str,
        arguments: BTreeMap<String, String>,
    ) -> McpResult<McpPromptInvocation> {
        if !matches!(self.state.load(), McpClientState::Ready) {
            self.initialize().await?;
        }
        let result: McpPromptInvocation = self
            .rpc_call("prompts/get", PromptsGetParams { name, arguments })
            .await?;
        Ok(result)
    }

    async fn complete(
        &self,
        reference: McpCompletionReference,
        argument: McpCompletionArgument,
    ) -> McpResult<McpCompletionResult> {
        if !matches!(self.state.load(), McpClientState::Ready) {
            self.initialize().await?;
        }
        let result: CompleteResult = self
            .rpc_call(
                "completion/complete",
                CompleteParams {
                    reference,
                    argument,
                },
            )
            .await?;
        Ok(result.completion)
    }

    async fn notify_roots_changed(&self) -> McpResult<()> {
        if !matches!(self.state.load(), McpClientState::Ready) {
            self.initialize().await?;
        }
        self.rpc_notify("notifications/roots/list_changed", json!({}))
            .await
    }

    fn state(&self) -> McpClientState {
        self.state.load()
    }
}

// ── inbound classification ─────────────────────────────────────────

enum Inbound {
    /// JSON-RPC response whose id matches the originating client
    /// request. The dispatcher returns its result.
    MatchingResponse,
    /// Server-initiated request (has both `id` and `method`).
    /// Dispatched to `handle_server_request`.
    ServerRequest(JsonRpcServerRequest),
    /// Notification from the server (`method` present, `id`
    /// absent). Logged but not acted on in 1.0.
    ServerNotification,
    /// Anything else — malformed envelope, unrelated response id.
    Other,
}

fn classify_inbound(value: &Value, expected_id: u64) -> Inbound {
    let has_id = value.get("id").is_some();
    let has_method = value.get("method").is_some();
    match (has_id, has_method) {
        (true, true) => match serde_json::from_value::<JsonRpcServerRequest>(value.clone()) {
            Ok(req) => Inbound::ServerRequest(req),
            Err(_) => Inbound::Other,
        },
        (true, false) => {
            let response_id = value.get("id").and_then(Value::as_u64).unwrap_or(u64::MAX);
            if response_id == expected_id {
                Inbound::MatchingResponse
            } else {
                Inbound::Other
            }
        }
        (false, true) => Inbound::ServerNotification,
        (false, false) => Inbound::Other,
    }
}

fn extract_result<R: DeserializeOwned>(envelope: JsonRpcResponse) -> McpResult<R> {
    if let Some(err) = envelope.error {
        return Err(McpError::JsonRpc {
            code: err.code,
            message: err.message,
        });
    }
    let result = envelope
        .result
        .ok_or_else(|| McpError::malformed_msg("response missing both `result` and `error`"))?;
    serde_json::from_value(result).map_err(McpError::from)
}

// ── server-initiated request dispatch ──────────────────────────────

async fn handle_server_request(
    client: &reqwest::Client,
    config: &McpServerConfig,
    session_id: Option<&str>,
    request: JsonRpcServerRequest,
) {
    let response = match request.method.as_str() {
        "roots/list" => match config.roots_provider() {
            Some(provider) => match provider.list_roots().await {
                Ok(roots) => success_response(&request.id, RootsListResult { roots }),
                Err(e) => {
                    tracing::error!(target: "entelix.mcp.dispatch", error = %e, "RootsProvider failed");
                    error_response(&request.id, -32603, "RootsProvider failed".into())
                }
            },
            None => error_response(
                &request.id,
                -32601,
                "Method not found: roots/list (no RootsProvider configured)".into(),
            ),
        },
        "elicitation/create" => match config.elicitation_provider() {
            Some(provider) => match serde_json::from_value::<crate::ElicitationRequest>(
                request.params.clone().unwrap_or(serde_json::Value::Null),
            ) {
                Ok(elicit_req) => match provider.elicit(elicit_req).await {
                    Ok(resp) => success_response(&request.id, resp),
                    Err(e) => {
                        tracing::error!(target: "entelix.mcp.dispatch", error = %e, "ElicitationProvider failed");
                        error_response(&request.id, -32603, "ElicitationProvider failed".into())
                    }
                },
                Err(e) => {
                    tracing::error!(target: "entelix.mcp.dispatch", error = %e, "Invalid elicitation params");
                    error_response(&request.id, -32602, "Invalid elicitation params".into())
                }
            },
            None => error_response(
                &request.id,
                -32601,
                "Method not found: elicitation/create (no ElicitationProvider configured)".into(),
            ),
        },
        "sampling/createMessage" => match config.sampling_provider() {
            Some(provider) => match serde_json::from_value::<crate::SamplingRequest>(
                request.params.clone().unwrap_or(serde_json::Value::Null),
            ) {
                Ok(sample_req) => match provider.sample(sample_req).await {
                    Ok(resp) => success_response(&request.id, resp),
                    Err(e) => {
                        tracing::error!(target: "entelix.mcp.dispatch", error = %e, "SamplingProvider failed");
                        error_response(&request.id, -32603, "SamplingProvider failed".into())
                    }
                },
                Err(e) => {
                    tracing::error!(target: "entelix.mcp.dispatch", error = %e, "Invalid sampling params");
                    error_response(&request.id, -32602, "Invalid sampling params".into())
                }
            },
            None => error_response(
                &request.id,
                -32601,
                "Method not found: sampling/createMessage (no SamplingProvider configured)".into(),
            ),
        },
        other => error_response(&request.id, -32601, format!("Method not found: {other}")),
    };
    if let Err(e) = post_server_response(client, config, session_id, &response).await {
        warn!(error = %e, "failed to deliver MCP server-request response");
    }
}

fn success_response<T: Serialize>(id: &Value, result: T) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    })
}

fn error_response(id: &Value, code: i64, message: String) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message },
    })
}

async fn post_server_response(
    client: &reqwest::Client,
    config: &McpServerConfig,
    session_id: Option<&str>,
    body: &Value,
) -> McpResult<()> {
    let mut req = client.post(config.url()).json(body);
    if let Some(token) = &config.bearer {
        req = req.header("authorization", format!("Bearer {}", token.expose_secret()));
    }
    if let Some(decorator) = config.request_decorator() {
        let extras = run_request_decorator(decorator)?;
        for (name, value) in &extras {
            req = req.header(name, value);
        }
    }
    if let Some(sid) = session_id {
        req = req.header(SESSION_ID_HEADER, sid);
    }
    let _ = req.send().await.map_err(McpError::network)?;
    Ok(())
}

// ── background SSE listener ────────────────────────────────────────

fn spawn_listener(
    client: reqwest::Client,
    config: McpServerConfig,
    session_id: String,
    cancel: CancellationToken,
    permits: Arc<Semaphore>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        if cancel.is_cancelled() {
            return;
        }
        let mut req = client
            .get(config.url())
            .header("accept", "text/event-stream")
            .header(SESSION_ID_HEADER, &session_id);
        if let Some(token) = &config.bearer {
            req = req.header("authorization", format!("Bearer {}", token.expose_secret()));
        }
        if let Some(decorator) = config.request_decorator() {
            match run_request_decorator(decorator) {
                Ok(extras) => {
                    for (name, value) in &extras {
                        req = req.header(name, value);
                    }
                }
                Err(e) => {
                    warn!(error = %e, "MCP listener decorator failed; aborting listener");
                    return;
                }
            }
        }

        let response = match req.send().await {
            Ok(r) if r.status().is_success() => r,
            Ok(r) => {
                debug!(status = %r.status(), "MCP listener got non-success status; closing");
                return;
            }
            Err(e) => {
                debug!(error = %e, "MCP listener failed to open SSE stream; closing");
                return;
            }
        };

        let cap = config.max_frame_bytes();
        let mut bytes = response.bytes_stream();
        let mut buf: Vec<u8> = Vec::new();
        loop {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => return,
                chunk = bytes.next() => match chunk {
                    Some(Ok(c)) => buf.extend_from_slice(&c),
                    Some(Err(_)) => return,
                    None => return,
                }
            }
            if buf.len() > cap {
                warn!(
                    cap,
                    "MCP listener SSE frame exceeded byte cap without `\\n\\n` terminator; closing connection"
                );
                return;
            }
            while let Some(pos) = find_double_newline(&buf) {
                let frame: Vec<u8> = buf.drain(..pos.saturating_add(2)).collect();
                let Ok(frame_str) = std::str::from_utf8(&frame) else {
                    continue;
                };
                let Some(payload) = parse_sse_data(frame_str) else {
                    continue;
                };
                let Ok(value) = serde_json::from_str::<Value>(&payload) else {
                    debug!(payload, "MCP listener: malformed JSON in SSE frame");
                    continue;
                };
                if let Ok(request) = serde_json::from_value::<JsonRpcServerRequest>(value.clone()) {
                    spawn_bounded_dispatch(
                        Arc::clone(&permits),
                        client.clone(),
                        config.clone(),
                        Some(session_id.clone()),
                        cancel.clone(),
                        request,
                    );
                } else {
                    // Notifications and unrelated payloads — log
                    // and continue. Future server-initiated
                    // surfaces (sampling, elicitation) plug their
                    // method dispatch here.
                    debug!(
                        ?value,
                        "MCP listener: server-side notification or stray payload"
                    );
                }
            }
        }
    })
}

/// Try to acquire a permit from `permits`, then `tokio::spawn` the
/// `handle_server_request` future under that permit. On saturation
/// (no permits available), drop the request and emit `tracing::warn!` —
/// the server is expected to retry per its own cadence.
///
/// The permit is moved into the spawned future so it stays held for
/// the dispatch's lifetime; dropping the future returns the permit.
fn spawn_bounded_dispatch(
    permits: Arc<Semaphore>,
    client: reqwest::Client,
    config: McpServerConfig,
    session_id: Option<String>,
    cancel: CancellationToken,
    request: JsonRpcServerRequest,
) {
    let permit = match Arc::clone(&permits).try_acquire_owned() {
        Ok(p) => p,
        Err(TryAcquireError::NoPermits) => {
            warn!(
                method = %request.method,
                available = permits.available_permits(),
                "MCP server-initiated dispatch concurrency cap reached; dropping request"
            );
            return;
        }
        Err(TryAcquireError::Closed) => {
            // Unreachable today — `HttpMcpClient` never closes the
            // semaphore. Surface the distinct case so a future
            // refactor that does close it produces a precise
            // diagnostic instead of a phantom "cap reached" message.
            warn!(
                method = %request.method,
                "MCP server-initiated dispatch dropped — listener semaphore closed"
            );
            return;
        }
    };
    tokio::spawn(async move {
        let _permit = permit;
        tokio::select! {
            _ = cancel.cancelled() => {},
            _ = handle_server_request(&client, &config, session_id.as_deref(), request) => {},
        }
    });
}

// ── operator-supplied closure plumbing ─────────────────────────────

/// Invoke an operator-supplied [`crate::RequestDecorator`] inside a
/// [`std::panic::catch_unwind`] guard, surfacing a panic as
/// [`McpError::Config`] rather than tearing down the async task.
fn run_request_decorator(
    decorator: &crate::server_config::RequestDecorator,
) -> McpResult<reqwest::header::HeaderMap> {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let mut headers = reqwest::header::HeaderMap::new();
    let result = catch_unwind(AssertUnwindSafe(|| {
        (decorator)(&mut headers);
    }));
    match result {
        Ok(()) => Ok(headers),
        Err(payload) => Err(McpError::Config(format!(
            "MCP request decorator panicked: {}",
            extract_panic_message(&payload)
        ))),
    }
}

const ERROR_BODY_TRUNCATION_BYTES: usize = 512;

fn truncate_for_error(body: &str) -> String {
    if body.len() <= ERROR_BODY_TRUNCATION_BYTES {
        return body.to_owned();
    }
    let mut cut = ERROR_BODY_TRUNCATION_BYTES;
    while cut > 0 && !body.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}… ({} bytes truncated)", &body[..cut], body.len() - cut)
}

fn extract_panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        return (*s).to_owned();
    }
    if let Some(s) = payload.downcast_ref::<String>() {
        return s.clone();
    }
    "unknown panic payload".to_owned()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::protocol::JsonRpcServerRequest;
    use serde_json::json;

    fn fake_request() -> JsonRpcServerRequest {
        JsonRpcServerRequest {
            id: json!(1),
            method: "roots/list".to_owned(),
            params: None,
        }
    }

    #[tokio::test]
    async fn server_dispatch_dropped_when_concurrency_saturated() {
        // Saturated semaphore — no permit available for the next dispatch.
        let permits = Arc::new(Semaphore::new(1));
        let _held = Arc::clone(&permits).try_acquire_owned().unwrap();
        assert_eq!(permits.available_permits(), 0);

        let client = reqwest::Client::new();
        let config = McpServerConfig::http("hostile", "http://127.0.0.1:1").unwrap();
        let cancel = CancellationToken::new();

        spawn_bounded_dispatch(
            Arc::clone(&permits),
            client,
            config,
            Some("sid".to_owned()),
            cancel,
            fake_request(),
        );

        // The request was dropped at the gate — no spawn happened, so
        // the held permit count is unchanged. (A successful dispatch
        // would have moved a permit into the spawned future.)
        assert_eq!(permits.available_permits(), 0);
    }

    #[tokio::test]
    async fn server_dispatch_returns_permit_after_completion() {
        let permits = Arc::new(Semaphore::new(1));
        assert_eq!(permits.available_permits(), 1);

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(50))
            .build()
            .unwrap();
        // Unroutable address so `handle_server_request`'s POST fails
        // fast — the dispatch future drops, returning the permit.
        let config = McpServerConfig::http("hostile", "http://127.0.0.1:1").unwrap();
        let cancel = CancellationToken::new();

        spawn_bounded_dispatch(
            Arc::clone(&permits),
            client,
            config,
            Some("sid".to_owned()),
            cancel,
            fake_request(),
        );

        // Yield until the spawned dispatch settles and releases the
        // permit. `yield_now` cooperatively schedules the dispatch
        // task between checks without burning a wall-clock budget,
        // and the outer `timeout` bounds the wait deterministically
        // even on slow CI runners.
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while permits.available_permits() != 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("permit not returned within 2s");
    }
}
