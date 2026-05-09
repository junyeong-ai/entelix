//! Route handlers.
//!
//! Each handler resolves the per-request [`ExecutionContext`]
//! ([`build_ctx`]) before dispatching. The helper reads the tenant
//! header (configurable on the [`AgentRouterBuilder`]) and seeds
//! `with_tenant_id` and `with_thread_id` on a fresh context.
//!
//! [`AgentRouterBuilder`]: crate::AgentRouterBuilder

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::HeaderMap;
use axum::response::IntoResponse;
use axum::response::sse::{Event, KeepAlive, Sse};
use futures::StreamExt;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use entelix_core::ThreadKey;
use entelix_core::approval::{ApprovalDecision, PendingApprovalDecisions};
use entelix_core::cancellation::CancellationToken;
use entelix_core::context::ExecutionContext;
use entelix_runnable::{StreamChunk, StreamMode};

use crate::error::{ServerError, ServerResult};
use crate::router::{AgentRouterState, TenantMode};

// ── shared helpers ───────────────────────────────────────────────

/// Build the per-request [`ExecutionContext`] paired with the
/// [`CancellationToken`] that backs `ctx.cancellation()`.
///
/// The handler holds the returned token through a `drop_guard()` —
/// when the handler future is dropped (the canonical axum signal for
/// client disconnect / connection close / timeout) the guard cancels
/// the token, and any tool polling `ctx.is_cancelled()` exits
/// gracefully instead of running to completion against a closed
/// connection.
fn build_ctx<S>(
    state: &AgentRouterState<S>,
    headers: &HeaderMap,
    thread_id: &str,
) -> ServerResult<(ExecutionContext, CancellationToken)>
where
    S: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    let token = CancellationToken::new();
    let mut ctx = ExecutionContext::with_cancellation(token.clone()).with_thread_id(thread_id);
    match state.tenant_mode() {
        TenantMode::Default => {
            // ctx keeps `ExecutionContext::new()`'s `DEFAULT_TENANT_ID`.
        }
        TenantMode::RequiredHeader { header } => {
            // Strict mode: the header is REQUIRED. Missing,
            // non-UTF-8, or whitespace-only → typed 4xx.
            let value = headers
                .get(header)
                .ok_or_else(|| ServerError::MissingTenantHeader {
                    header: header.as_str().to_owned(),
                })?;
            let raw = value.to_str().map_err(|_| {
                ServerError::BadRequest(format!(
                    "tenant header `{}` is not valid UTF-8",
                    header.as_str()
                ))
            })?;
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                return Err(ServerError::BadRequest(format!(
                    "tenant header `{}` is present but empty",
                    header.as_str()
                )));
            }
            // `TenantId::try_from` rejects empty up-front; the
            // trimmed-empty branch above already short-circuits, so
            // this conversion succeeds for every reachable path.
            // Any future surface that drops the empty-trim guard
            // still surfaces an empty header as a typed 4xx through
            // this validator (invariant 11).
            let tenant = entelix_core::TenantId::try_from(trimmed).map_err(ServerError::Core)?;
            ctx = ctx.with_tenant_id(tenant);
        }
    }
    Ok((ctx, token))
}

// ── /v1/health ───────────────────────────────────────────────────

/// Liveness probe. Always 200.
pub(crate) async fn health() -> &'static str {
    "ok"
}

// ── POST /v1/threads/{thread_id}/runs ────────────────────────────

#[derive(Debug, Deserialize)]
pub(crate) struct RunRequest<S> {
    input: S,
}

#[derive(Debug, Serialize)]
pub(crate) struct RunResponse<S> {
    output: S,
}

/// Synchronous invoke. Body shape: `{"input": <state>}` →
/// `{"output": <state>}`.
pub(crate) async fn run_sync<S>(
    State(state): State<Arc<AgentRouterState<S>>>,
    Path(thread_id): Path<String>,
    headers: HeaderMap,
    Json(body): Json<RunRequest<S>>,
) -> ServerResult<Json<RunResponse<S>>>
where
    S: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    let (ctx, token) = build_ctx(&state, &headers, &thread_id)?;
    let _cancel_on_disconnect = token.drop_guard();
    let output = state.runnable().invoke(body.input, &ctx).await?;
    Ok(Json(RunResponse { output }))
}

// ── GET /v1/threads/{thread_id}/stream ───────────────────────────

#[derive(Debug, Deserialize)]
pub(crate) struct StreamQuery {
    /// `values` | `updates` | `messages` | `debug` | `events`
    /// (default: `values`).
    #[serde(default)]
    mode: Option<String>,
}

fn parse_mode(raw: Option<&str>) -> Result<StreamMode, ServerError> {
    match raw.unwrap_or("values") {
        "values" => Ok(StreamMode::Values),
        "updates" => Ok(StreamMode::Updates),
        "messages" => Ok(StreamMode::Messages),
        "debug" => Ok(StreamMode::Debug),
        "events" => Ok(StreamMode::Events),
        other => Err(ServerError::BadRequest(format!(
            "unknown stream mode '{other}'; expected values|updates|messages|debug|events"
        ))),
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct StreamBody<S> {
    input: S,
}

/// SSE 5-mode stream. The body is `application/json` even on a GET
/// because the input state is structured. We accept the input via a
/// JSON query param (`?input={...}`) when no body is provided to keep
/// the GET safe for browser-side EventSource clients.
///
/// The client passes the input as a JSON-encoded query parameter
/// (`?input=...`) **or** as a request body (most HTTP libs allow GET
/// bodies). axum's `Json` extractor handles either case when the
/// `content-type` is `application/json`.
pub(crate) async fn run_stream<S>(
    State(state): State<Arc<AgentRouterState<S>>>,
    Path(thread_id): Path<String>,
    Query(q): Query<StreamQuery>,
    headers: HeaderMap,
    Json(body): Json<StreamBody<S>>,
) -> ServerResult<axum::response::Response>
where
    S: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    let mode = parse_mode(q.mode.as_deref())?;
    let (ctx, token) = build_ctx(&state, &headers, &thread_id)?;
    let runnable = Arc::clone(state.runnable());
    let (tx, rx) = mpsc::channel::<std::result::Result<Event, Infallible>>(16);

    // The streaming task itself holds the drop guard — when the task
    // ends (client-disconnect channel-send Err, or normal completion)
    // the guard fires the token, and any tool still polling
    // `ctx.is_cancelled()` from inside the stream exits gracefully
    // instead of running against a closed connection.
    let task_guard = token.drop_guard();

    // Clone the sender so the supervisor can emit a structured
    // error event if the streaming task panics — without a guard
    // a panic in `render_chunk` or downstream would close the
    // channel silently and the client would hang.
    let supervisor_tx = tx.clone();
    let join = tokio::spawn(async move {
        let _task_guard = task_guard;
        let stream = match runnable.stream(body.input, mode, &ctx).await {
            Ok(s) => s,
            Err(e) => {
                let _ = tx
                    .send(Ok(Event::default()
                        .event("error")
                        .data(error_payload(&e.to_string()))))
                    .await;
                return;
            }
        };
        futures::pin_mut!(stream);
        while let Some(item) = stream.next().await {
            let event = match item {
                Ok(chunk) => render_chunk(&chunk),
                Err(e) => Event::default()
                    .event("error")
                    .data(error_payload(&e.to_string())),
            };
            if tx.send(Ok(event)).await.is_err() {
                return; // client disconnected
            }
        }
        let _ = tx
            .send(Ok(Event::default().event("done").data("[done]")))
            .await;
    });

    tokio::spawn(async move {
        if let Err(err) = join.await
            && err.is_panic()
        {
            tracing::error!(
                target: "entelix_server::handlers",
                "SSE streaming task panicked"
            );
            let _ = supervisor_tx
                .send(Ok(Event::default()
                    .event("error")
                    .data(error_payload("internal streaming error"))))
                .await;
        }
    });

    let stream: futures::stream::BoxStream<'static, _> = ReceiverStream::new(rx).boxed();
    let sse = Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)));
    Ok(sse.into_response())
}

fn render_chunk<S: Serialize>(chunk: &StreamChunk<S>) -> Event {
    let (kind, data) = match chunk {
        StreamChunk::Value(v) => ("value", serde_json::to_string(v)),
        StreamChunk::Update { node, value } => (
            "update",
            serde_json::to_string(&serde_json::json!({"node": node, "value": value})),
        ),
        StreamChunk::Message(delta) => (
            "message",
            Ok(serde_json::json!({ "kind": format!("{delta:?}") }).to_string()),
        ),
        StreamChunk::Debug(d) => (
            "debug",
            serde_json::to_string(&serde_json::json!({
                "kind": format!("{d:?}"),
            })),
        ),
        StreamChunk::Event(e) => (
            "event",
            serde_json::to_string(&serde_json::json!({
                "kind": format!("{e:?}"),
            })),
        ),
        _ => ("unknown", Ok(String::new())),
    };
    let payload = data.unwrap_or_else(|err| error_payload(&err.to_string()));
    Event::default().event(kind).data(payload)
}

fn error_payload(message: &str) -> String {
    serde_json::json!({ "error": { "message": message } }).to_string()
}

// ── POST /v1/threads/{thread_id}/wake ────────────────────────────

/// Body schema for `/wake`. Mirrors the operator-resumable subset of
/// `entelix_graph::Command<S>`:
///
/// - `{"command": "resume"}`
/// - `{"command": {"update": <state>}}`
/// - `{"command": {"approve_tool": {"tool_use_id": "tu-1", "decision": "approve"}}}`
/// - `{"command": {"approve_tool": {"tool_use_id": "tu-1", "decision": {"reject": {"reason": "denied"}}}}}`
///
/// `Command::GoTo` is intentionally absent — it requires the
/// `CompiledGraph` surface, and the router is generic over any
/// `Runnable<S, S>`. Operators that need explicit node routing
/// drive their `CompiledGraph` directly rather than through this HTTP
/// boundary.
#[derive(Debug, Deserialize)]
pub(crate) struct WakeRequest<S> {
    command: WakeCommand<S>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WakeCommand<S> {
    Resume,
    Update(S),
    ApproveTool {
        tool_use_id: String,
        decision: ApprovalDecision,
    },
}

#[derive(Debug, Serialize)]
pub(crate) struct WakeResponse<S> {
    output: S,
}

/// Resume a thread from the latest checkpoint, applying the supplied
/// command. Requires a checkpointer attached on the router.
///
/// Implementation note: `Runnable<S, S>` does not include
/// `CompiledGraph::resume_with`. To keep the router runnable-agnostic,
/// this handler reconstructs the pre-state from the checkpointer and
/// calls `Runnable::invoke` with the appropriate state — for the
/// `ApproveTool` command this attaches a [`PendingApprovalDecisions`]
/// extension on the request context (the lower-level direct-attachment
/// path documented on `PendingApprovalDecisions`), which is the same
/// effect `Command::ApproveTool` produces inside `resume_with`.
pub(crate) async fn wake<S>(
    State(state): State<Arc<AgentRouterState<S>>>,
    Path(thread_id): Path<String>,
    headers: HeaderMap,
    Json(body): Json<WakeRequest<S>>,
) -> ServerResult<Json<WakeResponse<S>>>
where
    S: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    let cp = state.checkpointer().ok_or_else(|| {
        ServerError::Core(entelix_core::Error::config(
            "wake requires a checkpointer attached via AgentRouterBuilder::with_checkpointer",
        ))
    })?;
    let (ctx, token) = build_ctx(&state, &headers, &thread_id)?;
    let _cancel_on_disconnect = token.drop_guard();
    let key = ThreadKey::from_ctx(&ctx)?;
    let latest = cp.get_latest(&key).await?.ok_or_else(|| {
        ServerError::NotFound(format!(
            "no checkpoint exists for tenant '{}' thread '{}'",
            key.tenant_id(),
            key.thread_id(),
        ))
    })?;
    let (resume_state, ctx) = match body.command {
        WakeCommand::Resume => (latest.state.clone(), ctx),
        WakeCommand::Update(s) => (s, ctx),
        WakeCommand::ApproveTool {
            tool_use_id,
            decision,
        } => {
            if matches!(decision, ApprovalDecision::AwaitExternal) {
                return Err(ServerError::BadRequest(
                    "AwaitExternal is not a valid resume decision".into(),
                ));
            }
            let mut pending = PendingApprovalDecisions::new();
            pending.insert(tool_use_id, decision);
            (latest.state.clone(), ctx.add_extension(pending))
        }
    };
    let output = state.runnable().invoke(resume_state, &ctx).await?;
    Ok(Json(WakeResponse { output }))
}
