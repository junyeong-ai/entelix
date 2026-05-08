//! `CompiledGraph<S>` — frozen, executable form of a `StateGraph`.
//!
//! Implements `Runnable<S, S>` so a compiled graph composes via `.pipe()`
//! and serves as a node inside another `StateGraph` (sub-graph
//! composition).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use entelix_core::{Error, ExecutionContext, Result, ThreadKey};
use entelix_runnable::Runnable;
use entelix_runnable::stream::{BoxStream, DebugEvent, RunnableEvent, StreamChunk, StreamMode};

use crate::checkpoint::{Checkpoint, CheckpointId, Checkpointer};
use crate::command::Command;
use crate::finalizing_stream::FinalizingStream;
use crate::state_graph::END;

/// Closure that picks a conditional-edge target by inspecting state.
pub type EdgeSelector<S> = Arc<dyn Fn(&S) -> String + Send + Sync>;

/// One conditional-edge dispatch.
///
/// Exposed for advanced users that build `CompiledGraph`s outside the
/// standard `StateGraph::compile` flow. Most callers use
/// [`crate::StateGraph::add_conditional_edges`] instead.
pub struct ConditionalEdge<S>
where
    S: Clone + Send + Sync + 'static,
{
    /// Function that produces a routing key from the current state.
    pub selector: EdgeSelector<S>,
    /// Routing-key → target-node mapping. Targets are either node names
    /// or [`crate::END`].
    pub mapping: HashMap<String, String>,
}

/// Closure returning the parallel branches a send edge dispatches.
/// Each pair is `(target_node_name, branch_state)`; target nodes
/// run concurrently with their respective branch states.
pub type SendSelector<S> = Arc<dyn Fn(&S) -> Vec<(String, S)> + Send + Sync>;

/// Two-state merger applied per branch result during a send-edge
/// fold — the dispatch loop calls it once per branch with
/// `(folded_so_far, branch_state) -> next_folded`.
///
/// In production this is constructed by `StateGraph::add_send_edges`
/// as a thin wrap over the state's
/// [`StateMerge::merge`](crate::StateMerge::merge) impl, so
/// adding new state fields automatically participates in the join
/// shape — no manual reducer plumbing on the call site.
pub type SendMerger<S> = Arc<dyn Fn(S, S) -> S + Send + Sync>;

/// Parallel fan-out edge.
///
/// After the source node completes, the runtime evaluates `selector`
/// to obtain branches, runs each target node in parallel, folds the
/// post-branch states via the state's
/// [`StateMerge::merge`](crate::StateMerge::merge) impl, then jumps
/// to `join`.
///
/// The target list is stored as both an ordered `Vec` (preserving
/// the declaration order the operator passed to
/// [`crate::StateGraph::add_send_edges`]) and a `HashSet` for
/// O(1) membership checks at dispatch time. The split lets compile-
/// error messages and Debug output reflect the operator's source
/// order — flaky test output and grep-unfriendly logs would
/// otherwise leak into every dashboard.
pub struct SendEdge<S>
where
    S: Clone + Send + Sync + 'static,
{
    /// Declaration-ordered target list. Public accessor —
    /// inspected by tooling that introspects the compiled graph
    /// (visualisers, doc generators).
    targets: Vec<String>,
    /// Lookup-optimised view of `targets`. Built once at edge
    /// construction; never mutated.
    targets_set: HashSet<String>,
    /// Branch-set producer. Must return `(target, branch_state)`
    /// pairs whose `target` is a member of `targets`.
    pub selector: SendSelector<S>,
    /// Per-branch merger applied during the post-fan-out fold —
    /// see [`SendMerger`].
    pub merger: SendMerger<S>,
    /// Node that receives the merged state, or [`crate::END`] for
    /// terminal fan-outs that complete after the merge.
    pub join: String,
}

impl<S> SendEdge<S>
where
    S: Clone + Send + Sync + 'static,
{
    /// Construct a `SendEdge` from its parts. The target list is
    /// deduplicated while preserving first-occurrence order — a
    /// repeated declaration is a no-op (the deduped name still
    /// dispatches once).
    pub fn new(
        targets: impl IntoIterator<Item = String>,
        selector: SendSelector<S>,
        merger: SendMerger<S>,
        join: String,
    ) -> Self {
        let mut ordered: Vec<String> = Vec::new();
        let mut set: HashSet<String> = HashSet::new();
        for t in targets {
            if set.insert(t.clone()) {
                ordered.push(t);
            }
        }
        Self {
            targets: ordered,
            targets_set: set,
            selector,
            merger,
            join,
        }
    }

    /// Borrow the declaration-ordered target list. Order matches
    /// the names the operator passed to
    /// [`crate::StateGraph::add_send_edges`].
    pub fn targets(&self) -> &[String] {
        &self.targets
    }

    /// True when `name` is a declared dispatch target. O(1) via
    /// the internal `HashSet`.
    pub fn has_target(&self, name: &str) -> bool {
        self.targets_set.contains(name)
    }
}

/// Frozen graph ready to execute.
///
/// Built by [`crate::StateGraph::compile`]; use [`Runnable::invoke`] to run.
pub struct CompiledGraph<S>
where
    S: Clone + Send + Sync + 'static,
{
    nodes: HashMap<String, Arc<dyn Runnable<S, S>>>,
    edges: HashMap<String, String>,
    conditional_edges: HashMap<String, ConditionalEdge<S>>,
    send_edges: HashMap<String, SendEdge<S>>,
    entry_point: String,
    finish_points: HashSet<String>,
    recursion_limit: usize,
    checkpointer: Option<Arc<dyn Checkpointer<S>>>,
    checkpoint_granularity: crate::state_graph::CheckpointGranularity,
    interrupt_before: HashSet<String>,
    interrupt_after: HashSet<String>,
}

impl<S> std::fmt::Debug for CompiledGraph<S>
where
    S: Clone + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Deterministic Debug output across runs: sort every set /
        // map-keys projection so test snapshots and operator log
        // diffs are stable. `HashMap`/`HashSet` iteration order is
        // unspecified, so unsorted Debug output produced flaky
        // diffs the moment a node was added.
        f.debug_struct("CompiledGraph")
            .field("nodes", &sorted_keys(&self.nodes))
            .field("edges", &sorted_pairs(&self.edges))
            .field("conditional_edges", &sorted_keys(&self.conditional_edges))
            .field("send_edges", &sorted_keys(&self.send_edges))
            .field("entry_point", &self.entry_point)
            .field("finish_points", &sorted_set(&self.finish_points))
            .field("recursion_limit", &self.recursion_limit)
            .field("has_checkpointer", &self.checkpointer.is_some())
            .field("checkpoint_granularity", &self.checkpoint_granularity)
            .field("interrupt_before", &sorted_set(&self.interrupt_before))
            .field("interrupt_after", &sorted_set(&self.interrupt_after))
            .finish()
    }
}

fn sorted_keys<V>(m: &HashMap<String, V>) -> Vec<&String> {
    let mut out: Vec<&String> = m.keys().collect();
    out.sort();
    out
}

fn sorted_pairs(m: &HashMap<String, String>) -> Vec<(&String, &String)> {
    let mut out: Vec<(&String, &String)> = m.iter().collect();
    out.sort_by_key(|(k, _)| k.as_str());
    out
}

fn sorted_set(s: &HashSet<String>) -> Vec<&String> {
    let mut out: Vec<&String> = s.iter().collect();
    out.sort();
    out
}

impl<S> CompiledGraph<S>
where
    S: Clone + Send + Sync + 'static,
{
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        nodes: HashMap<String, Arc<dyn Runnable<S, S>>>,
        edges: HashMap<String, String>,
        conditional_edges: HashMap<String, ConditionalEdge<S>>,
        send_edges: HashMap<String, SendEdge<S>>,
        entry_point: String,
        finish_points: HashSet<String>,
        recursion_limit: usize,
        checkpointer: Option<Arc<dyn Checkpointer<S>>>,
        checkpoint_granularity: crate::state_graph::CheckpointGranularity,
        interrupt_before: HashSet<String>,
        interrupt_after: HashSet<String>,
    ) -> Self {
        Self {
            nodes,
            edges,
            conditional_edges,
            send_edges,
            entry_point,
            finish_points,
            recursion_limit,
            checkpointer,
            checkpoint_granularity,
            interrupt_before,
            interrupt_after,
        }
    }

    /// Effective checkpoint granularity. Honored by the executor
    /// when a [`Checkpointer`] is also attached.
    pub const fn checkpoint_granularity(&self) -> crate::state_graph::CheckpointGranularity {
        self.checkpoint_granularity
    }

    /// Borrow the entry-point node name.
    pub fn entry_point(&self) -> &str {
        &self.entry_point
    }

    /// Effective recursion limit.
    pub const fn recursion_limit(&self) -> usize {
        self.recursion_limit
    }

    /// Number of finish points.
    pub fn finish_point_count(&self) -> usize {
        self.finish_points.len()
    }

    /// True when a `Checkpointer` is bound to this graph.
    pub fn has_checkpointer(&self) -> bool {
        self.checkpointer.is_some()
    }

    /// Resume execution from the most recent checkpoint for the
    /// `(tenant_id, thread_id)` pair derived from `ctx`, continuing
    /// as-is. Equivalent to calling [`Self::resume_with`] with
    /// [`Command::Resume`].
    pub async fn resume(&self, ctx: &ExecutionContext) -> Result<S> {
        self.resume_with(Command::Resume, ctx).await
    }

    /// Resume execution applying a [`Command`] from the most recent
    /// checkpoint for the `(tenant_id, thread_id)` derived from
    /// `ctx`.
    ///
    /// Returns `Error::Config` if the graph has no checkpointer
    /// attached or `ctx.thread_id()` is unset; returns
    /// `Error::InvalidRequest` if the resolved [`ThreadKey`] has no
    /// recorded checkpoints. If the resolved next-node is `None`
    /// (the saved checkpoint represents a terminated graph), the
    /// (possibly updated) state is returned directly without
    /// re-running anything.
    pub async fn resume_with(&self, command: Command<S>, ctx: &ExecutionContext) -> Result<S> {
        let checkpointer = self
            .checkpointer
            .as_ref()
            .ok_or_else(|| Error::config("CompiledGraph::resume requires a Checkpointer"))?;
        let key = ThreadKey::from_ctx(ctx)?;
        let latest = checkpointer.latest(&key).await?.ok_or_else(|| {
            Error::invalid_request(format!(
                "CompiledGraph::resume: no checkpoint exists for tenant '{}' thread '{}'",
                key.tenant_id(),
                key.thread_id()
            ))
        })?;
        self.dispatch_from_checkpoint(latest, command, ctx).await
    }

    /// Time-travel resume: continue execution from a named
    /// checkpoint anywhere in history, applying the supplied
    /// [`Command`]. Combine with [`Checkpointer::update_state`] to
    /// branch off a historical state.
    pub async fn resume_from(
        &self,
        checkpoint_id: &CheckpointId,
        command: Command<S>,
        ctx: &ExecutionContext,
    ) -> Result<S> {
        let checkpointer = self
            .checkpointer
            .as_ref()
            .ok_or_else(|| Error::config("CompiledGraph::resume_from requires a Checkpointer"))?;
        let key = ThreadKey::from_ctx(ctx)?;
        let cp = checkpointer
            .by_id(&key, checkpoint_id)
            .await?
            .ok_or_else(|| {
                Error::invalid_request(format!(
                    "CompiledGraph::resume_from: checkpoint not found in tenant '{}' thread '{}'",
                    key.tenant_id(),
                    key.thread_id()
                ))
            })?;
        self.dispatch_from_checkpoint(cp, command, ctx).await
    }

    async fn dispatch_from_checkpoint(
        &self,
        checkpoint: Checkpoint<S>,
        command: Command<S>,
        ctx: &ExecutionContext,
    ) -> Result<S> {
        // Invariant #18 — resume is auditable. Operators replaying a
        // session log can distinguish a fresh run from a resumption,
        // and which checkpoint the resume lifted state from.
        if let Some(handle) = ctx.audit_sink() {
            handle
                .as_sink()
                .record_resumed(&checkpoint.id.to_hyphenated_string());
        }
        // `Command::ApproveTool` requires attaching the operator's
        // decision to a fresh ctx via `PendingApprovalDecisions`
        // before re-dispatch. The other variants modify state /
        // next-node directly.
        let mut scoped_ctx: Option<ExecutionContext> = None;
        let (state, next_node) = match command {
            Command::Resume => (checkpoint.state, checkpoint.next_node),
            Command::Update(s) => (s, checkpoint.next_node),
            Command::GoTo(node) => (checkpoint.state, Some(node)),
            Command::ApproveTool {
                tool_use_id,
                decision,
            } => {
                if matches!(decision, entelix_core::ApprovalDecision::AwaitExternal) {
                    return Err(Error::invalid_request(
                        "Command::ApproveTool: AwaitExternal is not a valid resume \
                         decision — pausing again on resume defeats the purpose. \
                         Supply Approve or Reject{reason}.",
                    ));
                }
                let mut pending = ctx
                    .extension::<entelix_core::PendingApprovalDecisions>()
                    .map(|h| (*h).clone())
                    .unwrap_or_default();
                pending.insert(tool_use_id, decision);
                scoped_ctx = Some(ctx.clone().add_extension(pending));
                (checkpoint.state, checkpoint.next_node)
            }
        };
        let effective_ctx = scoped_ctx.as_ref().unwrap_or(ctx);
        match next_node {
            None => Ok(state),
            Some(next) => {
                // The first node visited during a resume must not
                // re-trigger its own `interrupt_before` pause point —
                // otherwise resume from an `interrupt_before` pause
                // would deadlock by pausing again on the same node.
                self.execute_loop_inner(
                    state,
                    next,
                    checkpoint.step.saturating_add(1),
                    effective_ctx,
                    true,
                )
                .await
            }
        }
    }

    async fn execute_loop(
        &self,
        state: S,
        current: String,
        step_offset: usize,
        ctx: &ExecutionContext,
    ) -> Result<S> {
        self.execute_loop_inner(state, current, step_offset, ctx, false)
            .await
    }

    #[allow(clippy::too_many_lines)]
    async fn execute_loop_inner(
        &self,
        mut state: S,
        mut current: String,
        step_offset: usize,
        ctx: &ExecutionContext,
        mut skip_interrupt_before_for_current: bool,
    ) -> Result<S> {
        // Per-invocation step counter — `recursion_limit` caps cycles
        // within *this* call. `step_offset` carries the prior thread-wide
        // step count forward so checkpoint history stays monotonic.
        // Operators may *lower* (never raise) the cap per-call by
        // attaching `RunOverrides::with_max_iterations(n)` to the
        // ExecutionContext; the compile-time cap stays authoritative.
        let effective_recursion_limit = ctx
            .extension::<entelix_core::RunOverrides>()
            .and_then(|o| o.max_iterations())
            .map_or(self.recursion_limit, |n| n.min(self.recursion_limit));
        let mut steps_in_call: usize = 0;
        loop {
            if ctx.is_cancelled() {
                return Err(Error::Cancelled);
            }
            if steps_in_call >= effective_recursion_limit {
                return Err(Error::invalid_request(format!(
                    "StateGraph: recursion limit ({effective_recursion_limit}) exceeded — possible infinite cycle"
                )));
            }
            steps_in_call = steps_in_call.saturating_add(1);
            let total_step = step_offset.saturating_add(steps_in_call);

            let node = self.nodes.get(&current).ok_or_else(|| {
                Error::invalid_request(format!(
                    "StateGraph: control reached unknown node '{current}'"
                ))
            })?;

            // Snapshot pre-node state in case the node interrupts — only
            // when we actually have somewhere to persist it.
            let pre_state = if self.checkpointer.is_some() && ctx.thread_id().is_some() {
                Some(state.clone())
            } else {
                None
            };

            // interrupt_before pause point — raise before invoking
            // the node so resume re-runs it from the saved pre-state.
            // The `skip_interrupt_before_for_current` flag lets a
            // resume bypass the check on the first node it visits;
            // otherwise resume from an `interrupt_before` pause would
            // deadlock on the same node forever.
            if self.interrupt_before.contains(&current) && !skip_interrupt_before_for_current {
                if let (Some(cp), Some(thread_id), Some(pre)) =
                    (&self.checkpointer, ctx.thread_id(), pre_state.clone())
                {
                    let key = ThreadKey::new(ctx.tenant_id().clone(), thread_id);
                    cp.put(Checkpoint::new(
                        &key,
                        total_step,
                        pre,
                        Some(current.clone()),
                    ))
                    .await?;
                }
                return Err(Error::Interrupted {
                    payload: serde_json::json!({"kind": "before", "node": current}),
                });
            }
            // Subsequent iterations always honour `interrupt_before`.
            skip_interrupt_before_for_current = false;

            match node.invoke(state, ctx).await {
                Ok(new_state) => state = new_state,
                Err(Error::Interrupted { payload }) => {
                    // Persist a checkpoint with PRE-node state so resume re-
                    // runs the interrupted node (typically with updated
                    // state injected via `Command::Update`).
                    if let (Some(cp), Some(thread_id), Some(pre)) =
                        (&self.checkpointer, ctx.thread_id(), pre_state)
                    {
                        let key = ThreadKey::new(ctx.tenant_id().clone(), thread_id);
                        cp.put(Checkpoint::new(
                            &key,
                            total_step,
                            pre,
                            Some(current.clone()),
                        ))
                        .await?;
                    }
                    return Err(Error::Interrupted { payload });
                }
                Err(other) => return Err(other),
            }

            // interrupt_after pause point — raise after the node
            // returns Ok so resume continues forward, skipping a
            // re-run of the just-completed node. The checkpoint
            // carries the post-node state and points at the
            // resolved next node (or `None` for terminal nodes).
            if self.interrupt_after.contains(&current) && !self.send_edges.contains_key(&current) {
                let next_node = self.resolve_next_node(&current, &state)?;
                if let (Some(cp), Some(thread_id)) = (&self.checkpointer, ctx.thread_id()) {
                    let key = ThreadKey::new(ctx.tenant_id().clone(), thread_id);
                    cp.put(Checkpoint::new(
                        &key,
                        total_step,
                        state.clone(),
                        next_node.clone(),
                    ))
                    .await?;
                }
                return Err(Error::Interrupted {
                    payload: serde_json::json!({"kind": "after", "node": current}),
                });
            }

            // Send-edge fan-out — runs entirely between this loop
            // iteration and the next. The branches execute in
            // parallel (fail-fast), per-branch states fold via the
            // reducer, then control jumps to the join target. From
            // the rest of the loop's perspective the fan-out is
            // atomic: one step counts toward `recursion_limit`,
            // checkpointing happens after the fold, and the next
            // iteration starts on the join node with the merged
            // state.
            if let Some(send) = self.send_edges.get(&current) {
                state = self.execute_send_edge(send, state, ctx).await?;
                if send.join == END {
                    self.emit_depth_histogram(steps_in_call, ctx);
                    return Ok(state);
                }
                current = send.join.clone();
                continue;
            }

            // Determine the next node (or terminal) before persisting, so
            // resume knows what to do.
            let next_node = self.resolve_next_node(&current, &state)?;

            // Persist a checkpoint when a checkpointer is attached, a
            // thread_id is in scope, AND the configured granularity
            // requests a per-node write. Off skips writes entirely
            // even with a checkpointer bound (useful for ephemeral
            // graphs that share a backend with persistent ones).
            // tenant_id always present.
            let granularity_writes = matches!(
                self.checkpoint_granularity,
                crate::state_graph::CheckpointGranularity::PerNode
            );
            if granularity_writes
                && let (Some(cp), Some(thread_id)) = (&self.checkpointer, ctx.thread_id())
            {
                let key = ThreadKey::new(ctx.tenant_id().clone(), thread_id);
                cp.put(Checkpoint::new(
                    &key,
                    total_step,
                    state.clone(),
                    next_node.clone(),
                ))
                .await?;
            }

            match next_node {
                None => {
                    self.emit_depth_histogram(steps_in_call, ctx);
                    return Ok(state);
                }
                Some(next) => current = next,
            }
        }
    }

    /// Execute one send-edge fan-out: dispatch every branch in
    /// parallel, fail-fast on any branch error, fold post-states
    /// via the configured reducer.
    ///
    /// Returns the reduced state. Caller is responsible for
    /// continuing to the join target.
    async fn execute_send_edge(
        &self,
        send: &SendEdge<S>,
        state: S,
        ctx: &ExecutionContext,
    ) -> Result<S> {
        let branches = (send.selector)(&state);
        if branches.is_empty() {
            return Ok(state);
        }
        // Validate every target up front against the statically-
        // declared `targets` set. Fail with a clear error before
        // scheduling any work — typos in the selector closure are
        // the most common bug here.
        for (target, _) in &branches {
            if !send.has_target(target) {
                return Err(Error::invalid_request(format!(
                    "StateGraph: send edge dispatched to '{target}' which is not in the \
                     declared target set {:?}",
                    send.targets()
                )));
            }
            // Defensive: declared targets are validated at compile,
            // but a CompiledGraph constructed by hand outside the
            // builder might miss the check.
            if !self.nodes.contains_key(target) {
                return Err(Error::invalid_request(format!(
                    "StateGraph: send edge dispatched to unknown node '{target}'"
                )));
            }
        }
        // Schedule branches concurrently. `try_join_all` short-
        // circuits on the first error, dropping the still-running
        // futures cooperatively (they observe `ctx.cancellation()`
        // via the shared scope token if they bother to check).
        let scope_ctx = ctx.child();
        let futures = branches
            .into_iter()
            .map(|(target, branch_state)| {
                let node = self.nodes.get(&target).map(Arc::clone).ok_or_else(|| {
                    Error::invalid_request(format!(
                        "StateGraph: send edge dispatched to unknown node '{target}'"
                    ))
                })?;
                let scope_ctx = scope_ctx.clone();
                Ok::<_, Error>(async move { node.invoke(branch_state, &scope_ctx).await })
            })
            .collect::<Result<Vec<_>>>()?;
        let branch_states = futures::future::try_join_all(futures).await?;
        // Fold: start with the pre-fan-out state, merge each
        // branch result into it via `S::merge` (wrapped in
        // `SendMerger` at `add_send_edges` time). The merge
        // contract is associative-with-this-seed enough that
        // pre-state + branches captures the intuitive "merge
        // results back into the parent" shape across every
        // per-field reducer the state struct declares.
        let mut folded = state;
        for branch in branch_states {
            folded = (send.merger)(folded, branch);
        }
        Ok(folded)
    }

    /// Resolve the next node from the current node + state, applying
    /// finish-point check, conditional-edge selector, then static
    /// edge in that order. `None` means the run is terminal at
    /// `current`. Shared by the loop and stream paths so the
    /// branching logic stays in one place.
    fn resolve_next_node(&self, current: &str, state: &S) -> Result<Option<String>> {
        if self.finish_points.contains(current) {
            return Ok(None);
        }
        if let Some(cond) = self.conditional_edges.get(current) {
            let key = (cond.selector)(state);
            let target = cond.mapping.get(&key).ok_or_else(|| {
                Error::invalid_request(format!(
                    "StateGraph: conditional edge from '{current}' returned key '{key}' \
                     which is not present in the mapping"
                ))
            })?;
            return Ok(if target == END {
                None
            } else {
                Some(target.clone())
            });
        }
        let target = self.edges.get(current).ok_or_else(|| {
            Error::invalid_request(format!(
                "StateGraph: node '{current}' has no outgoing edge and is not terminal"
            ))
        })?;
        Ok(Some(target.clone()))
    }

    /// Surface the per-call recursion depth so OTel subscribers can
    /// build a histogram of typical graph depths and detect
    /// anomalies — sudden jumps near `recursion_limit` flag
    /// potential infinite-loop cycles long before the limit fires.
    /// Mirrored on the stream path so both surfaces emit the same
    /// signal.
    fn emit_depth_histogram(&self, depth: usize, ctx: &ExecutionContext) {
        tracing::event!(
            target: "entelix.graph",
            tracing::Level::DEBUG,
            entelix.graph.depth = depth,
            entelix.graph.recursion_limit = self.recursion_limit,
            entelix.tenant_id = ctx.tenant_id().as_str(),
            entelix.thread_id = ctx.thread_id(),
            entelix.run_id = ctx.run_id(),
            "entelix.graph.run_complete"
        );
    }
}

#[async_trait::async_trait]
impl<S> Runnable<S, S> for CompiledGraph<S>
where
    S: Clone + Send + Sync + 'static,
{
    async fn invoke(&self, input: S, ctx: &ExecutionContext) -> Result<S> {
        self.execute_loop(input, self.entry_point.clone(), 0, ctx)
            .await
    }

    async fn stream(
        &self,
        input: S,
        mode: StreamMode,
        ctx: &ExecutionContext,
    ) -> Result<BoxStream<'_, Result<StreamChunk<S>>>> {
        Ok(Box::pin(self.build_stream(input, mode, ctx.clone())))
    }
}

const GRAPH_STREAM_NAME: &str = "CompiledGraph";

fn finished<S>(ok: bool) -> StreamChunk<S> {
    StreamChunk::Event(RunnableEvent::Finished {
        name: GRAPH_STREAM_NAME.to_owned(),
        ok,
    })
}

impl<S> CompiledGraph<S>
where
    S: Clone + Send + Sync + 'static,
{
    #[allow(
        clippy::too_many_lines,
        clippy::single_match_else,
        clippy::manual_let_else,
        tail_expr_drop_order
    )]
    fn build_stream(
        &self,
        input: S,
        mode: StreamMode,
        ctx: ExecutionContext,
    ) -> impl futures::Stream<Item = Result<StreamChunk<S>>> + Send + '_ {
        let entry = self.entry_point.clone();
        // Carry the (tenant, thread, mode) tuple into the early-cancel
        // observability hook. `FinalizingStream` only fires the
        // closure when the consumer drops the stream before it
        // signals completion — normal end-of-graph paths are silent.
        let finalize_tenant = ctx.tenant_id().clone();
        let finalize_thread = ctx.thread_id().map(str::to_owned);
        let finalize_mode = mode;
        let effective_recursion_limit = ctx
            .extension::<entelix_core::RunOverrides>()
            .and_then(|o| o.max_iterations())
            .map_or(self.recursion_limit, |n| n.min(self.recursion_limit));
        let inner = async_stream::stream! {
            let mut state = input;
            let mut current = entry;
            let mut steps_in_call: usize = 0;

            if matches!(mode, StreamMode::Events) {
                yield Ok(StreamChunk::Event(RunnableEvent::Started {
                    name: GRAPH_STREAM_NAME.to_owned(),
                }));
            }

            loop {
                if ctx.is_cancelled() {
                    if matches!(mode, StreamMode::Events) {
                        yield Ok(finished::<S>(false));
                    }
                    yield Err(Error::Cancelled);
                    return;
                }
                if steps_in_call >= effective_recursion_limit {
                    if matches!(mode, StreamMode::Events) {
                        yield Ok(finished::<S>(false));
                    }
                    yield Err(Error::invalid_request(format!(
                        "StateGraph: recursion limit ({effective_recursion_limit}) exceeded — possible infinite cycle"
                    )));
                    return;
                }
                steps_in_call = steps_in_call.saturating_add(1);

                if matches!(mode, StreamMode::Debug) {
                    yield Ok(StreamChunk::Debug(DebugEvent::NodeStart {
                        node: current.clone(),
                        step: steps_in_call,
                    }));
                }

                let Some(node) = self.nodes.get(&current) else {
                    yield Err(Error::invalid_request(format!(
                        "StateGraph: control reached unknown node '{current}'"
                    )));
                    return;
                };

                match node.invoke(state, &ctx).await {
                    Ok(s) => state = s,
                    Err(e) => {
                        if matches!(mode, StreamMode::Events) {
                            yield Ok(finished::<S>(false));
                        }
                        yield Err(e);
                        return;
                    }
                }

                match mode {
                    StreamMode::Values => {
                        yield Ok(StreamChunk::Value(state.clone()));
                    }
                    StreamMode::Updates => {
                        yield Ok(StreamChunk::Update {
                            node: current.clone(),
                            value: state.clone(),
                        });
                    }
                    StreamMode::Debug => {
                        yield Ok(StreamChunk::Debug(DebugEvent::NodeEnd {
                            node: current.clone(),
                            step: steps_in_call,
                        }));
                    }
                    _ => {}
                }

                // Send-edge fan-out — same atomic shape as the
                // loop path. After the merge the stream resumes
                // emission from the join node.
                if let Some(send) = self.send_edges.get(&current) {
                    match self.execute_send_edge(send, state.clone(), &ctx).await {
                        Ok(merged) => state = merged,
                        Err(e) => {
                            if matches!(mode, StreamMode::Events) {
                                yield Ok(finished::<S>(false));
                            }
                            yield Err(e);
                            return;
                        }
                    }
                    if send.join == END {
                        self.emit_depth_histogram(steps_in_call, &ctx);
                        match mode {
                            StreamMode::Debug => {
                                yield Ok(StreamChunk::Debug(DebugEvent::Final));
                            }
                            StreamMode::Events => {
                                yield Ok(finished::<S>(true));
                            }
                            StreamMode::Messages => {
                                yield Ok(StreamChunk::Value(state));
                            }
                            _ => {}
                        }
                        return;
                    }
                    current = send.join.clone();
                    continue;
                }

                let next_node = match self.resolve_next_node(&current, &state) {
                    Ok(n) => n,
                    Err(e) => {
                        if matches!(mode, StreamMode::Events) {
                            yield Ok(finished::<S>(false));
                        }
                        yield Err(e);
                        return;
                    }
                };

                if let Some(next) = next_node {
                    current = next;
                } else {
                    self.emit_depth_histogram(steps_in_call, &ctx);
                    match mode {
                        StreamMode::Debug => {
                            yield Ok(StreamChunk::Debug(DebugEvent::Final));
                        }
                        StreamMode::Events => {
                            yield Ok(finished::<S>(true));
                        }
                        StreamMode::Messages => {
                            yield Ok(StreamChunk::Value(state));
                        }
                        _ => {}
                    }
                    return;
                }
            }
        };
        FinalizingStream::new(inner, move || {
            tracing::debug!(
                target: "entelix_graph::stream",
                tenant_id = %finalize_tenant,
                thread_id = ?finalize_thread,
                mode = ?finalize_mode,
                "graph stream dropped before completion"
            );
        })
    }
}
