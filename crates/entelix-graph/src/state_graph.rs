//! `StateGraph<S>` — the builder side of the graph contract.
//!
//! Surface:
//!
//! - `add_node(name, runnable)` — node is a `Runnable<S, S>` returning the
//!   new full state.
//! - `add_edge(from, to)` — single static next-hop per node.
//! - `add_conditional_edges(from, selector, mapping)` — predicate-based
//!   dispatch. The selector takes `&S`, returns a key, and the mapping
//!   resolves it to a target node (or [`END`]).
//! - `add_send_edges(from, targets, selector, join)` — parallel
//!   fan-out. The selector returns `Vec<(target, branch_state)>`;
//!   each branch runs its target node concurrently, results fold
//!   via the state's [`StateMerge::merge`](crate::StateMerge::merge)
//!   impl into the pre-fan-out state, then control flows to `join`.
//!   The state struct supplies the merge story via
//!   `#[derive(StateMerge)]` over per-field
//!   [`Annotated<T, R>`](crate::Annotated) wrappers — adding new
//!   state fields never edits send-edge call sites.
//! - `set_entry_point(name)` — required.
//! - `add_finish_point(name)` — running this node halts and returns state.
//! - `with_recursion_limit(n)` — F6 mitigation, default 25.
//! - `compile() → CompiledGraph<S>` — preflight validation; the result
//!   implements `Runnable<S, S>` so it composes via `.pipe()` and serves
//!   as a sub-graph node in a larger `StateGraph`.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use entelix_core::{Error, Result};
use entelix_runnable::Runnable;

use crate::checkpoint::Checkpointer;
use crate::compiled::{
    CompiledGraph, ConditionalEdge, EdgeSelector, SendEdge, SendMerger, SendSelector,
};
use crate::contributing_node::ContributingNodeAdapter;
use crate::merge_node::MergeNodeAdapter;
use crate::reducer::StateMerge;

/// Default cap on node executions per `invoke` (F6 mitigation — guards
/// against infinite cycles).
pub const DEFAULT_RECURSION_LIMIT: usize = 25;

/// Sentinel target meaning "terminate without running another node". Use
/// in `add_conditional_edges` mapping when a branch should end the graph.
pub const END: &str = "__entelix_graph_end__";

/// How often the compiled graph writes a checkpoint when a
/// `Checkpointer` is attached.
///
/// `PerNode` (the default) writes after every successful node
/// completion — durable enough that a crash mid-graph loses at most
/// one node's work. `Off` skips checkpointer writes entirely; the
/// graph still runs end-to-end but cannot resume after a crash.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum CheckpointGranularity {
    /// Skip checkpointer writes. Useful for ephemeral graphs or
    /// when the checkpointer is attached purely to satisfy a
    /// downstream API contract.
    Off,
    /// Write a checkpoint after each node completes successfully.
    /// This is the default and matches the F8 mitigation.
    #[default]
    PerNode,
}

/// Builder for a state-machine graph parameterised over its state type `S`.
///
/// Nodes are `Runnable<S, S>` instances; each one consumes the current state
/// and returns the new full state. Three node-registration shapes coexist:
///
/// - [`Self::add_node`] — full-state replace. The node owns the
///   entire shape and returns the next state.
/// - [`Self::add_node_with`] — delta + bespoke merger closure.
///   Best when the merge logic is graph-specific.
/// - [`Self::add_contributing_node`] — declarative per-field merge
///   via the [`StateMerge`] trait. The state struct advertises its
///   merge story (typically through `#[derive(StateMerge)]` and
///   per-field [`Annotated<T, R>`](crate::Annotated) wrappers); the
///   node returns an `Option`-wrapped `S::Contribution` naming
///   exactly the slots it touched. Slots left as `None` keep the
///   current value; slots set to `Some` merge through the
///   per-field reducer.
pub struct StateGraph<S>
where
    S: Clone + Send + Sync + 'static,
{
    nodes: HashMap<String, Arc<dyn Runnable<S, S>>>,
    edges: HashMap<String, String>,
    conditional_edges: HashMap<String, ConditionalEdge<S>>,
    send_edges: HashMap<String, SendEdge<S>>,
    entry_point: Option<String>,
    finish_points: HashSet<String>,
    recursion_limit: usize,
    checkpointer: Option<Arc<dyn Checkpointer<S>>>,
    checkpoint_granularity: CheckpointGranularity,
    /// Nodes whose pre-execution position is a HITL pause point —
    /// the runtime raises `Error::Interrupted` *before* invoking
    /// the node, persists a checkpoint pointing back at the same
    /// node, and lets the host application resume via
    /// `Command::Resume` (re-runs the node) or `Command::Update`
    /// (re-runs with new state).
    interrupt_before: HashSet<String>,
    /// Nodes whose post-execution position is a HITL pause point —
    /// the runtime raises `Error::Interrupted` *after* the node
    /// completes successfully, persists a checkpoint with the new
    /// state pointing at the resolved next node, and lets the host
    /// application resume forward (skipping the just-run node).
    interrupt_after: HashSet<String>,
}

impl<S> std::fmt::Debug for StateGraph<S>
where
    S: Clone + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Deterministic Debug output — see `CompiledGraph::fmt`
        // for the rationale: HashMap / HashSet iteration order is
        // unspecified, so sorted projections keep test snapshots
        // and operator log diffs stable.
        let mut nodes: Vec<&String> = self.nodes.keys().collect();
        nodes.sort();
        let mut edges: Vec<(&String, &String)> = self.edges.iter().collect();
        edges.sort_by_key(|(k, _)| k.as_str());
        let mut conditional: Vec<&String> = self.conditional_edges.keys().collect();
        conditional.sort();
        let mut send: Vec<&String> = self.send_edges.keys().collect();
        send.sort();
        let mut finish: Vec<&String> = self.finish_points.iter().collect();
        finish.sort();
        let mut interrupt_before: Vec<&String> = self.interrupt_before.iter().collect();
        interrupt_before.sort();
        let mut interrupt_after: Vec<&String> = self.interrupt_after.iter().collect();
        interrupt_after.sort();
        f.debug_struct("StateGraph")
            .field("nodes", &nodes)
            .field("edges", &edges)
            .field("conditional_edges", &conditional)
            .field("send_edges", &send)
            .field("entry_point", &self.entry_point)
            .field("finish_points", &finish)
            .field("recursion_limit", &self.recursion_limit)
            .field("has_checkpointer", &self.checkpointer.is_some())
            .field("checkpoint_granularity", &self.checkpoint_granularity)
            .field("interrupt_before", &interrupt_before)
            .field("interrupt_after", &interrupt_after)
            .finish()
    }
}

impl<S> StateGraph<S>
where
    S: Clone + Send + Sync + 'static,
{
    /// Empty graph.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            conditional_edges: HashMap::new(),
            send_edges: HashMap::new(),
            entry_point: None,
            finish_points: HashSet::new(),
            recursion_limit: DEFAULT_RECURSION_LIMIT,
            checkpointer: None,
            checkpoint_granularity: CheckpointGranularity::default(),
            interrupt_before: HashSet::new(),
            interrupt_after: HashSet::new(),
        }
    }

    /// Attach a checkpointer.
    ///
    /// When set, the compiled graph writes a checkpoint after every node
    /// invocation if the executing `ExecutionContext` carries a
    /// `thread_id`. Use [`CompiledGraph::resume`] to continue from the
    /// most recent checkpoint after a crash. Tune the write frequency
    /// via [`Self::with_checkpoint_granularity`].
    #[must_use]
    pub fn with_checkpointer(mut self, checkpointer: Arc<dyn Checkpointer<S>>) -> Self {
        self.checkpointer = Some(checkpointer);
        self
    }

    /// Override how often the compiled graph writes a checkpoint
    /// when a checkpointer is attached. Defaults to
    /// [`CheckpointGranularity::PerNode`].
    #[must_use]
    pub const fn with_checkpoint_granularity(mut self, g: CheckpointGranularity) -> Self {
        self.checkpoint_granularity = g;
        self
    }

    /// Register a node. A second registration with the same name replaces
    /// the prior runnable (calls during construction are append-or-replace,
    /// not append-only).
    #[must_use]
    pub fn add_node<R>(mut self, name: impl Into<String>, runnable: R) -> Self
    where
        R: Runnable<S, S> + 'static,
    {
        self.nodes.insert(name.into(), Arc::new(runnable));
        self
    }

    /// Register a *delta-style* node. The inner runnable produces an
    /// update of arbitrary type `U`; the merger combines it with the
    /// inbound state into a fresh full state.
    ///
    /// Use this when the natural shape of a node is "compute and
    /// return what changed" rather than "thread the entire state
    /// through". The merger has access to both the snapshot of the
    /// inbound state and the delta, so per-field
    /// [`Reducer<T>`](crate::Reducer) calls (`Append`, `MergeMap`,
    /// `Max`, …) plug in directly:
    ///
    /// ```ignore
    /// graph.add_node_with(
    ///     "plan",
    ///     planner_runnable,
    ///     |mut state: PlanState, update: PlannerOutput| {
    ///         state.log = Append::<String>::new()
    ///             .reduce(state.log, update.new_log_entries);
    ///         state.iterations += 1;
    ///         Ok(state)
    ///     },
    /// );
    /// ```
    ///
    /// Existing [`Self::add_node`] (full-state replace) keeps working
    /// unchanged — the two patterns coexist node-by-node.
    #[must_use]
    pub fn add_node_with<R, U, F>(self, name: impl Into<String>, runnable: R, merger: F) -> Self
    where
        R: Runnable<S, U> + 'static,
        U: Send + Sync + 'static,
        F: Fn(S, U) -> Result<S> + Send + Sync + 'static,
    {
        self.add_node(name, MergeNodeAdapter::new(runnable, merger))
    }

    /// Register a *contribution-style* node whose output names
    /// exactly the slots it touched, folded into the current state
    /// through [`StateMerge::merge_contribution`]. The inner
    /// runnable returns `S::Contribution` — an `Option`-wrapped
    /// shape mirroring LangGraph's TypedDict partial-return:
    /// `None` slots keep the current value, `Some` slots merge
    /// through the per-field reducer.
    ///
    /// Use this when the state type owns its merge story
    /// declaratively (via `#[derive(StateMerge)]` over fields wrapped
    /// in [`Annotated<T, R>`](crate::Annotated)). Adding a new
    /// state field never edits the graph builder — the per-field
    /// reducer annotation does the work.
    ///
    /// ```ignore
    /// use entelix_graph::{Annotated, Append, Max, StateGraph, StateMerge};
    /// use entelix_runnable::RunnableLambda;
    ///
    /// #[derive(Clone, Default, StateMerge)]
    /// struct AgentState {
    ///     log: Annotated<Vec<String>, Append<String>>,
    ///     score: Annotated<i32, Max<i32>>,
    ///     last_message: String,
    /// }
    ///
    /// // Node writes only `log` and `last_message`; `score`
    /// // stays at whatever the upstream produced (the contribution
    /// // shape carries `None` for it, which means "I didn't touch this").
    /// let planner = RunnableLambda::new(|_state: AgentState, _ctx| async {
    ///     Ok(AgentStateContribution::default()
    ///         .with_log(vec!["planned".into()])
    ///         .with_last_message("scheduled".into()))
    /// });
    /// let graph = StateGraph::<AgentState>::new()
    ///     .add_contributing_node("planner", planner);
    /// ```
    #[must_use]
    pub fn add_contributing_node<R>(self, name: impl Into<String>, runnable: R) -> Self
    where
        R: Runnable<S, S::Contribution> + 'static,
        S: StateMerge,
    {
        self.add_node(name, ContributingNodeAdapter::new(runnable))
    }

    /// Register a static `from → to` edge. Calling twice with the same
    /// `from` replaces the previous target — single static next-hop per
    /// node.
    ///
    /// A node may not have both a static edge and a conditional edge; the
    /// `compile()` step rejects that combination.
    #[must_use]
    pub fn add_edge(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.edges.insert(from.into(), to.into());
        self
    }

    /// Register a conditional dispatch: after `from` runs, evaluate
    /// `selector(&state)` and route to the node named by
    /// `mapping[selector_output]`. Mapping targets may be node names or
    /// [`END`].
    ///
    /// A second call with the same `from` replaces the prior conditional.
    /// Mixing with `add_edge` on the same `from` is rejected at compile
    /// time.
    #[must_use]
    pub fn add_conditional_edges<F, K, V>(
        mut self,
        from: impl Into<String>,
        selector: F,
        mapping: impl IntoIterator<Item = (K, V)>,
    ) -> Self
    where
        F: Fn(&S) -> String + Send + Sync + 'static,
        K: Into<String>,
        V: Into<String>,
    {
        let mapping: HashMap<String, String> = mapping
            .into_iter()
            .map(|(k, v)| (k.into(), v.into()))
            .collect();
        let edge = ConditionalEdge {
            selector: Arc::new(selector) as EdgeSelector<S>,
            mapping,
        };
        self.conditional_edges.insert(from.into(), edge);
        self
    }

    /// Register a parallel fan-out from `from`.
    ///
    /// `targets` lists every node the selector may dispatch to —
    /// statically declared so `compile()` can validate each name
    /// resolves to a registered node and so leaf-validation knows
    /// these nodes have a defined control path (the fan-out merges
    /// their results back into the join node, no per-branch
    /// outgoing edge is required).
    ///
    /// After `from` runs, the runtime evaluates `selector(&state)`
    /// to obtain a list of `(target_node, branch_state)` pairs.
    /// Each branch is invoked in parallel; the resulting per-branch
    /// states fold via `reducer` into a single `S`. Control then
    /// flows to the `join` node, which sees the reduced state.
    ///
    /// Selector outputs that name a node not in `targets` cause a
    /// runtime [`Error::InvalidRequest`] — typo-resistant by
    /// construction.
    ///
    /// Mutually exclusive with [`Self::add_edge`] /
    /// [`Self::add_conditional_edges`] on the same `from` — `compile`
    /// rejects the combination. The join target must be registered
    /// or [`END`].
    #[must_use]
    pub fn add_send_edges<F, I, T>(
        mut self,
        from: impl Into<String>,
        targets: I,
        selector: F,
        join: impl Into<String>,
    ) -> Self
    where
        F: Fn(&S) -> Vec<(String, S)> + Send + Sync + 'static,
        I: IntoIterator<Item = T>,
        T: Into<String>,
        S: StateMerge,
    {
        let edge = SendEdge::new(
            targets.into_iter().map(Into::into),
            Arc::new(selector) as SendSelector<S>,
            Arc::new(<S as StateMerge>::merge) as SendMerger<S>,
            join.into(),
        );
        self.send_edges.insert(from.into(), edge);
        self
    }

    /// Mark the start node. Required at compile time.
    #[must_use]
    pub fn set_entry_point(mut self, name: impl Into<String>) -> Self {
        self.entry_point = Some(name.into());
        self
    }

    /// Mark a node as terminal — running it halts the graph and returns
    /// the post-node state. A graph may have more than one finish point;
    /// any path that reaches one terminates.
    #[must_use]
    pub fn add_finish_point(mut self, name: impl Into<String>) -> Self {
        self.finish_points.insert(name.into());
        self
    }

    /// Override the per-invocation recursion limit (F6 mitigation).
    #[must_use]
    pub const fn with_recursion_limit(mut self, n: usize) -> Self {
        self.recursion_limit = n;
        self
    }

    /// Mark `nodes` as HITL pause points evaluated **before** the
    /// node runs. When control reaches a marked node the runtime
    /// raises `Error::Interrupted` with
    /// `kind: InterruptionKind::ScheduledPause { phase: Before, node }`
    /// (the `payload` is `Value::Null` — every distinguishing
    /// detail is on the typed kind) and (when a `Checkpointer` is
    /// attached) persists a checkpoint pointing back at the same
    /// node.
    ///
    /// Resume via the existing `Command` machinery:
    /// - `Command::Resume` re-runs the marked node from the saved
    ///   pre-state.
    /// - `Command::Update(s)` re-runs the marked node from `s`.
    /// - `Command::GoTo(other)` jumps to `other` instead.
    ///
    /// Calling twice unions the supplied node sets.
    #[must_use]
    pub fn interrupt_before<I, T>(mut self, nodes: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        self.interrupt_before
            .extend(nodes.into_iter().map(Into::into));
        self
    }

    /// Mark `nodes` as HITL pause points evaluated **after** the
    /// node completes successfully. When such a node returns Ok
    /// the runtime raises `Error::Interrupted` with
    /// `kind: InterruptionKind::ScheduledPause { phase: After, node }`
    /// and persists a checkpoint with the post-node state pointing
    /// at the resolved next node — `Command::Resume` then continues
    /// forward, skipping a re-run of the just-completed node.
    #[must_use]
    pub fn interrupt_after<I, T>(mut self, nodes: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        self.interrupt_after
            .extend(nodes.into_iter().map(Into::into));
        self
    }

    /// Number of registered nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of registered static edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Number of nodes with a conditional dispatch.
    pub fn conditional_edge_count(&self) -> usize {
        self.conditional_edges.len()
    }

    /// Validate and freeze the graph.
    ///
    /// Returns `Err(Error::Config(_))` for:
    /// - Missing entry point.
    /// - Entry point referencing an unregistered node.
    /// - Static edge referencing an unregistered `from` or `to`.
    /// - Conditional edge `from` not registered, or any mapping target
    ///   that is neither a registered node nor [`END`].
    /// - A node with both a static edge AND a conditional edge.
    /// - No finish points registered.
    /// - Finish point referencing an unregistered node.
    /// - A non-finish-point node has no outgoing edge (static or
    ///   conditional).
    /// - `interrupt_before` / `interrupt_after` referencing a node
    ///   that is not registered.
    pub fn compile(self) -> Result<CompiledGraph<S>> {
        let entry = self
            .entry_point
            .as_ref()
            .ok_or_else(|| Error::config("StateGraph: no entry point set"))?
            .clone();
        if !self.nodes.contains_key(&entry) {
            return Err(Error::config(format!(
                "StateGraph: entry point '{entry}' is not a registered node"
            )));
        }
        self.validate_finish_points()?;
        self.validate_static_edges()?;
        self.validate_conditional_edges()?;
        let send_branch_targets = self.validate_send_edges()?;
        self.validate_node_termination(&send_branch_targets)?;
        self.validate_interrupt_points()?;

        Ok(CompiledGraph::new(
            self.nodes,
            self.edges,
            self.conditional_edges,
            self.send_edges,
            entry,
            self.finish_points,
            self.recursion_limit,
            self.checkpointer,
            self.checkpoint_granularity,
            self.interrupt_before,
            self.interrupt_after,
        ))
    }

    /// Every name in `interrupt_before` / `interrupt_after` must
    /// resolve to a registered node — typo-resistant by
    /// construction.
    fn validate_interrupt_points(&self) -> Result<()> {
        for name in &self.interrupt_before {
            if !self.nodes.contains_key(name) {
                return Err(Error::config(format!(
                    "StateGraph: interrupt_before names '{name}' which is not a registered node"
                )));
            }
        }
        for name in &self.interrupt_after {
            if !self.nodes.contains_key(name) {
                return Err(Error::config(format!(
                    "StateGraph: interrupt_after names '{name}' which is not a registered node"
                )));
            }
        }
        Ok(())
    }

    /// Validate finish-point set: at least one, every entry must
    /// resolve to a registered node.
    fn validate_finish_points(&self) -> Result<()> {
        if self.finish_points.is_empty() {
            return Err(Error::config(
                "StateGraph: no finish points registered (graph would never terminate)",
            ));
        }
        for fp in &self.finish_points {
            if !self.nodes.contains_key(fp) {
                return Err(Error::config(format!(
                    "StateGraph: finish point '{fp}' is not a registered node"
                )));
            }
        }
        Ok(())
    }

    /// Validate static `from → to` edges.
    fn validate_static_edges(&self) -> Result<()> {
        for (from, to) in &self.edges {
            if !self.nodes.contains_key(from) {
                return Err(Error::config(format!(
                    "StateGraph: edge source '{from}' is not a registered node"
                )));
            }
            if !self.nodes.contains_key(to) {
                return Err(Error::config(format!(
                    "StateGraph: edge target '{to}' is not a registered node"
                )));
            }
        }
        Ok(())
    }

    /// Validate conditional-edge dispatch tables.
    fn validate_conditional_edges(&self) -> Result<()> {
        for (from, cond) in &self.conditional_edges {
            if !self.nodes.contains_key(from) {
                return Err(Error::config(format!(
                    "StateGraph: conditional edge source '{from}' is not a registered node"
                )));
            }
            if self.edges.contains_key(from) {
                return Err(Error::config(format!(
                    "StateGraph: node '{from}' has both a static edge and a conditional edge \
                     — pick one"
                )));
            }
            for target in cond.mapping.values() {
                if target != END && !self.nodes.contains_key(target) {
                    return Err(Error::config(format!(
                        "StateGraph: conditional edge from '{from}' maps to '{target}' which is \
                         neither a registered node nor END"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Validate send-edge fan-outs and return the union of
    /// statically-declared branch targets (used by leaf-validation
    /// to recognise these nodes as having a defined control path).
    fn validate_send_edges(&self) -> Result<HashSet<String>> {
        let mut send_branch_targets: HashSet<String> = HashSet::new();
        for (from, send) in &self.send_edges {
            if !self.nodes.contains_key(from) {
                return Err(Error::config(format!(
                    "StateGraph: send edge source '{from}' is not a registered node"
                )));
            }
            if self.edges.contains_key(from) || self.conditional_edges.contains_key(from) {
                return Err(Error::config(format!(
                    "StateGraph: node '{from}' has more than one outgoing edge type — \
                     send edges are mutually exclusive with static and conditional edges"
                )));
            }
            if send.join != END && !self.nodes.contains_key(&send.join) {
                return Err(Error::config(format!(
                    "StateGraph: send edge from '{from}' joins on '{}' which is \
                     neither a registered node nor END",
                    send.join
                )));
            }
            for target in send.targets() {
                if !self.nodes.contains_key(target) {
                    return Err(Error::config(format!(
                        "StateGraph: send edge from '{from}' lists target '{target}' \
                         which is not a registered node"
                    )));
                }
                send_branch_targets.insert(target.clone());
            }
        }
        Ok(send_branch_targets)
    }

    /// Every non-finish node must have a defined control-flow path:
    /// a static edge, a conditional edge, a send edge, or be the
    /// dispatch target of someone else's send edge.
    fn validate_node_termination(&self, send_branch_targets: &HashSet<String>) -> Result<()> {
        for name in self.nodes.keys() {
            if !self.finish_points.contains(name)
                && !self.edges.contains_key(name)
                && !self.conditional_edges.contains_key(name)
                && !self.send_edges.contains_key(name)
                && !send_branch_targets.contains(name)
            {
                return Err(Error::config(format!(
                    "StateGraph: node '{name}' has no outgoing edge and is not a finish point"
                )));
            }
        }
        Ok(())
    }
}

impl<S> Default for StateGraph<S>
where
    S: Clone + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}
