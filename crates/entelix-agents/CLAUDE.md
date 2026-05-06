# entelix-agents

Production agent SDK on top of `entelix-runnable` + `entelix-graph`. Recipes (ReAct, Supervisor, Hierarchical, Chat) wrap a `StateGraph<S>` into a typed `Agent<S>` with execution-mode + observer + sink wiring.

## Surface

- **`Agent<S>`** + **`AgentBuilder<S>`** — runtime entity wrapping any `Runnable<S, S>` with `AgentEventSink`, optional `Approver` (HITL gate), `ExecutionMode::{Auto, Supervised}` (`Supervised` requires an `Approver` — `build` returns `Error::Config` otherwise), and `Vec<Arc<dyn AgentObserver<S>>>`. `Agent::execute` opens the `entelix.agent.run` OTel span (ADR-0057).
- **Recipes** — `create_react_agent` / `create_supervisor_agent` / `create_hierarchical_agent` / `create_chat_agent`. Each returns a ready-to-stream `Agent<StateType>` so the common patterns are one call.
- **`Subagent::builder(model, &parent_registry, name, description)`** — returns a `SubagentBuilder` that narrows the parent `ToolRegistry` through `restrict_to(&[…])` / `filter(predicate)` selection verbs and accepts `with_skills` / `with_sink` / `with_approver` configuration. `build()` finalises into a `Subagent` whose identity (`name()` / `description()` / `metadata()`) is inspectable before the `into_tool()` conversion exposes it as a parent-side `Tool`. Layer factory rides over by `Arc` (ADR-0035, invariant 7). ADR-0089 + ADR-0093.
- **`AgentEventSink` trait** + `BroadcastSink` / `CaptureSink` / `ChannelSink` / `DroppingSink` — fan-out for `AgentEvent<S>`.
- **`AgentObserver` trait** + `DynObserver` adapter — `pre_turn` / `on_complete` lifecycle hooks (ctx-last per naming taxonomy).
- **`Approver` trait** + `AlwaysApprove` / `ChannelApprover` — `decide(&request, ctx)` for HITL gating.
- **`SupervisorDecision { Agent(String), Finish }`** — replaces the prior `String` + `SUPERVISOR_FINISH` sentinel pairing (ADR-0034, invariant 17). `team_from_supervisor` builds a hierarchical fan-out.

## Crate-local rules

- **`SubagentBuilder::restrict_to` strict, `filter` graceful** — typo in `restrict_to` returns `Error::Config` at `build()`; predicate filter accepts empty result (pure-orchestration shape). The asymmetry is regression-locked (ADR-0035). `build()` also rejects empty `name` or `description` so operators fail-fast instead of shipping an empty tool name to the LLM (ADR-0093).
- **Sub-agent dispatch emits `record_sub_agent_invoked`** on the parent's `AuditSink` (invariant 18, ADR-0037). Sub-agents NEVER bypass the parent registry's layer stack — `scripts/check-managed-shape.sh` fails the PR.
- **Recipe routers MUST return typed `SupervisorDecision`**, not stringly-typed sentinels. Probability literals (`0.X`) in recipe routing are forbidden (`scripts/check-magic-constants.sh`, invariant 17).
- **`Agent::execute_stream`** opens its own root span; consumers compose with `tower::Layer<S>` (e.g. `OtelLayer`, `PolicyLayer`) on the underlying `ChatModel`, not on the agent itself.

## Forbidden

- A `Subagent` path that constructs a fresh `ToolRegistry::new()` instead of narrowing the parent registry — silently drops the parent's layer stack.
- A recipe that holds `ApprovalRequest` state in agent struct fields (state lives in `SessionGraph` events, invariant 1).
- Looping `while !done { … }` in recipe code instead of modelling as a `StateGraph` with conditional edge to `END` (invariant 8).

## References

- ADR-0035 — managed-agent shape enforcement (`Subagent` narrows parent registry).
- ADR-0037 — `AuditSink` typed channel (sub-agent dispatch + supervisor handoff + resume + memory recall).
- ADR-0057 — `entelix.agent.run` OTel span semantics.
- F7 mitigation — sub-agent permission narrowing.
