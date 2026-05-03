# entelix-agents

Production agent SDK on top of `entelix-runnable` + `entelix-graph`. Recipes (ReAct, Supervisor, Hierarchical, Chat) wrap a `StateGraph<S>` into a typed `Agent<S>` with execution-mode + observer + sink wiring.

## Surface

- **`Agent<S>`** + **`AgentBuilder<S>`** — runtime entity wrapping any `Runnable<S, S>` with `AgentEventSink`, optional `Approver` (HITL gate), `ExecutionMode::{Auto, Supervised}` (`Supervised` requires an `Approver` — `build` returns `Error::Config` otherwise), and `Vec<Arc<dyn AgentObserver<S>>>`. `Agent::execute` opens the `entelix.agent.run` OTel span (ADR-0057).
- **Recipes** — `create_react_agent` / `create_supervisor_agent` / `create_hierarchical_agent` / `create_chat_agent`. Each returns a ready-to-stream `Agent<StateType>` so the common patterns are one call.
- **`Subagent::from_whitelist(model, &parent_registry, &[...])` + `Subagent::from_filter(...)`** — narrows the parent `ToolRegistry` through `restricted_to` / `filter`. Layer factory rides over by `Arc` (ADR-0035, invariant 7). `into_tool` exposes the sub-agent as a parent-side `Tool`.
- **`AgentEventSink` trait** + `BroadcastSink` / `CaptureSink` / `ChannelSink` / `DroppingSink` — fan-out for `AgentEvent<S>`.
- **`AgentObserver` trait** + `DynObserver` adapter — `pre_turn` / `on_complete` lifecycle hooks (ctx-last per naming taxonomy).
- **`Approver` trait** + `AlwaysApprove` / `ChannelApprover` — `decide(&request, ctx)` for HITL gating.
- **`SupervisorDecision { Agent(String), Finish }`** — replaces the prior `String` + `SUPERVISOR_FINISH` sentinel pairing (ADR-0034, invariant 17). `team_from_supervisor` builds a hierarchical fan-out.

## Crate-local rules

- **`Subagent::from_whitelist` strict, `from_filter` graceful** — typo in whitelist returns `Error::Config` at construction; predicate filter accepts empty result (pure-orchestration shape). The asymmetry is regression-locked (ADR-0035).
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
