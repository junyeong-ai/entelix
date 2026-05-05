//! `McpManager` — per-tenant, lazily-provisioned pool of `McpClient`s.
//!
//! ## Why a manager exists
//!
//! - **F9 mitigation** — pool keyed by `(TenantId, ServerName)` so the
//!   *same* MCP server registered under two tenants opens *two*
//!   independent clients. Tenant A's bearer token is structurally
//!   unable to leak into tenant B's call because the pool entries
//!   are different `Arc<dyn McpClient>`s. (Invariant 11
//!   strengthening; see ADR-0017.)
//!
//! - **Lazy provisioning** (Anthropic managed-agent shape) — the
//!   builder records [`McpServerConfig`]s but never opens a
//!   connection. The first `list_tools` / `call_tool` for a
//!   `(tenant, server)` pair triggers `initialize`. This is critical
//!   for serverless deployments where most requests don't touch every
//!   configured MCP server.
//!
//! - **Replaceable transport** — production wires
//!   [`HttpMcpClient`](crate::HttpMcpClient); tests inject a
//!   deterministic mock by overriding the [`McpClientFactory`]. The
//!   trait stays sealed: invariant 12 forbids us from advertising a
//!   public extension point we don't have to.

// `pub(crate)` items satisfy the workspace `unreachable_pub` rust lint;
// clippy nursery's `redundant_pub_crate` disagrees and we side with the
// rust lint.
#![allow(clippy::redundant_pub_crate)]

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use std::collections::BTreeMap;

use async_trait::async_trait;
use dashmap::DashMap;
use serde_json::Value;

use entelix_core::context::ExecutionContext;

use crate::client::{HttpMcpClient, McpClient};
use crate::completion::{McpCompletionArgument, McpCompletionReference, McpCompletionResult};
use crate::error::{McpError, McpResult};
use crate::prompt::{McpPrompt, McpPromptInvocation};
use crate::resource::{McpResource, McpResourceContent};
use crate::server_config::McpServerConfig;
use crate::tool_definition::McpToolDefinition;

/// Pool key: `(tenant_id, server_name)`. The exact pair the F9
/// mitigation depends on — never collapse to either side alone.
type PoolKey = (String, String);

/// One pooled client plus its last-touched timestamp. The timestamp
/// is `seconds since pool start` so it fits in a `u64` and updates
/// can be atomic and lock-free.
struct PoolEntry {
    client: Arc<dyn McpClient>,
    last_used_secs: AtomicU64,
}

fn pool_origin() -> Instant {
    use std::sync::OnceLock;
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    *ORIGIN.get_or_init(Instant::now)
}

fn now_pool_secs() -> u64 {
    Instant::now()
        .saturating_duration_since(pool_origin())
        .as_secs()
}

/// Factory trait used internally to build new `McpClient` instances on
/// pool miss. Production uses [`HttpClientFactory`]; tests substitute
/// deterministic mocks. Sealed — not part of the public surface.
#[async_trait]
pub(crate) trait McpClientFactory: Send + Sync {
    async fn build(&self, config: &McpServerConfig) -> McpResult<Arc<dyn McpClient>>;
}

/// Default factory — produces real [`HttpMcpClient`] instances.
struct HttpClientFactory;

#[async_trait]
impl McpClientFactory for HttpClientFactory {
    async fn build(&self, config: &McpServerConfig) -> McpResult<Arc<dyn McpClient>> {
        let client = HttpMcpClient::new(config.clone())?;
        Ok(Arc::new(client))
    }
}

/// Builder for [`McpManager`]. Records configurations only — no I/O,
/// no connection attempts.
pub struct McpManagerBuilder {
    configs: Vec<McpServerConfig>,
    factory: Arc<dyn McpClientFactory>,
}

impl Default for McpManagerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl McpManagerBuilder {
    /// Fresh builder with the default HTTP factory.
    pub fn new() -> Self {
        Self {
            configs: Vec::new(),
            factory: Arc::new(HttpClientFactory),
        }
    }

    /// Register one MCP server. Names must be unique within a manager
    /// — duplicates are rejected at [`build`](Self::build) time.
    #[must_use]
    pub fn register(mut self, config: McpServerConfig) -> Self {
        self.configs.push(config);
        self
    }

    /// Internal: swap in a custom factory (used by tests).
    #[cfg(test)]
    pub(crate) fn with_factory(mut self, factory: Arc<dyn McpClientFactory>) -> Self {
        self.factory = factory;
        self
    }

    /// Finalize the manager. Validates that every server name is
    /// unique; does **not** open any connections.
    pub fn build(self) -> McpResult<McpManager> {
        let mut configs: std::collections::HashMap<String, McpServerConfig> =
            std::collections::HashMap::new();
        for cfg in self.configs {
            if configs.insert(cfg.name().to_owned(), cfg.clone()).is_some() {
                return Err(McpError::Config(format!(
                    "duplicate MCP server name '{}'",
                    cfg.name()
                )));
            }
        }
        Ok(McpManager {
            configs: Arc::new(configs),
            pool: Arc::new(DashMap::new()),
            factory: self.factory,
        })
    }
}

/// Per-tenant pool of `McpClient` instances.
///
/// Cloning is cheap (`Arc` swap). One manager per process is enough;
/// the pool is internally synchronized.
///
/// Entries are evicted only when the caller invokes
/// [`McpManager::prune_idle`] — there is no implicit background
/// task. Operators that want periodic eviction spawn one (typically
/// at the same cadence as the platform's other housekeeping
/// loops).
#[derive(Clone)]
pub struct McpManager {
    configs: Arc<std::collections::HashMap<String, McpServerConfig>>,
    pool: Arc<DashMap<PoolKey, Arc<PoolEntry>>>,
    factory: Arc<dyn McpClientFactory>,
}

#[allow(
    clippy::missing_fields_in_debug,
    reason = "factory is dyn-trait without Debug; configs/pool printed as counts"
)]
impl std::fmt::Debug for McpManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpManager")
            .field("configured_servers", &self.configs.len())
            .field("active_clients", &self.pool.len())
            .finish()
    }
}

impl McpManager {
    /// Start a new builder.
    pub fn builder() -> McpManagerBuilder {
        McpManagerBuilder::new()
    }

    /// List the tools published by `server` for the tenant in `ctx`.
    /// Triggers `initialize` on first call for this `(tenant, server)`
    /// pair; subsequent calls hit the cached tool list.
    pub async fn list_tools(
        &self,
        ctx: &ExecutionContext,
        server: &str,
    ) -> McpResult<Vec<McpToolDefinition>> {
        let client = self.client_for(ctx.tenant_id(), server).await?;
        client.initialize().await
    }

    /// Invoke `tool` on `server` with `arguments`. Auto-connects on
    /// first use.
    ///
    /// Errors from the underlying `McpClient` are wrapped with the
    /// `(server, tool)` pair so a multi-server agent run can
    /// correlate which call returned which error. The bare
    /// `McpError::JsonRpc { code, message }` would otherwise
    /// produce identical strings across N servers, making triage
    /// impossible.
    pub async fn call_tool(
        &self,
        ctx: &ExecutionContext,
        server: &str,
        tool: &str,
        arguments: Value,
    ) -> McpResult<Value> {
        let client = self.client_for(ctx.tenant_id(), server).await?;
        client
            .call_tool(tool, arguments)
            .await
            .map_err(|e| correlate(e, server, &format!("tool '{tool}'")))
    }

    /// List the resources advertised by `server` for the tenant in
    /// `ctx`. Triggers `initialize` on first call.
    pub async fn list_resources(
        &self,
        ctx: &ExecutionContext,
        server: &str,
    ) -> McpResult<Vec<McpResource>> {
        let client = self.client_for(ctx.tenant_id(), server).await?;
        client
            .list_resources()
            .await
            .map_err(|e| correlate(e, server, "resources/list"))
    }

    /// Read one resource's content blocks from `server`.
    pub async fn read_resource(
        &self,
        ctx: &ExecutionContext,
        server: &str,
        uri: &str,
    ) -> McpResult<Vec<McpResourceContent>> {
        let client = self.client_for(ctx.tenant_id(), server).await?;
        client
            .read_resource(uri)
            .await
            .map_err(|e| correlate(e, server, &format!("resource '{uri}'")))
    }

    /// List the prompts advertised by `server`.
    pub async fn list_prompts(
        &self,
        ctx: &ExecutionContext,
        server: &str,
    ) -> McpResult<Vec<McpPrompt>> {
        let client = self.client_for(ctx.tenant_id(), server).await?;
        client
            .list_prompts()
            .await
            .map_err(|e| correlate(e, server, "prompts/list"))
    }

    /// Bind `prompt` arguments and fetch the resulting transcript.
    pub async fn prompt(
        &self,
        ctx: &ExecutionContext,
        server: &str,
        prompt: &str,
        arguments: BTreeMap<String, String>,
    ) -> McpResult<McpPromptInvocation> {
        let client = self.client_for(ctx.tenant_id(), server).await?;
        client
            .prompt(prompt, arguments)
            .await
            .map_err(|e| correlate(e, server, &format!("prompt '{prompt}'")))
    }

    /// Notify `server` that this client's roots changed. Servers
    /// that opted into `roots/listChanged` will re-issue
    /// `roots/list` against the [`crate::RootsProvider`] wired
    /// through [`crate::McpServerConfig::with_roots_provider`].
    /// Stateless servers (no `Mcp-Session-Id`) accept the
    /// notification but cannot re-fetch — fire-and-forget either way.
    pub async fn notify_roots_changed(
        &self,
        ctx: &ExecutionContext,
        server: &str,
    ) -> McpResult<()> {
        let client = self.client_for(ctx.tenant_id(), server).await?;
        client
            .notify_roots_changed()
            .await
            .map_err(|e| correlate(e, server, "notifications/roots/list_changed"))
    }

    /// Ask `server` to complete a partial argument value.
    pub async fn complete(
        &self,
        ctx: &ExecutionContext,
        server: &str,
        reference: McpCompletionReference,
        argument: McpCompletionArgument,
    ) -> McpResult<McpCompletionResult> {
        let client = self.client_for(ctx.tenant_id(), server).await?;
        client
            .complete(reference, argument)
            .await
            .map_err(|e| correlate(e, server, "completion/complete"))
    }

    /// Borrow (or build) the `McpClient` for `(tenant, server)`. Pool
    /// hits are lock-free reads; pool misses serialize on the bucket
    /// shard while the new client is constructed. Every successful
    /// lookup updates the entry's last-used timestamp so
    /// [`Self::prune_idle`] can distinguish hot from cold clients.
    pub(crate) async fn client_for(
        &self,
        tenant_id: &str,
        server: &str,
    ) -> McpResult<Arc<dyn McpClient>> {
        let key = (tenant_id.to_owned(), server.to_owned());
        if let Some(existing) = self.pool.get(&key) {
            existing
                .last_used_secs
                .store(now_pool_secs(), Ordering::Relaxed);
            return Ok(existing.client.clone());
        }
        let config = self
            .configs
            .get(server)
            .ok_or_else(|| McpError::UnknownServer {
                tenant_id: tenant_id.to_owned(),
                server: server.to_owned(),
            })?;
        let client = self.factory.build(config).await?;
        let new_entry = Arc::new(PoolEntry {
            client,
            last_used_secs: AtomicU64::new(now_pool_secs()),
        });
        // `entry().or_insert_with` collapses races: if a concurrent task
        // built a client for the same key, we drop ours and adopt the
        // winner. A discarded client owns no live connection — the FSM
        // is `Queued` until first `initialize`, so this is free.
        let entry = self.pool.entry(key).or_insert(new_entry);
        entry
            .last_used_secs
            .store(now_pool_secs(), Ordering::Relaxed);
        Ok(entry.client.clone())
    }

    /// Drop pool entries whose last `client_for` call landed more
    /// than `max_idle` ago. `max_idle` of zero evicts every entry
    /// not used at the exact same second as the call. Returns the
    /// number of evicted entries. Cheap O(N) walk over the pool —
    /// operators typically run this once a minute or so.
    pub fn prune_idle(&self, max_idle: Duration) -> usize {
        let now = now_pool_secs();
        let max = max_idle.as_secs();
        let mut evicted = 0;
        self.pool.retain(|_, entry| {
            let last = entry.last_used_secs.load(Ordering::Relaxed);
            let idle = now.saturating_sub(last);
            if idle > max {
                evicted += 1;
                false
            } else {
                true
            }
        });
        evicted
    }

    /// Drop pool entries based on each server's configured
    /// [`crate::McpServerConfig::idle_ttl`]. Use this when servers
    /// have heterogeneous lifecycle expectations — long-lived
    /// shared sidecars get a long TTL while short-lived per-tenant
    /// proxies get a short one.
    pub fn prune_idle_per_config(&self) -> usize {
        let now = now_pool_secs();
        let mut evicted = 0;
        let configs = Arc::clone(&self.configs);
        self.pool.retain(|(_, server), entry| {
            let last = entry.last_used_secs.load(Ordering::Relaxed);
            let idle = now.saturating_sub(last);
            let max = configs
                .get(server)
                .map_or(crate::DEFAULT_IDLE_TTL.as_secs(), |c| {
                    c.idle_ttl().as_secs()
                });
            if idle > max {
                evicted += 1;
                false
            } else {
                true
            }
        });
        evicted
    }

    /// Number of `(tenant, server)` clients currently in the pool.
    #[must_use]
    pub fn pool_size(&self) -> usize {
        self.pool.len()
    }

    /// Iterate the registered server names. Order is unspecified.
    pub fn server_names(&self) -> impl Iterator<Item = &str> {
        self.configs.keys().map(String::as_str)
    }
}

/// Decorate a transport-layer [`McpError`] with the `(server, op)`
/// pair so a multi-server agent run can correlate which dispatch
/// produced which error. Bare `McpError::JsonRpc { code, message }`
/// would otherwise produce identical strings across N servers,
/// making triage impossible.
fn correlate(err: McpError, server: &str, op: &str) -> McpError {
    match err {
        McpError::JsonRpc { code, message } => McpError::JsonRpc {
            code,
            message: format!("MCP server '{server}' {op}: {message}"),
        },
        McpError::Network { message, source } => McpError::Network {
            message: format!("MCP server '{server}' {op}: {message}"),
            source,
        },
        McpError::MalformedResponse { message, source } => McpError::MalformedResponse {
            message: format!("MCP server '{server}' {op}: {message}"),
            source,
        },
        other => other,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU32, Ordering};

    use serde_json::json;

    use entelix_core::context::ExecutionContext;

    use super::*;
    use crate::fsm::McpClientState;
    use crate::tool_definition::McpToolDefinition;

    /// Mock `McpClient` — records every `call_tool` invocation and
    /// stays in a chosen state for `state()` assertions.
    struct MockClient {
        state: parking_lot::Mutex<McpClientState>,
        calls: Mutex<Vec<(String, Value)>>,
    }

    impl MockClient {
        fn new() -> Self {
            Self {
                state: parking_lot::Mutex::new(McpClientState::Queued),
                calls: Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl McpClient for MockClient {
        async fn initialize(&self) -> McpResult<Vec<McpToolDefinition>> {
            *self.state.lock() = McpClientState::Ready;
            Ok(vec![McpToolDefinition {
                name: "echo".into(),
                description: "test".into(),
                input_schema: json!({"type":"object"}),
                extras: serde_json::Map::new(),
            }])
        }

        async fn call_tool(&self, name: &str, arguments: Value) -> McpResult<Value> {
            self.calls
                .lock()
                .unwrap()
                .push((name.to_owned(), arguments.clone()));
            Ok(json!({"echoed": arguments}))
        }

        fn state(&self) -> McpClientState {
            *self.state.lock()
        }
    }

    /// Factory that counts invocations and hands out `MockClient`s.
    #[derive(Default)]
    struct CountingFactory {
        builds: AtomicU32,
        last: Mutex<Option<Arc<MockClient>>>,
    }

    impl CountingFactory {
        fn builds(&self) -> u32 {
            self.builds.load(Ordering::SeqCst)
        }
        fn last_client(&self) -> Arc<MockClient> {
            self.last
                .lock()
                .unwrap()
                .clone()
                .expect("at least one build")
        }
    }

    #[async_trait]
    impl McpClientFactory for CountingFactory {
        async fn build(&self, _config: &McpServerConfig) -> McpResult<Arc<dyn McpClient>> {
            self.builds.fetch_add(1, Ordering::SeqCst);
            let client = Arc::new(MockClient::new());
            *self.last.lock().unwrap() = Some(client.clone());
            Ok(client)
        }
    }

    fn manager_with(servers: &[&str], factory: Arc<CountingFactory>) -> McpManager {
        let mut builder = McpManagerBuilder::new().with_factory(factory);
        for s in servers {
            builder = builder
                .register(McpServerConfig::http(*s, format!("https://{s}.invalid")).unwrap());
        }
        builder.build().unwrap()
    }

    #[test]
    fn builder_rejects_duplicate_server_names() {
        let err = McpManagerBuilder::new()
            .register(McpServerConfig::http("dup", "https://a").unwrap())
            .register(McpServerConfig::http("dup", "https://b").unwrap())
            .build()
            .unwrap_err();
        assert!(matches!(err, McpError::Config(ref m) if m.contains("duplicate")));
    }

    #[tokio::test]
    async fn unknown_server_returns_tenant_scoped_error() {
        let manager = McpManagerBuilder::new().build().unwrap();
        let ctx = ExecutionContext::new().with_tenant_id("tenant-x");
        let err = manager.list_tools(&ctx, "nope").await.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("nope"), "{msg}");
        assert!(msg.contains("tenant-x"), "{msg}");
    }

    #[tokio::test]
    async fn lazy_no_clients_until_first_use() {
        let factory = Arc::new(CountingFactory::default());
        let manager = manager_with(&["fs", "git"], factory.clone());
        assert_eq!(manager.pool_size(), 0);
        assert_eq!(factory.builds(), 0);

        let ctx = ExecutionContext::new().with_tenant_id("t1");
        manager.list_tools(&ctx, "fs").await.unwrap();
        assert_eq!(factory.builds(), 1);
        assert_eq!(manager.pool_size(), 1);
    }

    #[tokio::test]
    async fn second_call_for_same_tenant_server_reuses_client() {
        let factory = Arc::new(CountingFactory::default());
        let manager = manager_with(&["fs"], factory.clone());
        let ctx = ExecutionContext::new().with_tenant_id("t1");

        manager.list_tools(&ctx, "fs").await.unwrap();
        manager.list_tools(&ctx, "fs").await.unwrap();
        manager
            .call_tool(&ctx, "fs", "echo", json!({"x": 1}))
            .await
            .unwrap();

        assert_eq!(factory.builds(), 1);
        assert_eq!(manager.pool_size(), 1);
    }

    #[tokio::test]
    async fn f9_two_tenants_get_independent_clients() {
        let factory = Arc::new(CountingFactory::default());
        let manager = manager_with(&["fs"], factory.clone());

        let ctx_a = ExecutionContext::new().with_tenant_id("alpha");
        let ctx_b = ExecutionContext::new().with_tenant_id("bravo");

        manager.list_tools(&ctx_a, "fs").await.unwrap();
        manager.list_tools(&ctx_b, "fs").await.unwrap();

        assert_eq!(factory.builds(), 2, "F9: per-(tenant, server) isolation");
        assert_eq!(manager.pool_size(), 2);
    }

    #[tokio::test]
    async fn fsm_advances_to_ready_after_initialize() {
        let factory = Arc::new(CountingFactory::default());
        let manager = manager_with(&["fs"], factory.clone());
        let ctx = ExecutionContext::new().with_tenant_id("t1");

        manager.list_tools(&ctx, "fs").await.unwrap();
        assert_eq!(factory.last_client().state(), McpClientState::Ready);
    }

    #[tokio::test]
    async fn call_tool_routes_arguments_to_named_server() {
        let factory = Arc::new(CountingFactory::default());
        let manager = manager_with(&["fs", "git"], factory.clone());
        let ctx = ExecutionContext::new().with_tenant_id("t1");

        manager
            .call_tool(&ctx, "fs", "read", json!({"path": "/etc/hosts"}))
            .await
            .unwrap();
        manager
            .call_tool(&ctx, "git", "log", json!({"limit": 5}))
            .await
            .unwrap();

        assert_eq!(factory.builds(), 2);
    }

    #[tokio::test]
    async fn prune_idle_keeps_recently_used_clients() {
        let factory = Arc::new(CountingFactory::default());
        let manager = manager_with(&["fs"], factory.clone());
        let ctx = ExecutionContext::new().with_tenant_id("t1");
        manager.list_tools(&ctx, "fs").await.unwrap();
        assert_eq!(manager.pool_size(), 1);
        // Anything 0 seconds idle stays in.
        let evicted = manager.prune_idle(Duration::from_secs(60));
        assert_eq!(evicted, 0);
        assert_eq!(manager.pool_size(), 1);
    }

    #[tokio::test]
    async fn prune_idle_evicts_stale_clients() {
        // Advance pool origin so `now_pool_secs()` returns a value
        // strictly above zero before we backdate the entry to zero.
        let _ = pool_origin();
        std::thread::sleep(Duration::from_millis(1100));

        let factory = Arc::new(CountingFactory::default());
        let manager = manager_with(&["fs"], factory.clone());
        let ctx = ExecutionContext::new().with_tenant_id("t1");
        manager.list_tools(&ctx, "fs").await.unwrap();
        // Backdate the entry so it's well past any plausible TTL.
        for entry in manager.pool.iter() {
            entry.last_used_secs.store(0, Ordering::Relaxed);
        }
        // `Duration::ZERO` means "evict anything idle for any time"
        // — independent of how many wall-seconds the test took.
        let evicted = manager.prune_idle(Duration::ZERO);
        assert_eq!(evicted, 1);
        assert_eq!(manager.pool_size(), 0);

        // Re-using the (tenant, server) pair after prune triggers a
        // fresh build — the pool entry is genuinely gone.
        manager.list_tools(&ctx, "fs").await.unwrap();
        assert_eq!(factory.builds(), 2);
    }

    #[test]
    fn server_names_iterates_registered_configs() {
        let manager = McpManagerBuilder::new()
            .register(McpServerConfig::http("a", "https://a").unwrap())
            .register(McpServerConfig::http("b", "https://b").unwrap())
            .build()
            .unwrap();
        let mut names: Vec<&str> = manager.server_names().collect();
        names.sort_unstable();
        assert_eq!(names, vec!["a", "b"]);
    }
}
