//! `PostgresPersistence` — handle owning a `sqlx::PgPool` plus
//! builder for the four backend handles (lock, checkpointer, store,
//! session log).

use std::sync::Arc;
use std::time::Duration;

use sqlx::postgres::{PgPool, PgPoolOptions};

use crate::error::{PersistenceError, PersistenceResult};

// Defaults documented on the corresponding builder methods.
const DEFAULT_ACQUIRE_TIMEOUT: Duration = Duration::from_secs(10);
const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_mins(5);
const DEFAULT_MAX_CONNECTIONS: u32 = 16;

/// Postgres-backed persistence bundle. Cheap to clone — the pool is
/// reference-counted internally.
#[derive(Clone)]
pub struct PostgresPersistence {
    pool: Arc<PgPool>,
}

impl PostgresPersistence {
    /// Build via [`PostgresPersistenceBuilder::builder`]. Direct
    /// construction is not exposed — every backend handle ultimately
    /// depends on a configured pool.
    fn from_pool(pool: PgPool) -> Self {
        Self {
            pool: Arc::new(pool),
        }
    }

    /// Borrow the underlying `sqlx::PgPool`. Useful for users that
    /// want to share the pool with their own SQL operations.
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Run pending schema migrations bundled with the crate. Idempotent.
    pub async fn migrate(&self) -> PersistenceResult<()> {
        sqlx::migrate!("./migrations")
            .run(&*self.pool)
            .await
            .map_err(|e| PersistenceError::Backend(format!("migrate: {e}")))?;
        Ok(())
    }

    /// Build the lock handle backing this pool.
    pub fn lock(&self) -> super::PostgresLock {
        super::PostgresLock::new(Arc::clone(&self.pool))
    }

    /// Build a typed [`super::PostgresCheckpointer`] for a state type.
    pub fn checkpointer<S>(&self) -> super::PostgresCheckpointer<S>
    where
        S: Clone + Send + Sync + serde::Serialize + serde::de::DeserializeOwned + 'static,
    {
        super::PostgresCheckpointer::new(Arc::clone(&self.pool))
    }

    /// Build a typed [`super::PostgresStore`] for a stored value type.
    pub fn store<V>(&self) -> super::PostgresStore<V>
    where
        V: Clone + Send + Sync + serde::Serialize + serde::de::DeserializeOwned + 'static,
    {
        super::PostgresStore::new(Arc::clone(&self.pool))
    }

    /// Build the session-log handle backing this pool.
    pub fn session_log(&self) -> super::PostgresSessionLog {
        super::PostgresSessionLog::new(Arc::clone(&self.pool))
    }
}

/// Fluent builder for [`PostgresPersistence`]. Use
/// `PostgresPersistence::builder()`.
#[derive(Debug)]
#[must_use]
pub struct PostgresPersistenceBuilder {
    url: Option<String>,
    max_connections: u32,
    acquire_timeout: Duration,
    idle_timeout: Duration,
    test_before_acquire: bool,
}

impl PostgresPersistence {
    /// Start a builder. `connection_string` is the only required
    /// field; everything else has a sensible default.
    pub fn builder() -> PostgresPersistenceBuilder {
        PostgresPersistenceBuilder {
            url: None,
            max_connections: DEFAULT_MAX_CONNECTIONS,
            acquire_timeout: DEFAULT_ACQUIRE_TIMEOUT,
            idle_timeout: DEFAULT_IDLE_TIMEOUT,
            test_before_acquire: true,
        }
    }
}

impl PostgresPersistenceBuilder {
    /// Postgres connection string (`postgres://user:pass@host/db`).
    pub fn with_connection_string(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }

    /// Override the pool size cap.
    pub const fn with_max_connections(mut self, n: u32) -> Self {
        self.max_connections = n;
        self
    }

    /// Override the per-acquire timeout.
    pub const fn with_acquire_timeout(mut self, timeout: Duration) -> Self {
        self.acquire_timeout = timeout;
        self
    }

    /// Override the idle-connection eviction timeout.
    pub const fn with_idle_timeout(mut self, timeout: Duration) -> Self {
        self.idle_timeout = timeout;
        self
    }

    /// Toggle pre-acquire health check (`SELECT 1`). On by default;
    /// disable if your network is reliable enough to amortise the
    /// extra round-trip.
    pub const fn with_test_before_acquire(mut self, on: bool) -> Self {
        self.test_before_acquire = on;
        self
    }

    /// Open the pool, optionally run migrations, return the bundle.
    pub async fn connect(self) -> PersistenceResult<PostgresPersistence> {
        let url = self
            .url
            .ok_or_else(|| PersistenceError::Config("connection_string is required".into()))?;
        let pool = PgPoolOptions::new()
            .max_connections(self.max_connections)
            .acquire_timeout(self.acquire_timeout)
            .idle_timeout(Some(self.idle_timeout))
            .test_before_acquire(self.test_before_acquire)
            .connect(&url)
            .await
            .map_err(|e| PersistenceError::Backend(format!("connect: {e}")))?;
        Ok(PostgresPersistence::from_pool(pool))
    }

    /// Connect and run migrations in one call. The 90% path.
    pub async fn connect_and_migrate(self) -> PersistenceResult<PostgresPersistence> {
        let p = self.connect().await?;
        p.migrate().await?;
        Ok(p)
    }
}
