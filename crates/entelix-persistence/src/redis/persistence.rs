//! `RedisPersistence` — handle owning a `redis::aio::ConnectionManager`
//! plus builder for the four backend handles.

use std::sync::Arc;

use redis::Client;
use redis::aio::ConnectionManager;

use crate::error::{PersistenceError, PersistenceResult};

/// Redis-backed persistence bundle. Cheap to clone — the connection
/// manager is reference-counted internally.
#[derive(Clone)]
pub struct RedisPersistence {
    manager: Arc<ConnectionManager>,
}

impl RedisPersistence {
    fn from_manager(manager: ConnectionManager) -> Self {
        Self {
            manager: Arc::new(manager),
        }
    }

    /// Borrow the underlying connection manager.
    pub fn manager(&self) -> &ConnectionManager {
        &self.manager
    }

    /// Build the lock handle.
    pub fn lock(&self) -> super::RedisLock {
        super::RedisLock::new(Arc::clone(&self.manager))
    }

    /// Build a typed [`super::RedisCheckpointer`].
    pub fn checkpointer<S>(&self) -> super::RedisCheckpointer<S>
    where
        S: Clone + Send + Sync + serde::Serialize + serde::de::DeserializeOwned + 'static,
    {
        super::RedisCheckpointer::new(Arc::clone(&self.manager))
    }

    /// Build a typed [`super::RedisStore`].
    pub fn store<V>(&self) -> super::RedisStore<V>
    where
        V: Clone + Send + Sync + serde::Serialize + serde::de::DeserializeOwned + 'static,
    {
        super::RedisStore::new(Arc::clone(&self.manager))
    }

    /// Build the session-log handle.
    pub fn session_log(&self) -> super::RedisSessionLog {
        super::RedisSessionLog::new(Arc::clone(&self.manager))
    }

    /// Start a builder.
    pub fn builder() -> RedisPersistenceBuilder {
        RedisPersistenceBuilder { url: None }
    }
}

/// Fluent builder for [`RedisPersistence`].
#[derive(Debug)]
#[must_use]
pub struct RedisPersistenceBuilder {
    url: Option<String>,
}

impl RedisPersistenceBuilder {
    /// Redis connection URL (`redis://host:6379/0`).
    pub fn with_connection_string(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }

    /// Open the connection manager and return the bundle.
    pub async fn connect(self) -> PersistenceResult<RedisPersistence> {
        let url = self
            .url
            .ok_or_else(|| PersistenceError::Config("connection_string is required".into()))?;
        let client = Client::open(url.as_str())
            .map_err(|e| PersistenceError::Backend(format!("client open: {e}")))?;
        let manager = ConnectionManager::new(client)
            .await
            .map_err(|e| PersistenceError::Backend(format!("connection manager: {e}")))?;
        Ok(RedisPersistence::from_manager(manager))
    }
}
