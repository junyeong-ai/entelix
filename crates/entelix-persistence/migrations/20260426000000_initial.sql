-- entelix-persistence initial schema (v1).
--
-- Four tenant-scoped tables backing the three storage traits + a
-- schema_version stamp for the bundle as a whole.

CREATE TABLE IF NOT EXISTS schema_version (
    component TEXT PRIMARY KEY,
    version BIGINT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO schema_version (component, version)
VALUES ('entelix-persistence', 1)
ON CONFLICT (component) DO NOTHING;

-- ── Tier-2: SessionLog ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS session_events (
    tenant_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    seq BIGINT NOT NULL,
    event JSONB NOT NULL,
    ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, thread_id, seq)
);

CREATE INDEX IF NOT EXISTS session_events_thread_idx
    ON session_events (tenant_id, thread_id);

-- ── Tier-1 snapshots: Checkpointer ────────────────────────────────
-- (tenant_id, thread_id, id) is the canonical addressing tuple
-- (Invariant 11). Every read/write partitions by tenant_id; cross-
-- tenant reads are structurally impossible at the SQL surface.
CREATE TABLE IF NOT EXISTS checkpoints (
    tenant_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    id UUID NOT NULL,
    parent_id UUID,
    step BIGINT NOT NULL,
    state JSONB NOT NULL,
    next_node TEXT,
    ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, thread_id, id)
);

CREATE INDEX IF NOT EXISTS checkpoints_thread_step_idx
    ON checkpoints (tenant_id, thread_id, step DESC, ts DESC);

CREATE INDEX IF NOT EXISTS checkpoints_parent_idx
    ON checkpoints (tenant_id, thread_id, parent_id)
    WHERE parent_id IS NOT NULL;

-- ── Tier-3: Memory Store ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS memory_items (
    tenant_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    key TEXT NOT NULL,
    value JSONB NOT NULL,
    ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Absolute expiry — `NULL` = no TTL. The Store::evict_expired
    -- sweeper deletes rows where expires_at <= now().
    expires_at TIMESTAMPTZ,
    PRIMARY KEY (tenant_id, namespace, key)
);

CREATE INDEX IF NOT EXISTS memory_items_tenant_ns_idx
    ON memory_items (tenant_id, namespace);

-- Partial index — only TTL'd rows pay the index-maintenance cost,
-- keeping the table fast for tenants that never opt into TTL.
CREATE INDEX IF NOT EXISTS memory_items_expires_at_idx
    ON memory_items (expires_at)
    WHERE expires_at IS NOT NULL;
