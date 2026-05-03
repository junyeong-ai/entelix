-- entelix-persistence v2 — Row-Level Security on tenant-scoped tables.
--
-- Defense-in-depth for invariant #11 (multi-tenant isolation). Every
-- tenant-scoped table gains a policy that consults
-- `current_setting('entelix.tenant_id', true)` to gate both reads
-- (USING) and writes (WITH CHECK). The SDK's persistence layer sets
-- this variable inside a transaction wrapper before issuing tenant-
-- scoped queries; deployments that wire third-party tooling against
-- the same DB must do the same or run as a role with BYPASSRLS.
--
-- FORCE ROW LEVEL SECURITY makes the policy apply to the table
-- owner as well (without FORCE the table owner is exempt). This
-- means the SDK's normal database role — even if it owns the
-- tables — is subject to the policy. Maintenance operations that
-- legitimately need cross-tenant access (eg `Store::evict_expired`
-- TTL sweepers) require a separate role with the BYPASSRLS
-- attribute, run on a maintenance schedule outside the per-request
-- application path.
--
-- Application code that forgets to set the variable does not
-- silently return cross-tenant data: `current_setting(name, true)`
-- returns NULL when unset, and `tenant_id = NULL` is NULL/unknown
-- which the policy treats as false. Both reads and writes fail
-- loudly (empty result set / WITH CHECK violation) — the
-- fail-closed default invariant #15 demands at the persistence
-- boundary.

-- ── memory_items ────────────────────────────────────────────────────
ALTER TABLE memory_items ENABLE ROW LEVEL SECURITY;
ALTER TABLE memory_items FORCE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON memory_items
    USING (tenant_id = current_setting('entelix.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('entelix.tenant_id', true));

-- ── session_events ──────────────────────────────────────────────────
ALTER TABLE session_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE session_events FORCE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON session_events
    USING (tenant_id = current_setting('entelix.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('entelix.tenant_id', true));

-- ── checkpoints ─────────────────────────────────────────────────────
ALTER TABLE checkpoints ENABLE ROW LEVEL SECURITY;
ALTER TABLE checkpoints FORCE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON checkpoints
    USING (tenant_id = current_setting('entelix.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('entelix.tenant_id', true));

-- Bump the bundle's recorded schema version.
UPDATE schema_version
SET version = 2, applied_at = now()
WHERE component = 'entelix-persistence';
