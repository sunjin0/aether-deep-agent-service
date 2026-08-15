ALTER TABLE deep_agent_run ADD COLUMN IF NOT EXISTS session_id VARCHAR(64);
ALTER TABLE deep_agent_run ADD COLUMN IF NOT EXISTS task_id VARCHAR(64);
CREATE INDEX IF NOT EXISTS deep_agent_run_session_updated_idx
    ON deep_agent_run(session_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS deep_agent_run_session_task_updated_idx
    ON deep_agent_run(session_id, task_id, updated_at DESC);
