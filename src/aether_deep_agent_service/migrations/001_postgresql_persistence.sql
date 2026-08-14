ALTER TABLE deep_agent_run ADD COLUMN IF NOT EXISTS pause_requested BOOLEAN NOT NULL DEFAULT FALSE;
CREATE TABLE IF NOT EXISTS deep_agent_checkpoint (
  id BIGSERIAL PRIMARY KEY, run_id VARCHAR(64) NOT NULL REFERENCES deep_agent_run(run_id), checkpoint_no INTEGER NOT NULL,
  state JSONB NOT NULL, created_at BIGINT NOT NULL, UNIQUE(run_id, checkpoint_no)
);
CREATE TABLE IF NOT EXISTS deep_agent_pending_interaction (
  run_id VARCHAR(64) PRIMARY KEY REFERENCES deep_agent_run(run_id), interaction_type VARCHAR(32) NOT NULL,
  payload JSONB NOT NULL, updated_at BIGINT NOT NULL
);
CREATE TABLE IF NOT EXISTS deep_agent_callback_outbox (
  event_id VARCHAR(64) PRIMARY KEY, run_id VARCHAR(64) NOT NULL, event_type VARCHAR(64) NOT NULL,
  data JSONB NOT NULL, occurred_at BIGINT NOT NULL, delivered BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS deep_agent_checkpoint_run_idx ON deep_agent_checkpoint(run_id, checkpoint_no DESC);
CREATE INDEX IF NOT EXISTS deep_agent_callback_outbox_pending_idx ON deep_agent_callback_outbox(delivered, occurred_at) WHERE delivered = FALSE;
