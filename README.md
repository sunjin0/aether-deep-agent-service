# Aether Deep Agent Service

Asynchronous Deep Agents runtime for Aether complex, read-only knowledge tasks.

## Local development

```powershell
uv sync
$env:AETHER_DEEP_AGENT_SHARED_SECRET = "change-me"
$env:AETHER_DEEP_AGENT_CALLBACK_BASE_URL = "http://localhost:8080"
$env:AETHER_DEEP_AGENT_MODEL = "openai:gpt-5.5"
$env:AETHER_DEEP_AGENT_MCP_URL = "http://localhost:8000/mcp"
uv run aether-deep-agent-service
```

`POST /v1/runs` and `POST /v1/runs/{run_id}/cancel` require the HMAC headers
`X-Aether-Key-Id`, `X-Aether-Timestamp`, and `X-Aether-Signature`. Callback
events use the same signing scheme. The service fails a run when no model is
configured; it never fabricates an agent answer.
