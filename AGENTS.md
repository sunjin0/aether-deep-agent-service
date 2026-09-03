# AGENTS.md

## Project overview

Python 3.11+ asynchronous Deep Agent execution service. The service is a FastAPI application used by Java Admin for run creation, cancellation, callbacks, planning, streaming and MCP tool execution. It is not a browser-facing application.

## Development

Use `uv` for dependency management and execution:

```powershell
uv sync
uv run pytest -q
uv run aether-deep-agent-service
```

The service entry point is `src/aether_deep_agent_service/__main__.py`; health is `GET /health`. Docker Compose deployment is `docker compose up -d --build deep-agent`.

## Runtime contracts

Java Admin signs requests and callbacks with the configured HMAC headers. MCP delegation uses short-lived JWTs signed by Java; preserve `runId`, `userId`, `agentId` and `allowedTools` semantics. Do not add static token allowlists or persist plaintext credentials. Configuration is environment-driven; never commit `.env` secrets.

## Code organization

Keep HTTP routes and lifecycle wiring in the application package, execution logic in the agent/service modules, and MCP integration in its adapter layer. Preserve streaming event names and response schemas because Java Admin consumes them. Use Pydantic settings and existing async database abstractions.

## Verification and commits

Run focused tests for changed modules, then `uv run pytest -q`. Use Conventional Commits: `<type>(<scope>): <中文提交描述>` with types such as `feat`, `fix`, `refactor`, `test`, `build`, `docs`, and `chore`. The commit description must be in Chinese and focused on one change; the commit body must list what was changed, affected runtime/API/configuration behavior, and verification results. Review `git diff`, and exclude secrets, caches and generated files.
