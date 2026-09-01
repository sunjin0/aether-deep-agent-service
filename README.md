# Aether Deep Agent Service

Aether 的 Python Deep Agent 执行服务，用于多步骤复杂任务、任务规划、MCP 工具调用、用户追问与流式回调。它不直接面向浏览器；Java Admin 负责创建运行、签发委派令牌、接收回调和保存审计记录。

## 职责

- 接收 Java 签名的运行请求并执行 Deep Agent。
- 先生成任务计划，再流式发送任务步骤、文本增量、工具审批、`ask_user` 与最终结果。
- 仅使用 Java 提供的知识库片段；引用保留文档名、分块 ID 和引用编号。
- 通过 MCP 调用工具时透传 Java 签发的短期 JWT，不保存静态工具 Token。
- 默认使用 SQLite 保存运行状态，也可通过 `AETHER_DEEP_AGENT_DATABASE_URL` 改为其他 SQLAlchemy 支持的数据库。

## 开发

要求 Python 3.11+ 与 [uv](https://docs.astral.sh/uv/)。

```powershell
uv sync
$env:AETHER_DEEP_AGENT_SHARED_SECRET = "change-me"
$env:AETHER_DEEP_AGENT_CALLBACK_BASE_URL = "http://localhost:8080"
$env:AETHER_DEEP_AGENT_MCP_URL = "http://localhost:8000/mcp"
$env:AETHER_DEEP_AGENT_MODEL = "openai:gpt-4.1-mini"
$env:OPENAI_API_KEY = "..."
uv run aether-deep-agent-service
```

健康检查：`GET /health`。测试：

```powershell
uv run pytest -q
```

## 安全协议

`POST /v1/runs`、`POST /v1/runs/{run_id}/cancel` 与恢复接口要求 Java 的 HMAC 请求头：`X-Aether-Key-Id`、`X-Aether-Timestamp`、`X-Aether-Signature`。回调 Java Admin 时使用同一签名机制。

关键变量：

| 变量 | 说明 |
| --- | --- |
| `AETHER_DEEP_AGENT_SHARED_SECRET` | 与 Java 共用的回调/请求验签密钥。 |
| `AETHER_DEEP_AGENT_CALLBACK_BASE_URL` | Java Admin 容器内地址。 |
| `AETHER_DEEP_AGENT_MCP_URL` | MCP Streamable HTTP 端点，通常为 `http://aether-mcp:8000/mcp`。 |
| `AETHER_DEEP_AGENT_MODEL` | LangChain 模型标识。 |
| `OPENAI_API_KEY`、`OPENAI_BASE_URL` | OpenAI 兼容模型服务配置。 |

## OpenTelemetry

Trace 和 Log OTLP/HTTP 导出默认关闭。对接 Collector 时显式配置：

```powershell
$env:AETHER_OTLP_TRACES_ENABLED = "true"
$env:AETHER_OTLP_TRACES_URL = "http://otel-collector:4318/v1/traces"
$env:AETHER_OTLP_LOGS_URL = "http://otel-collector:4318/v1/logs"
$env:OTEL_SERVICE_NAME = "aether-deep-agent-service"
```

Runtime 会继承入站 `traceparent`，记录 HTTP span，并在服务退出时 flush exporter；不会导出 Prompt、请求体或凭据。

## Docker

项目自带 `docker-compose.yml`，单独部署时需接入 Java、MCP 和基础设施所在的共享网络：

```powershell
docker compose up -d --build deep-agent
```

完整平台部署请使用 Java 项目的 `docker-compose.all.yml`。
