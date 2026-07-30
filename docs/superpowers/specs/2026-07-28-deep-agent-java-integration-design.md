# Deep Agent 与 Java 平台联动设计

## 目标

保持 Deep Agent 服务的异步任务模型，使其可由 Aether Java Admin 安全创建、可观测回调，并可在 MCP 调用时使用 Java 签发的最小权限委托令牌。

## 范围

- 保持 `POST /v1/runs` 的 `202 Accepted` 异步响应以及 `POST /v1/runs/{run_id}/cancel` 取消语义。
- 保持向 Java `POST /api/agent/deep-runs/callback/{run_id}` 的回调入口约定。
- 回调失败时在服务侧进行有限次数重试，整个重试周期复用同一个事件 ID 和事件体。
- 明确接收 Java 提供的知识来源、MCP 工具白名单和委托 JWT，并以集成测试覆盖协议。

## 非目标

- 不提供浏览器 SSE 接口。
- 不直接访问 Java 数据库或管理 Java 会话/消息。
- 不接受用户登录令牌作为 MCP 授权。
- 不把 Java 的回调失败视为可以伪造最终答案的理由。

## 创建任务契约

`POST /v1/runs` 接收 `DeepRunRequest`，字段包括：

```json
{
  "run_id": "Java agent_run.id",
  "user_id": "用户ID",
  "agent_id": "Agent定义ID",
  "conversation_id": "会话ID",
  "task": "用户任务",
  "system_prompt": "系统提示词",
  "knowledge_sources": [
    { "title": "文档标题", "content": "片段", "citation": "【1】" }
  ],
  "allowed_tools": ["mcp_tool_name"],
  "delegation_token": "Java签发的JWT",
  "max_steps": 12
}
```

Java 不传 `timeout_seconds` 时使用服务当前默认的 `run_timeout_seconds`（默认 600 秒）。`run_id` 是请求幂等键：同一 ID 再次提交只返回现有任务状态，不创建第二个后台任务。

`POST /v1/runs` 与取消接口要求 HMAC 请求头 `X-Aether-Key-Id`、`X-Aether-Timestamp`、`X-Aether-Signature`。签名是 `HMAC-SHA256(sharedSecret, timestamp + "." + 原始请求体字节)`，最多接受五分钟前的时间戳。

## MCP 授权边界

Deep Agent 只有在 `allowed_tools` 非空时才加载 MCP 工具，并仅保留 MCP 返回工具中名称匹配白名单的项目。调用 MCP 时只携带 `Authorization: Bearer <delegation_token>`。

委托 JWT 由 Java 使用 `AETHER_MCP_DELEGATION_SECRET` 签发，包含 `runId`、`userId`、`agentId`、`allowedTools`、`iat`、`exp`，有效期五分钟。Deep Agent 不解析或扩大该授权范围；MCP 服务负责验证 JWT 并再次强制工具白名单。

## 回调契约

每个事件生成一次 UUID 作为 `event_id`，并构造：

```json
{
  "event_id": "uuid",
  "event_type": "tool.completed",
  "run_id": "Java agent_run.id",
  "occurred_at": 1760000000000,
  "data": { "toolName": "search", "message": "Completed search" }
}
```

回调 URL 固定为 `{callback_base_url}/api/agent/deep-runs/callback/{run_id}`。请求使用与入站请求完全相同的 HMAC 签名规则和服务共享密钥。

事件顺序为：

1. `run.started`，数据含 `status=RUNNING`。
2. `plan.updated`，数据含计划摘要和 `maxSteps`。
3. `step.started`。
4. 零次或多次 `tool.started`、`tool.completed`。
5. 仅一个终态事件：`run.completed`、`run.failed` 或 `run.cancelled`。

`run.completed.data` 必须含 `content`、`citations`、`model`、`tools`、`promptTokens`、`completionTokens`、`totalTokens`。`run.failed.data` 必须含可展示的 `error` 字符串。

## 回调可靠性

单次回调的超时仍使用 `callback_timeout_seconds`。网络错误、超时、5xx 和 429 触发有限重试；4xx（除 429）作为不可恢复错误记录日志后停止重试。每次重试必须复用相同的 JSON 事件体、`event_id` 和 `occurred_at`，但签名时间戳和签名可重新生成。

Java 按 `(run_id, event_id)` 幂等保存，因此重试不会重复创建聊天消息或工具步骤。终态回调无法投递时，服务本地 RunStore 仍保存最终状态和结果/错误，供后续排障。

## 执行限制

- 服务未配置模型时运行失败，并回调 `run.failed`，不生成虚构回答。
- 使用只读知识分析提示词；来源仅来自请求提供的 `knowledge_sources`。
- 禁用本地文件、Shell 和写入工具；可用 MCP 工具仅来自 `allowed_tools`。
- `max_steps` 必须处于 1 至 50；执行总超时处于 30 至 3600 秒。

## 测试和验收

- 测试 Java 风格的 HMAC 请求可创建幂等任务，错误签名和过期签名被拒绝。
- 测试回调 URL、回调 HMAC、事件 JSON 与 Java 契约一致。
- 使用 HTTP mock 测试 5xx/429 重试、4xx 不重试、重试期间 `event_id` 与 `occurred_at` 不变。
- 测试所有终态事件与完成结果字段。
- 测试空白名单不加载 MCP 工具，白名单只加载匹配工具且 Bearer Token 等于委托令牌。
- 执行 Python 项目已有测试命令，确认健康检查、签名、任务创建、遥测和回调测试通过。
