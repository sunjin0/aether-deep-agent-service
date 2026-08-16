from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class RunStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    PAUSED = "PAUSED"


class KnowledgeSource(BaseModel):
    title: str
    content: str
    citation: str
    citationIndex: int
    documentName: str | None = None
    documentId: str | None = None
    chunkId: str | None = None
    sectionPath: str | None = None
    similarity: float | None = None
    retrievalScore: float | None = None


class ConversationMemoryMessage(BaseModel):
    """A bounded, persisted message supplied from the owning conversation."""
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str = Field(min_length=1, max_length=24000)


class DeepRunRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=64)
    user_id: str = Field(min_length=1, max_length=64)
    agent_id: str = Field(min_length=1, max_length=64)
    conversation_id: str = Field(min_length=1, max_length=64)
    session_id: str | None = Field(default=None, min_length=1, max_length=64)
    task_id: str | None = Field(default=None, min_length=1, max_length=64)
    task: str = Field(min_length=1, max_length=20000)
    task_state: dict[str, Any] = Field(default_factory=dict)
    system_prompt: str = ""
    conversation_memory: list[ConversationMemoryMessage] = Field(default_factory=list, max_length=24)
    knowledge_sources: list[KnowledgeSource] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    delegation_token: str = Field(min_length=1)
    tool_approval_policy: str = Field(default="ask", pattern="^(ask|risky|never)$")
    max_steps: int = Field(default=12, ge=1, le=50)
    # 计划先行：生成初始计划后暂停等待用户确认，批准后才开始执行（Codex/Claude 风格）。
    plan_approval_required: bool = False
    timeout_seconds: int | None = Field(default=None, ge=30, le=3600)


class DeepRunResponse(BaseModel):
    run_id: str
    status: RunStatus
    created: bool


class ResumeRunRequest(BaseModel):
    decisions: list[dict[str, Any]] = Field(default_factory=list, max_length=20)
    answers: dict[str, Any] = Field(default_factory=dict)
    plan_feedback: str | None = None


class DeepRunStatusResponse(BaseModel):
    run_id: str
    status: RunStatus
    checkpoint_no: int = 0
    updated_at: int


class DeepSessionStatusResponse(BaseModel):
    """Latest durable execution state for one Admin-owned Agent Session."""
    session_id: str
    task_id: str | None = None
    run_id: str | None = None
    status: RunStatus | None = None
    checkpoint_no: int = 0
    updated_at: int | None = None


class CallbackEvent(BaseModel):
    event_id: str
    event_type: str
    run_id: str
    occurred_at: int
    data: dict[str, Any] = Field(default_factory=dict)
