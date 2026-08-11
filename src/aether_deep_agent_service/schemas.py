from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class RunStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


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


class DeepRunRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=64)
    user_id: str = Field(min_length=1, max_length=64)
    agent_id: str = Field(min_length=1, max_length=64)
    conversation_id: str = Field(min_length=1, max_length=64)
    task: str = Field(min_length=1, max_length=20000)
    system_prompt: str = ""
    knowledge_sources: list[KnowledgeSource] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    delegation_token: str = Field(min_length=1)
    max_steps: int = Field(default=12, ge=1, le=50)
    timeout_seconds: int | None = Field(default=None, ge=30, le=3600)


class DeepRunResponse(BaseModel):
    run_id: str
    status: RunStatus
    created: bool


class ResumeRunRequest(BaseModel):
    decisions: list[dict[str, Any]] = Field(default_factory=list, max_length=20)
    answers: dict[str, Any] = Field(default_factory=dict)


class CallbackEvent(BaseModel):
    event_id: str
    event_type: str
    run_id: str
    occurred_at: int
    data: dict[str, Any] = Field(default_factory=dict)
