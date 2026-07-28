import time

from sqlalchemy import JSON, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .schemas import DeepRunRequest, RunStatus


class Base(DeclarativeBase):
    pass


class RunRecord(Base):
    __tablename__ = "deep_agent_run"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    request: Mapped[dict] = mapped_column(JSON, nullable=False)
    result: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_at: Mapped[int] = mapped_column(Integer, nullable=False)


class RunStore:
    def __init__(self, database_url: str) -> None:
        self.engine = create_async_engine(database_url)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    async def initialize(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def create_if_absent(self, request: DeepRunRequest) -> tuple[RunRecord, bool]:
        now = int(time.time() * 1000)
        async with self.sessions() as session:
            existing = await session.get(RunRecord, request.run_id)
            if existing is not None:
                return existing, False
            record = RunRecord(
                run_id=request.run_id,
                status=RunStatus.QUEUED,
                request=request.model_dump(mode="json"),
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            await session.commit()
            return record, True

    async def get(self, run_id: str) -> RunRecord | None:
        async with self.sessions() as session:
            return await session.get(RunRecord, run_id)

    async def update(self, run_id: str, status: RunStatus, result: str | None = None,
                     error: str | None = None) -> None:
        async with self.sessions() as session:
            record = await session.get(RunRecord, run_id)
            if record is None:
                return
            record.status = status
            record.result = result
            record.error = error
            record.updated_at = int(time.time() * 1000)
            await session.commit()
