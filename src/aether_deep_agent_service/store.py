import time
from pathlib import Path

from sqlalchemy import JSON, BigInteger, Boolean, ForeignKey, Integer, String, Text, select, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .schemas import DeepRunRequest, RunStatus


class Base(DeclarativeBase):
    pass


JsonValue = JSON().with_variant(JSONB, "postgresql")


class RunRecord(Base):
    __tablename__ = "deep_agent_run"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    task_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    request: Mapped[dict] = mapped_column(JsonValue, nullable=False)
    result: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False)
    updated_at: Mapped[int] = mapped_column(BigInteger, nullable=False)
    pause_requested: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class RunCheckpoint(Base):
    __tablename__ = "deep_agent_checkpoint"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), ForeignKey("deep_agent_run.run_id"), nullable=False, index=True)
    checkpoint_no: Mapped[int] = mapped_column(Integer, nullable=False)
    state: Mapped[dict] = mapped_column(JsonValue, nullable=False)
    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False)


class PendingInteraction(Base):
    __tablename__ = "deep_agent_pending_interaction"
    run_id: Mapped[str] = mapped_column(String(64), ForeignKey("deep_agent_run.run_id"), primary_key=True)
    interaction_type: Mapped[str] = mapped_column(String(32), nullable=False)
    payload: Mapped[dict] = mapped_column(JsonValue, nullable=False)
    updated_at: Mapped[int] = mapped_column(BigInteger, nullable=False)


class CallbackOutbox(Base):
    __tablename__ = "deep_agent_callback_outbox"
    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    data: Mapped[dict] = mapped_column(JsonValue, nullable=False)
    occurred_at: Mapped[int] = mapped_column(BigInteger, nullable=False)
    delivered: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class RunStore:
    def __init__(self, database_url: str) -> None:
        self.engine = create_async_engine(database_url)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    async def initialize(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
            if connection.dialect.name == "postgresql":
                await connection.execute(text("CREATE TABLE IF NOT EXISTS deep_agent_schema_migration (version VARCHAR(64) PRIMARY KEY, applied_at BIGINT NOT NULL)"))
                migration_dir = Path(__file__).with_name("migrations")
                for migration in sorted(migration_dir.glob("*.sql")):
                    applied = (await connection.execute(text("SELECT 1 FROM deep_agent_schema_migration WHERE version = :version"), {"version": migration.name})).scalar_one_or_none()
                    if applied is None:
                        for statement in migration.read_text(encoding="utf-8").split(";\n"):
                            if statement.strip():
                                await connection.execute(text(statement))
                        await connection.execute(text("INSERT INTO deep_agent_schema_migration(version, applied_at) VALUES (:version, :applied_at)"), {"version": migration.name, "applied_at": int(time.time() * 1000)})

    async def create_if_absent(self, request: DeepRunRequest) -> tuple[RunRecord, bool]:
        now = int(time.time() * 1000)
        async with self.sessions() as session:
            existing = await session.get(RunRecord, request.run_id)
            if existing is not None:
                return existing, False
            record = RunRecord(
                run_id=request.run_id,
                session_id=request.session_id,
                task_id=request.task_id,
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

    async def latest_for_session(self, session_id: str, task_id: str | None = None) -> RunRecord | None:
        async with self.sessions() as session:
            query = select(RunRecord).where(RunRecord.session_id == session_id)
            if task_id is not None:
                query = query.where(RunRecord.task_id == task_id)
            query = query.order_by(RunRecord.updated_at.desc(), RunRecord.created_at.desc()).limit(1)
            return (await session.execute(query)).scalar_one_or_none()

    async def update(self, run_id: str, status: RunStatus, result: str | None = None,
                     error: str | None = None) -> None:
        async with self.sessions() as session:
            record = await session.get(RunRecord, run_id)
            if record is None:
                return
            record.status = status
            if status == RunStatus.RUNNING:
                record.pause_requested = False
            record.result = result
            record.error = error
            record.updated_at = int(time.time() * 1000)
            await session.commit()

    async def request_pause(self, run_id: str) -> RunRecord | None:
        async with self.sessions() as session:
            record = await session.get(RunRecord, run_id)
            if record is not None:
                record.pause_requested = True
                record.updated_at = int(time.time() * 1000)
                await session.commit()
            return record

    async def checkpoint(self, run_id: str, state: dict) -> int:
        now = int(time.time() * 1000)
        async with self.sessions() as session:
            latest = (await session.execute(select(RunCheckpoint.checkpoint_no).where(RunCheckpoint.run_id == run_id).order_by(RunCheckpoint.checkpoint_no.desc()).limit(1))).scalar_one_or_none()
            checkpoint_no = (latest or 0) + 1
            session.add(RunCheckpoint(run_id=run_id, checkpoint_no=checkpoint_no, state=state, created_at=now))
            await session.commit()
            return checkpoint_no

    async def latest_checkpoint(self, run_id: str) -> RunCheckpoint | None:
        async with self.sessions() as session:
            return (await session.execute(select(RunCheckpoint).where(RunCheckpoint.run_id == run_id).order_by(RunCheckpoint.checkpoint_no.desc()).limit(1))).scalar_one_or_none()

    async def save_interaction(self, run_id: str, interaction_type: str, payload: dict) -> None:
        async with self.sessions() as session:
            interaction = await session.get(PendingInteraction, run_id)
            if interaction is None:
                interaction = PendingInteraction(run_id=run_id, interaction_type=interaction_type, payload=payload, updated_at=int(time.time() * 1000))
                session.add(interaction)
            else:
                interaction.interaction_type = interaction_type
                interaction.payload = payload
                interaction.updated_at = int(time.time() * 1000)
            await session.commit()

    async def take_interaction(self, run_id: str) -> PendingInteraction | None:
        async with self.sessions() as session:
            interaction = await session.get(PendingInteraction, run_id)
            if interaction is not None:
                await session.delete(interaction)
                await session.commit()
            return interaction

    async def pause_incomplete_runs(self) -> list[str]:
        async with self.sessions() as session:
            records = list((await session.execute(select(RunRecord).where(RunRecord.status.in_([RunStatus.QUEUED, RunStatus.RUNNING])))).scalars())
            now = int(time.time() * 1000)
            for record in records:
                record.status = RunStatus.PAUSED
                record.pause_requested = False
                record.updated_at = now
            await session.commit()
            return [record.run_id for record in records]

    async def status(self, run_id: str) -> tuple[RunRecord | None, int]:
        record = await self.get(run_id)
        checkpoint = await self.latest_checkpoint(run_id)
        return record, checkpoint.checkpoint_no if checkpoint is not None else 0

    async def enqueue_callback(self, event_id: str, run_id: str, event_type: str, data: dict, occurred_at: int) -> CallbackOutbox:
        async with self.sessions() as session:
            event = await session.get(CallbackOutbox, event_id)
            if event is None:
                event = CallbackOutbox(event_id=event_id, run_id=run_id, event_type=event_type, data=data, occurred_at=occurred_at, delivered=False)
                session.add(event); await session.commit()
            return event

    async def mark_callback_delivered(self, event_id: str) -> None:
        async with self.sessions() as session:
            event = await session.get(CallbackOutbox, event_id)
            if event is not None:
                event.delivered = True; await session.commit()

    async def pending_callbacks(self) -> list[CallbackOutbox]:
        async with self.sessions() as session:
            return list((await session.execute(select(CallbackOutbox).where(CallbackOutbox.delivered == False).order_by(CallbackOutbox.occurred_at))).scalars())
