import os
from typing import Any

from opentelemetry import trace
from opentelemetry.context import attach, detach
from opentelemetry.propagate import extract
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry._logs import set_logger_provider
import logging
from starlette.middleware.base import BaseHTTPMiddleware

_provider: TracerProvider | None = None
_log_provider: LoggerProvider | None = None


def configure_tracing() -> None:
    global _provider, _log_provider
    endpoint = os.getenv("AETHER_OTLP_TRACES_URL", "").strip()
    if os.getenv("AETHER_OTLP_TRACES_ENABLED", "false").lower() not in {"1", "true", "yes", "on"} or not endpoint:
        return
    if isinstance(trace.get_tracer_provider(), TracerProvider):
        return
    provider = TracerProvider(resource=Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "aether-deep-agent-service")}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    _provider = provider
    logs_endpoint = os.getenv("AETHER_OTLP_LOGS_URL", "").strip() or endpoint.replace("/v1/traces", "/v1/logs")
    log_provider = LoggerProvider(resource=Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "aether-deep-agent-service")}))
    log_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter(endpoint=logs_endpoint)))
    set_logger_provider(log_provider)
    _log_provider = log_provider
    logging.getLogger().addHandler(LoggingHandler(level=logging.NOTSET, logger_provider=log_provider))


def shutdown_tracing() -> None:
    if _provider is not None:
        _provider.shutdown()
    if _log_provider is not None:
        _log_provider.shutdown()


class OTelMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Any, call_next: Any) -> Any:
        carrier = {"traceparent": request.headers["traceparent"]} if "traceparent" in request.headers else {}
        token = attach(extract(carrier))
        tracer = trace.get_tracer("aether.deep-agent.http")
        try:
            with tracer.start_as_current_span("deep-agent.http", attributes={
                "http.request.method": request.method,
                "url.path": request.url.path,
            }) as span:
                response = await call_next(request)
                span.set_attribute("http.response.status_code", response.status_code)
                return response
        finally:
            detach(token)
