"""Small OpenTelemetry pipeline: local JSONL or authenticated Langfuse OTLP/HTTP."""

from __future__ import annotations

import atexit
import base64
import json
import os
from pathlib import Path
from threading import Lock

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult, SimpleSpanProcessor

_provider: TracerProvider | None = None
_lock = Lock()


class JsonlExporter(SpanExporter):
    def __init__(self, path: str):
        self.path = Path(path)

    def export(self, spans):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as out:
            for span in spans:
                out.write(json.dumps({
                    "trace_id": f"{span.context.trace_id:032x}",
                    "span_id": f"{span.context.span_id:016x}",
                    "parent_span_id": f"{span.parent.span_id:016x}" if span.parent else None,
                    "name": span.name, "attributes": dict(span.attributes or {}),
                    "start_ns": span.start_time, "end_ns": span.end_time,
                }, default=str) + "\n")
        return SpanExportResult.SUCCESS


def configure():
    global _provider
    with _lock:
        if _provider is not None:
            return _provider.get_tracer("weather-workshop")
        provider = TracerProvider(resource=Resource.create({"service.name": "workshop-weather-agent"}))
        trace_file = os.getenv("TRACE_FILE", str(Path(__file__).resolve().parents[2] / "traces.jsonl"))
        if trace_file:
            provider.add_span_processor(SimpleSpanProcessor(JsonlExporter(trace_file)))
        public = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret = os.getenv("LANGFUSE_SECRET_KEY")
        if bool(public) != bool(secret):
            raise RuntimeError("Set both LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
        if public and secret:
            host = os.getenv("LANGFUSE_HOST", "http://127.0.0.1:3000").rstrip("/")
            auth = base64.b64encode(f"{public}:{secret}".encode()).decode()
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
                endpoint=f"{host}/api/public/otel/v1/traces",
                headers={"Authorization": f"Basic {auth}", "x-langfuse-ingestion-version": "4"},
            )))
        trace.set_tracer_provider(provider)
        _provider = provider
        atexit.register(flush)
        return provider.get_tracer("weather-workshop")


def flush():
    if _provider is not None:
        _provider.force_flush(timeout_millis=5000)
