"""OpenTelemetry setup. You can treat this file as boilerplate.

Every span is appended to traces.jsonl in the exercise folder. When the Langfuse
keys from .env are set, spans are also sent to Langfuse.
"""

import base64
import json
import os
from pathlib import Path

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter, SpanExportResult

EXERCISE_DIR = Path(__file__).resolve().parents[1]


class JsonlExporter(SpanExporter):
    """Write each finished span as one line of JSON."""

    def __init__(self, path):
        self.path = Path(path)

    def export(self, spans):
        with self.path.open("a") as out:
            for span in spans:
                parent_span_id = None
                if span.parent:
                    parent_span_id = f"{span.parent.span_id:016x}"
                row = {
                    "trace_id": f"{span.context.trace_id:032x}",
                    "span_id": f"{span.context.span_id:016x}",
                    "parent_span_id": parent_span_id,
                    "name": span.name,
                    "attributes": dict(span.attributes),
                }
                out.write(json.dumps(row) + "\n")
        return SpanExportResult.SUCCESS


def langfuse_exporter(public_key, secret_key):
    host = os.getenv("LANGFUSE_HOST", "http://127.0.0.1:3000")
    auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    return OTLPSpanExporter(
        endpoint=f"{host}/api/public/otel/v1/traces",
        headers={"Authorization": f"Basic {auth}", "x-langfuse-ingestion-version": "4"},
    )


def create_provider():
    provider = TracerProvider(resource=Resource.create({"service.name": "workshop-weather-agent"}))
    trace_file = os.getenv("TRACE_FILE", str(EXERCISE_DIR / "traces.jsonl"))
    if trace_file:
        provider.add_span_processor(SimpleSpanProcessor(JsonlExporter(trace_file)))
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if public_key and secret_key:
        provider.add_span_processor(BatchSpanProcessor(langfuse_exporter(public_key, secret_key)))
    trace.set_tracer_provider(provider)
    return provider


provider = create_provider()
tracer = provider.get_tracer("weather-workshop")


def record(span, kind, input, output, model=None):
    """Save what Langfuse shows for a span: its kind, input and output."""
    span.set_attribute("langfuse.observation.type", kind)
    span.set_attribute("langfuse.observation.input", json.dumps(input, ensure_ascii=False))
    span.set_attribute("langfuse.observation.output", json.dumps(output, ensure_ascii=False))
    if model:
        span.set_attribute("langfuse.observation.model.name", model)


def trace_id_of(span):
    return f"{span.get_span_context().trace_id:032x}"


def flush():
    provider.force_flush()
