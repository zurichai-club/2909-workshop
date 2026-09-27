"""Verify the actual authenticated OTLP request without requiring Docker."""

import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
from pathlib import Path
import subprocess
from threading import Thread
import unittest

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

ROOT = Path(__file__).resolve().parents[1]


class Receiver(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.requests.append((self.path, dict(self.headers), body))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


class LangfuseExportTest(unittest.TestCase):
    def test_authenticated_otlp_export_contains_entire_run(self):
        Receiver.requests = []
        try:
            server = HTTPServer(("127.0.0.1", 0), Receiver)
        except PermissionError:
            self.skipTest("This sandbox does not allow binding a local test receiver")
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            env = dict(os.environ,
                       LANGFUSE_HOST=f"http://127.0.0.1:{server.server_port}",
                       LANGFUSE_PUBLIC_KEY="pk-lf-test", LANGFUSE_SECRET_KEY="sk-lf-test",
                       TRACE_FILE="")
            subprocess.run(
                [str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "exercises" / "03-observability" / "ask.py"),
                 "Do I need an umbrella in Berlin tomorrow?"],
                cwd=ROOT, env=env, capture_output=True, text=True, check=True,
            )
        finally:
            server.shutdown()
            thread.join()
            server.server_close()
        self.assertTrue(Receiver.requests)
        path, headers, body = Receiver.requests[0]
        headers = {key.lower(): value for key, value in headers.items()}
        self.assertEqual(path, "/api/public/otel/v1/traces")
        auth = base64.b64encode(b"pk-lf-test:sk-lf-test").decode()
        self.assertEqual(headers["authorization"], f"Basic {auth}")
        self.assertEqual(headers["x-langfuse-ingestion-version"], "4")
        request = ExportTraceServiceRequest()
        request.ParseFromString(body)
        spans = [span for resource in request.resource_spans
                 for scope in resource.scope_spans for span in scope.spans]
        self.assertEqual({span.name for span in spans},
                         {"weather_agent.run", "chat plan", "chat answer", "execute_tool get_forecast"})


if __name__ == "__main__":
    unittest.main()
