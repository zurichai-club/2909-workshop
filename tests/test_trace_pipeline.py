"""The workshop's key promise: one run yields a connected agent/model/tool trace."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TracePipelineTest(unittest.TestCase):
    def test_recorded_run_has_connected_spans_and_content(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_file = Path(directory) / "traces.jsonl"
            env = dict(os.environ, TRACE_FILE=str(trace_file))
            env.pop("LANGFUSE_PUBLIC_KEY", None)
            env.pop("LANGFUSE_SECRET_KEY", None)
            completed = subprocess.run(
                [str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "example_agent" / "cli.py"),
                 "Do I need an umbrella in Berlin tomorrow?"],
                cwd=ROOT, env=env, capture_output=True, text=True, check=True,
            )
            spans = [json.loads(line) for line in trace_file.read_text().splitlines()]
            self.assertEqual(len(spans), 4)
            self.assertEqual(len({span["trace_id"] for span in spans}), 1)
            self.assertIn(f"trace_id={spans[0]['trace_id']}", completed.stdout)
            root = next(span for span in spans if span["parent_span_id"] is None)
            self.assertEqual(root["name"], "weather_agent.run")
            self.assertEqual({span["parent_span_id"] for span in spans if span is not root}, {root["span_id"]})
            types = {span["attributes"]["langfuse.observation.type"] for span in spans}
            self.assertEqual(types, {"agent", "generation", "tool"})
            tool = next(span for span in spans if span["name"].startswith("execute_tool"))
            self.assertIn("Berlin, Germany", tool["attributes"]["langfuse.observation.input"])
            self.assertIn("precipitation_probability_pct", tool["attributes"]["langfuse.observation.output"])


if __name__ == "__main__":
    unittest.main()
