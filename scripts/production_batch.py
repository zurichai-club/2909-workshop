"""Run messy illustrative queries and write a batch index linked to traces."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "example_agent"))
from weather_agent.agent import build_agent  # noqa: E402

QUERIES = [
    ("p1", "Do I need an umbrella in Berlin tomorrow?"),
    ("p2", "Will it rain in Berlin tomorrow? Please answer in Fahrenheit."),
    ("p3", "Will it rain in Paris tomorrow?"),
    ("p4", "Will it rain in Berlin three weeks from now?"),
]


def main():
    agent = build_agent()
    path = ROOT / "batch.jsonl"
    with path.open("w") as out:
        for query_id, question in QUERIES:
            run = agent.run(question)
            issue = "unit_mismatch" if "Fahrenheit" in question and "°F" not in run.answer else None
            row = {"query_id": query_id, "input": question, "answer": run.answer,
                   "tool_calls": run.tool_calls, "trace_id": run.trace_id, "flag": issue}
            out.write(json.dumps(row) + "\n")
            print(f"{query_id}: trace={run.trace_id} flag={issue or '-'}")
    print(f"Batch index: {path}")


if __name__ == "__main__":
    main()
