"""Simulate production traffic: run messy questions and log each run to batch.jsonl."""

import json
from pathlib import Path

from weather_agent.agent import run_agent

BATCH_FILE = Path(__file__).parent / "batch.jsonl"

QUERIES = [
    ("p1", "Do I need an umbrella in Berlin tomorrow?"),
    ("p2", "Will it rain in Berlin tomorrow? Please answer in Fahrenheit."),
    ("p3", "Will it rain in Paris tomorrow?"),
    ("p4", "Will it rain in Berlin three weeks from now?"),
]

with BATCH_FILE.open("w") as out:
    for query_id, question in QUERIES:
        result = run_agent(question)

        # A cheap online check: the user asked for Fahrenheit, but the answer has no °F.
        flag = None
        if "Fahrenheit" in question and "°F" not in result.answer:
            flag = "unit_mismatch"

        row = {"query_id": query_id, "input": question, "answer": result.answer,
               "tool_calls": result.tool_calls, "trace_id": result.trace_id, "flag": flag}
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{query_id}: trace={result.trace_id} flag={flag or '-'}")

print(f"Batch index: {BATCH_FILE}")
