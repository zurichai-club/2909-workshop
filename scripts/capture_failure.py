"""Promote a triaged batch failure into a reproducible golden case."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_id", help="Batch query id, e.g. p2")
    args = parser.parse_args()
    rows = [json.loads(line) for line in (ROOT / "batch.jsonl").read_text().splitlines()]
    row = next((item for item in rows if item["query_id"] == args.query_id), None)
    if not row:
        raise SystemExit("No such query id")
    if not row["flag"]:
        raise SystemExit("This query is not flagged; triage it before capture")
    call = row["tool_calls"][0]
    case = {"id": f"production-{args.query_id}", "input": row["input"],
            "expected_tool": call["name"], "expected_city": call["arguments"]["city"],
            "date_offset": 1, "answer_contains": ["°F"],
            "source_trace_id": row["trace_id"], "recorded_tool_output": call["output"]}
    destination = ROOT / "captured_case.json"
    destination.write_text(json.dumps(case, indent=2) + "\n")
    print(f"Review {destination}, then append the case to fixtures/golden.json")


if __name__ == "__main__":
    main()
