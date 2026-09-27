"""Turn a flagged production run into a golden case.

    python capture_failure.py p2
"""

import json
from pathlib import Path
import sys

HERE = Path(__file__).parent


def find_batch_row(query_id):
    for line in (HERE / "batch.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["query_id"] == query_id:
            return row
    sys.exit(f"No run with query id {query_id} in batch.jsonl")


query_id = sys.argv[1]
row = find_batch_row(query_id)
if not row["flag"]:
    sys.exit("This run is not flagged; triage it before capturing it")

call = row["tool_calls"][0]
case = {
    "id": f"production-{query_id}",
    "input": row["input"],
    "expected_tool": call["name"],
    "expected_city": call["arguments"]["city"],
    "date_offset": 1,
    "answer_contains": ["°F"],
    "source_trace_id": row["trace_id"],        # why this case exists
    "recorded_tool_output": call["output"],    # replay the exact data the agent saw
}

destination = HERE / "captured_case.json"
destination.write_text(json.dumps(case, indent=2, ensure_ascii=False) + "\n")
print(f"Review {destination.name}, then append the case to golden.json")
