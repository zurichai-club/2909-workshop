"""Show a flagged production run: everything you need to write a golden test for it.

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


row = find_batch_row(sys.argv[1])
call = row["tool_calls"][0]

print("question:   ", row["input"])
print("answer:     ", row["answer"])
print("flag:       ", row["flag"])
print("trace ID:   ", row["trace_id"])
print("tool call:  ", call["name"], json.dumps(call["arguments"]))
print("tool output:", json.dumps(call["output"], ensure_ascii=False))
