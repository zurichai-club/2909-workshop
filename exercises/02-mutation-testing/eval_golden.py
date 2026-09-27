"""Run the golden set: check every case in golden.json with plain Python."""

import datetime
import json
from pathlib import Path
import sys

from weather_agent.agent import SYSTEM_PROMPT, run_agent

GOLDEN_FILE = Path(__file__).parent / "golden.json"


def check(case, result):
    """Return the list of checks this run failed. An empty list means the case passed."""
    failed = []
    called = [call["name"] for call in result.tool_calls]
    expected_tool = case.get("expected_tool")

    if expected_tool is None:
        # No expected tool: the agent should ask a clarifying question instead.
        if called:
            failed.append("tool")
    elif called != [expected_tool]:
        failed.append("tool")
    else:
        arguments = result.tool_calls[0]["arguments"]
        expected_date = datetime.date.today() + datetime.timedelta(days=case["date_offset"])
        if arguments["city"] != case["expected_city"]:
            failed.append("city")
        if arguments["date"] != expected_date.isoformat():
            failed.append("date")

    for text in case["answer_contains"]:
        if text.lower() not in result.answer.lower():
            failed.append(f"answer should contain {text!r}")
    return failed


def run_golden_set(system_prompt=SYSTEM_PROMPT):
    """Run every golden case and return how many failed."""
    cases = json.loads(GOLDEN_FILE.read_text())
    failures = 0
    for case in cases:
        result = run_agent(case["input"], system_prompt=system_prompt)
        failed = check(case, result)
        if failed:
            failures += 1
            print(f"FAIL {case['id']}: {', '.join(failed)}")
        else:
            print(f"PASS {case['id']}")
    print(f"{len(cases) - failures}/{len(cases)} cases passed")
    return failures


if __name__ == "__main__":
    failures = run_golden_set()
    sys.exit(1 if failures else 0)
