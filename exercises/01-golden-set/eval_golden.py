"""Run the golden set: check every case in golden.json with plain Python.

    python eval_golden.py              # deterministic checks
    python eval_golden.py --deepeval   # also DeepEval's tool correctness report
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import sys

from weather_agent.agent import run_agent

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepeval", action="store_true", help="Also run DeepEval ToolCorrectness")
    args = parser.parse_args()

    cases = json.loads(GOLDEN_FILE.read_text())
    results = []
    failures = 0
    for case in cases:
        result = run_agent(case["input"])
        results.append(result)
        failed = check(case, result)
        if failed:
            failures += 1
            print(f"FAIL {case['id']}: {', '.join(failed)}")
        else:
            print(f"PASS {case['id']}")
    print(f"{len(cases) - failures}/{len(cases)} cases passed")

    if args.deepeval:
        deepeval_report(cases, results)
    return failures


def deepeval_report(cases, results):
    """Score tool choice with DeepEval. Note: it only sees tool names, not arguments."""
    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
    from deepeval import evaluate
    from deepeval.metrics import ToolCorrectnessMetric
    from deepeval.test_case import LLMTestCase, ToolCall

    test_cases = []
    for case, result in zip(cases, results):
        tools_called = [ToolCall(name=call["name"]) for call in result.tool_calls]
        expected_tools = []
        if "expected_tool" in case:
            expected_tools = [ToolCall(name=case["expected_tool"])]
        test_cases.append(LLMTestCase(input=case["input"], actual_output=result.answer,
                                      tools_called=tools_called, expected_tools=expected_tools))
    evaluate(test_cases=test_cases, metrics=[ToolCorrectnessMetric()])


if __name__ == "__main__":
    failures = main()
    sys.exit(1 if failures else 0)
