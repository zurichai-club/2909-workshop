"""Run the golden set: every case in golden.json is checked with DeepEval metrics."""

import datetime
import json
import os
from pathlib import Path
import re
import sys

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"  # must be set before deepeval is imported
from deepeval.metrics import PatternMatchMetric, ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams

from weather_agent.agent import run_agent

GOLDEN_FILE = Path(__file__).parent / "golden.json"


def to_test_case(case, result):
    """Put what the agent did next to what the golden case expects."""
    tools_called = []
    for call in result.tool_calls:
        tools_called.append(ToolCall(name=call["name"], input_parameters=call["arguments"]))

    # No expected tool means the agent should ask a clarifying question instead.
    expected_tools = []
    if "expected_tool" in case:
        expected_date = datetime.date.today() + datetime.timedelta(days=case["date_offset"])
        expected_arguments = {"city": case["expected_city"], "date": expected_date.isoformat()}
        expected_tools.append(ToolCall(name=case["expected_tool"], input_parameters=expected_arguments))

    return LLMTestCase(input=case["input"], actual_output=result.answer,
                       tools_called=tools_called, expected_tools=expected_tools)


def run_golden_set():
    """Run every golden case and return how many failed."""
    cases = json.loads(GOLDEN_FILE.read_text())
    failures = 0
    for case in cases:
        result = run_agent(case["input"])
        test_case = to_test_case(case, result)
        failed = []

        # The right tool, called with exactly the right city and date.
        tool_metric = ToolCorrectnessMetric(evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
                                            should_exact_match=True)
        tool_metric.measure(test_case)
        if not tool_metric.success:
            failed.append("tool call")

        # The answer contains each expected text (regex: anything, the text, anything).
        for text in case["answer_contains"]:
            answer_metric = PatternMatchMetric(pattern=".*" + re.escape(text) + ".*", ignore_case=True)
            answer_metric.measure(test_case)
            if not answer_metric.success:
                failed.append(f"answer should contain {text!r}")

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
