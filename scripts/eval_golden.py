"""Deterministic golden checks, optionally reported with DeepEval ToolCorrectness."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "example_agent"))
from weather_agent.agent import build_agent  # noqa: E402


def evaluate_cases(*, mutation=None, deepeval=False, path=None):
    if deepeval:
        os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
    cases = json.loads(Path(path or ROOT / "fixtures" / "golden.json").read_text())
    agent = build_agent(mode="mock", fixture=True, mutation=mutation)
    failures = []
    deep_cases = []
    for case in cases:
        run = agent.run(case["input"], recorded_tool_output=case.get("recorded_tool_output"))
        actual = run.tool_calls
        wanted = case.get("expected_tool")
        checks = {
            "tool": [call["name"] for call in actual] == ([wanted] if wanted else []),
            "city": not wanted or (bool(actual) and actual[0]["arguments"].get("city") == case["expected_city"]),
            "date": not wanted or (bool(actual) and actual[0]["arguments"].get("date") == (date.today() + timedelta(days=case["date_offset"])).isoformat()),
            "answer": all(term.casefold() in run.answer.casefold() for term in case["answer_contains"]),
        }
        bad = [name for name, passed in checks.items() if not passed]
        print(f"{'PASS' if not bad else 'FAIL'} {case['id']}" + (f": {', '.join(bad)}" if bad else ""))
        if bad:
            failures.append(case["id"])
        if deepeval:
            from deepeval.test_case import LLMTestCase, ToolCall
            deep_cases.append(LLMTestCase(
                input=case["input"], actual_output=run.answer,
                tools_called=[ToolCall(name=call["name"]) for call in actual],
                expected_tools=[ToolCall(name=wanted)] if wanted else [],
            ))
    if deepeval:
        from deepeval import evaluate
        from deepeval.metrics import ToolCorrectnessMetric
        evaluate(test_cases=deep_cases, metrics=[ToolCorrectnessMetric()])
    print(f"{len(cases) - len(failures)}/{len(cases)} deterministic cases passed")
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepeval", action="store_true", help="Also run DeepEval ToolCorrectness")
    parser.add_argument("--case-file", type=Path)
    args = parser.parse_args()
    raise SystemExit(bool(evaluate_cases(deepeval=args.deepeval, path=args.case_file)))
