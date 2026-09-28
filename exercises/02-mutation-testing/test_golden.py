"""The golden set: questions with the behaviour we expect, checked with DeepEval metrics.

Each test_ function is one case. Run them with pytest:

    pytest -v test_golden.py
"""

import datetime
import os
import re

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"  # must be set before deepeval is imported
from deepeval import assert_test
from deepeval.metrics import PatternMatchMetric, ToolCorrectnessMetric
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams

from agent import SYSTEM_PROMPT, run_agent

# mutate.py runs this file again with a broken prompt in the SYSTEM_PROMPT environment variable.
PROMPT = os.getenv("SYSTEM_PROMPT", SYSTEM_PROMPT)


def days_from_today(days):
    """Dates relative to today, so the cases still work next week."""
    return (datetime.date.today() + datetime.timedelta(days=days)).isoformat()


def agent_test_case(question, expected_tools):
    """Run the agent on one question and put the result next to what we expect."""
    result = run_agent(question, system_prompt=PROMPT)
    tools_called = []
    for call in result.tool_calls:
        tools_called.append(ToolCall(name=call["name"], input_parameters=call["arguments"]))
    return LLMTestCase(input=question, actual_output=result.answer,
                       tools_called=tools_called, expected_tools=expected_tools)


# The right tool, called with exactly the right arguments (or no tool when none is expected).
# This check never calls an LLM, but DeepEval still wants a model object, so it gets one with a dummy key.
TOOL_CORRECTNESS = ToolCorrectnessMetric(evaluation_params=[ToolCallParams.INPUT_PARAMETERS], should_exact_match=True,
                                         model=GPTModel(api_key="not-used"))


def contains(text):
    """The answer contains this text (regex: anything, the text, anything)."""
    return PatternMatchMetric(pattern=".*" + re.escape(text) + ".*", ignore_case=True)


# ----------------------------------------------------------------------------
def test_berlin_rain():
    """Rain is likely, so the agent should recommend an umbrella."""
    test_case = agent_test_case(
        "Do I need an umbrella in Berlin tomorrow?",
        expected_tools=[ToolCall(name="get_forecast",
                                 input_parameters={"city": "Berlin, Germany", "date": days_from_today(1)})],
    )
    assert_test(test_case, [TOOL_CORRECTNESS, contains("Berlin"), contains("take an umbrella")])


# ----------------------------------------------------------------------------
def test_beyond_horizon():
    """The date is too far ahead; the tool returns an error the agent must explain."""
    test_case = agent_test_case(
        "Will it rain in Berlin three weeks from now?",
        expected_tools=[ToolCall(name="get_forecast",
                                 input_parameters={"city": "Berlin, Germany", "date": days_from_today(21)})],
    )
    assert_test(test_case, [TOOL_CORRECTNESS, contains("outside the available 16-day forecast horizon")])


# ----------------------------------------------------------------------------
# Your turn: run mutate.py. The mutant (no units rule in the prompt) survives, because no
# test checks for units. Add a check to one of the tests above so the mutant is killed.


# ----------------------------------------------------------------------------
# Solution: in test_berlin_rain, also check for the units.
#
#     assert_test(test_case, [TOOL_CORRECTNESS, contains("Berlin"), contains("take an umbrella"), contains("°C")])
