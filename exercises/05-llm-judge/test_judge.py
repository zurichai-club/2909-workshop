"""LLM as a judge: DeepEval's built-in judge metrics, with an OpenAI model as the judge.

The golden set checks what can be written as an exact rule. These metrics ask an
LLM instead: did the agent do the job, is the answer on topic, is it backed by
the tool output, were the tool arguments sensible? Run them with pytest:

    pytest -v test_judge.py
"""

import json
import os

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"  # must be set before deepeval is imported
from deepeval import assert_test
from deepeval.metrics import (AnswerRelevancyMetric, ArgumentCorrectnessMetric, FaithfulnessMetric,
                              TaskCompletionMetric)
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase, ToolCall

from agent import run_agent


def agent_test_case(question):
    """Run the agent on one question. The judge sees the tool calls, and their outputs as context."""
    result = run_agent(question)
    tools_called = []
    tool_outputs = []
    for call in result.tool_calls:
        tools_called.append(ToolCall(name=call["name"], input_parameters=call["arguments"], output=call["output"]))
        tool_outputs.append(json.dumps(call["output"]))
    return LLMTestCase(input=question, actual_output=result.answer,
                       tools_called=tools_called, retrieval_context=tool_outputs)


# The judge model. DeepEval reads the key from the OPENAI_API_KEY environment variable.
# gpt-6-luna only accepts the default temperature (1).
JUDGE = GPTModel(model="gpt-6-luna", temperature=1)

# What "done" means for this agent. Without it, the judge guesses the task from each
# question and fails correct refusals ("too far ahead") and clarifying questions ("which Paris?").
# See the comparison of judge setups at the bottom of this file.
TASK = ("Answer the user's weather question using the forecast tool, in the units the user asked for. "
        "If the city is ambiguous or missing, asking a clarifying question completes the task. "
        "If the forecast is unavailable, clearly explaining why completes the task.")

ANSWER_RELEVANCY = AnswerRelevancyMetric(model=JUDGE)          # does the answer address the question?
FAITHFULNESS = FaithfulnessMetric(model=JUDGE)                 # is every claim backed by the tool output?
ARGUMENT_CORRECTNESS = ArgumentCorrectnessMetric(model=JUDGE)  # do the tool arguments fit the question?



def task_completion(question):
    """Did the agent do the job? This judge sees only the task and the answer, not the
    question, so the question goes into the task."""
    return TaskCompletionMetric(model=JUDGE, task=TASK + " The user asked: " + question, threshold=0.8)


def judges(question):
    """All four judges for one question."""
    return [task_completion(question), ANSWER_RELEVANCY, FAITHFULNESS, ARGUMENT_CORRECTNESS]


# ----------------------------------------------------------------------------
def test_berlin_rain():
    """A good answer."""
    test_case = agent_test_case("Do I need an umbrella in Berlin tomorrow?")
    assert_test(test_case, judges(test_case.input))


# ----------------------------------------------------------------------------
def test_beyond_horizon():
    """A correct refusal: the date is outside the forecast horizon."""
    test_case = agent_test_case("Will it rain in Berlin three weeks from now?")
    assert_test(test_case, judges(test_case.input))


# ----------------------------------------------------------------------------
def test_ambiguous_paris():
    """A correct clarifying question: there is more than one Paris."""
    test_case = agent_test_case("Will it rain in Paris tomorrow?")
    assert_test(test_case, judges(test_case.input))


# ----------------------------------------------------------------------------
def test_fahrenheit():
    """The user asked for Fahrenheit, but this agent answers in °C (the bug from exercise 3),
    so this test fails until the agent is fixed. Which judge catches it?"""
    test_case = agent_test_case("Will it rain in Berlin tomorrow? Please answer in Fahrenheit.")
    assert_test(test_case, judges(test_case.input))


# ----------------------------------------------------------------------------
def test_judge_catches_made_up_answer():
    """Test the judge itself: replace the agent's answer with a made-up one.
    The tool said 70% rain and 14 °C, so the faithfulness judge must fail it."""
    test_case = agent_test_case("Do I need an umbrella in Berlin tomorrow?")
    test_case.actual_output = "No umbrella needed in Berlin tomorrow: it will be sunny with a high of 25 °C."
    FAITHFULNESS.measure(test_case)
    assert not FAITHFULNESS.success


# ----------------------------------------------------------------------------
# Your turn: a judge for your own rule, written in plain English, with DeepEval's GEval.
#   - rule: the temperature in the answer is in the units the user asked for (Celsius by default)
#   - GEval(name=..., criteria=..., evaluation_params=[...], model=JUDGE)
#     evaluation_params says what the judge reads: SingleTurnParams.INPUT and SingleTurnParams.ACTUAL_OUTPUT
#   - write two tests that use only this judge: the Berlin question, and the Fahrenheit question.
#     Does it catch the Fahrenheit answer on its own?


# ----------------------------------------------------------------------------
# Judge setups we tried for task completion, on four of the cases above:
#
# | Question                                  | Right | Qwen 7B (local) | gpt-6-luna, no TASK | gpt-6-luna, TASK only | gpt-5-nano, TASK only | gpt-6-luna, TASK + question |
# |                                           |       |                 |                     | (threshold 0.9)       | (threshold 0.9)       | (threshold 0.8)             |
# |-------------------------------------------|-------|-----------------|---------------------|-----------------------|-----------------------|-----------------------------|
# | Umbrella in Berlin tomorrow               | pass  | ok    0.95      | ok    1.00          | FLAKY 0.75-1.00       | ok    1.00            | ok   1.00 (4 of 4 runs)     |
# | Berlin in three weeks (correct refusal)   | pass  | WRONG 0.00      | WRONG 0.20          | ok    1.00            | ok    0.95            | ok   1.00                   |
# | Asked for Fahrenheit, answer only says °C | fail  | WRONG passed    | WRONG passed 0.70   | ok    fails 0.60-0.70 | WRONG passed 1.00     | ok   fails 0.70-0.75        |
# | Paris (correct clarifying question)       | pass  | WRONG 0.00      | WRONG 0.10          | ok    1.00            | ok    1.00            | ok   0.85-1.00              |
#
# Lessons: the task must say what "done" means. The judge sees only what you give it: with a
# fixed TASK it never sees the question, so it cannot tell whether Celsius was right, and the
# score of a good answer wanders. The threshold decides whether a half-right answer passes, and
# the judge model matters: the cheapest model claims °C was requested. LLM judges are also not
# deterministic, so look at several runs before trusting a threshold.


# ----------------------------------------------------------------------------
# Solution (uncomment to run it):
#
# from deepeval.metrics import GEval
# from deepeval.test_case import SingleTurnParams
#
# UNITS = GEval(
#     name="Units",
#     criteria="The temperature in the answer is in the units the user asked for (Celsius if not stated).",
#     evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
#     model=JUDGE,
# )
#
# def test_units_celsius():
#     test_case = agent_test_case("Do I need an umbrella in Berlin tomorrow?")
#     assert_test(test_case, [UNITS])
#
# def test_units_fahrenheit():
#     test_case = agent_test_case("Will it rain in Berlin tomorrow? Please answer in Fahrenheit.")
#     assert_test(test_case, [UNITS])
