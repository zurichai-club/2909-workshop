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

from agent import run_agent

# The mock model by default. AGENT_MODEL=openai runs the same tests against a real model (needs OPENAI_API_KEY).
MODEL = os.getenv("AGENT_MODEL", "mock")


def days_from_today(days):
    """Dates relative to today, so the cases still work next week."""
    return (datetime.date.today() + datetime.timedelta(days=days)).isoformat()


def agent_test_case(question, expected_tools, history=[]):
    """Run the agent on one question and put the result next to what we expect."""
    result = run_agent(question, model=MODEL, history=history)
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
# Your turn: a follow-up message. The agent asked "Which Paris?", and the user answers.
#   - question: "Paris, France"
#   - history: the earlier messages, as a list of {"role": "user" or "assistant", "content": "..."}
#   - expect get_forecast for "Paris, France" tomorrow: the agent must remember "tomorrow"
#     from the first message.
# (DeepEval also has ConversationalTestCase, to judge a whole conversation with an LLM;
# see the multi-turn metrics at the bottom of this file.)


# ----------------------------------------------------------------------------
# Other DeepEval metrics (deepeval 3.9) you could use for this agent
#
# Deterministic (no LLM, no API key):
#   ToolCorrectnessMetric      right tool with the right arguments         (used above)
#   PatternMatchMetric         the answer matches a regex                  (used above)
#   ExactMatchMetric           the answer is exactly an expected text, e.g. a fixed clarifying question
#   JsonCorrectnessMetric      the output matches a Pydantic schema, e.g. a structured forecast
#
# LLM-judged (need a judge model):
#   TaskCompletionMetric       did the agent complete the task?            (exercise 5)
#   GEval                      your own rubric in plain English, e.g. "uses the units the user asked for"
#   FaithfulnessMetric         every claim is supported by the context (pass the tool output as retrieval_context)
#   HallucinationMetric        the answer does not contradict the given context
#   AnswerRelevancyMetric      the answer addresses the question (yes/no to "do I need an umbrella?")
#   ArgumentCorrectnessMetric  the tool arguments make sense for the question, without exact expected values
#   ToolUseMetric              the agent picks and uses tools sensibly
#   StepEfficiencyMetric       no unnecessary steps (reads a DeepEval @observe trace)
#   PlanAdherenceMetric        the agent follows its own plan (reads a trace)
#   PlanQualityMetric          the agent's plan is sensible (reads a trace)
#   GoalAccuracyMetric         the agent reached the user's goal
#   PromptAlignmentMetric      the answer follows instructions from the system prompt
#   ToxicityMetric, BiasMetric, PIILeakageMetric, MisuseMetric, NonAdviceMetric   safety checks
#   RoleAdherenceMetric, TopicAdherenceMetric                                     stays a weather assistant
#   SummarizationMetric                                                           summary quality
#   ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric  retrieval (RAG) quality
#
# Multi-turn (ConversationalTestCase with Turns, LLM-judged):
#   ConversationCompletenessMetric, KnowledgeRetentionMetric, TurnRelevancyMetric,
#   TurnFaithfulnessMetric, TurnContextualPrecisionMetric, TurnContextualRecallMetric,
#   TurnContextualRelevancyMetric, ConversationalGEval
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Solution (uncomment to run it):
#
# def test_follow_up():
#     """The user answers "Which Paris?"; the agent must remember "tomorrow"."""
#     test_case = agent_test_case(
#         "Paris, France",
#         expected_tools=[ToolCall(name="get_forecast",
#                                  input_parameters={"city": "Paris, France", "date": days_from_today(1)})],
#         history=[{"role": "user", "content": "Will it rain in Paris tomorrow?"},
#                  {"role": "assistant", "content": "Which Paris do you mean? Please include the country or state."}],
#     )
#     assert_test(test_case, [TOOL_CORRECTNESS, contains("Paris")])
