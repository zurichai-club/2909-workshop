"""A weather agent with one tool.

The default "mock" model is a few hand-written rules that behave like a
tool-calling LLM, so every run is repeatable and needs no API key.
Use model="openai" to run the same loop with a real LLM.
"""

import datetime
import json
import os
import re
from dataclasses import dataclass

from openai import OpenAI

from workshop_support import tracing
from workshop_support.tools import TOOL_SCHEMA, get_forecast

SYSTEM_PROMPT = """You are a weather assistant. Today is {today}.
You have one tool, get_forecast, for dates from today through 15 days ahead.
Ask which city when a place is missing or ambiguous. Ask for a date when missing.
Never invent a forecast. Explain tool errors plainly.
Always include units (°C, mm and %) when reporting numbers.
Use only facts returned by the tool. Give a clear answer to the user's question.
"""


@dataclass
class RunResult:
    question: str
    answer: str
    tool_calls: list
    trace_id: str


def run_agent(question, model="mock", live=False):
    """Answer one question. Returns the answer, the tool calls made and the trace ID."""
    prompt = SYSTEM_PROMPT.format(today=datetime.date.today().isoformat())
    with tracing.tracer.start_as_current_span("weather_agent.run") as root:
        root.set_attribute("langfuse.trace.name", "weather-workshop")
        if model == "mock":
            answer, tool_calls = run_mock_model(prompt, question, live)
        else:
            answer, tool_calls = run_openai_model(prompt, question, live)
        tracing.record(root, "agent", question, answer)
        trace_id = tracing.trace_id_of(root)
    tracing.flush()
    return RunResult(question, answer, tool_calls, trace_id)


def call_tool(arguments, live):
    with tracing.tracer.start_as_current_span("execute_tool get_forecast") as span:
        forecast = get_forecast(arguments.get("city", ""), arguments.get("date", ""), live)
        tracing.record(span, "tool", arguments, forecast)
    return forecast


# --- The mock model -----------------------------------------------------------

def run_mock_model(prompt, question, live):
    """Plan -> call the tool -> write the answer, like a real tool-calling LLM."""
    with tracing.tracer.start_as_current_span("chat plan") as span:
        plan = mock_plan(question)
        tracing.record(span, "generation", {"system": prompt, "user": question}, plan, model="recorded-model")

    if "answer" in plan:
        # The model asks a clarifying question instead of calling the tool.
        return plan["answer"], []

    forecast = call_tool(plan["arguments"], live)

    with tracing.tracer.start_as_current_span("chat answer") as span:
        answer = mock_answer(question, forecast)
        tracing.record(span, "generation", forecast, answer, model="recorded-model")

    tool_calls = [{"name": "get_forecast", "arguments": plan["arguments"], "output": forecast}]
    return answer, tool_calls


KNOWN_CITIES = {"berlin": "Berlin, Germany", "zurich": "Zurich, Switzerland", "oslo": "Oslo, Norway"}


def mock_plan(question):
    """Decide what to do: return a tool call, or an answer asking for clarification."""
    q = question.lower()
    if "paris" in q and "france" not in q and "texas" not in q:
        return {"answer": "Which Paris do you mean? Please include the country or state."}

    city = None
    for name in KNOWN_CITIES:
        if name in q:
            city = KNOWN_CITIES[name]
            break
    if city is None:
        return {"answer": "Please tell me which city you mean."}

    today = datetime.date.today()
    if "three weeks" in q or "3 weeks" in q:
        day = today + datetime.timedelta(days=21)
    elif "tomorrow" in q:
        day = today + datetime.timedelta(days=1)
    elif "yesterday" in q:
        day = today - datetime.timedelta(days=1)
    elif "today" in q:
        day = today
    else:
        match = re.search(r"\d{4}-\d{2}-\d{2}", q)
        if match is None:
            return {"answer": "Please give a date (for example, tomorrow)."}
        day = datetime.date.fromisoformat(match.group())

    return {"arguments": {"city": city, "date": day.isoformat()}}


def mock_answer(question, forecast):
    """Write the final answer from the tool output."""
    if "error" in forecast:
        return f"I can't provide that forecast: {forecast['error']}."

    if forecast["precipitation_probability_pct"] >= 50:
        lead = "Yes, take an umbrella"
    else:
        lead = "An umbrella is probably unnecessary"
    return (f"{lead} in {forecast['city']} on {forecast['date']}: "
            f"{forecast['precipitation_probability_pct']}% chance of precipitation, "
            f"{forecast['precipitation_mm']} mm expected, and a high of "
            f"{forecast['temperature_max_c']} °C. ({forecast['source']})")


# --- A real LLM (OpenAI, or any compatible gateway with tool calling) ---------

def run_openai_model(prompt, question, live):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "local-proxy"), base_url=os.getenv("OPENAI_BASE_URL"))
    model_name = os.getenv("OPENAI_MODEL", "gpt-6-luna")  # the same model as the judge in exercise 5
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": question}]
    tool_calls = []

    for step in range(3):
        with tracing.tracer.start_as_current_span("chat model") as span:
            response = client.chat.completions.create(
                model=model_name, messages=messages, tools=[TOOL_SCHEMA],
                reasoning_effort="none")  # gpt-6-luna needs this for tool calls
            message = response.choices[0].message
            tracing.record(span, "generation", messages, message.model_dump(), model=model_name)

        if not message.tool_calls:
            return message.content or "No answer returned.", tool_calls

        messages.append(message.model_dump(exclude_none=True))
        for call in message.tool_calls:
            arguments = json.loads(call.function.arguments)
            forecast = call_tool(arguments, live)
            tool_calls.append({"name": call.function.name, "arguments": arguments, "output": forecast})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(forecast)})

    return "I could not finish this request within the tool-call limit.", tool_calls
