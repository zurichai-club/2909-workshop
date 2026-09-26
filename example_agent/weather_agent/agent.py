"""Adapted from the repository's example weather agent for the workshop.

The source example used LangGraph, a local proxy, and three fake tools. This
copy keeps the simple ask/build_agent interface while using one real forecast
tool, an optional recorded response, and explicit per-step OTEL spans.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
import json
import os
import re
from typing import Any

from opentelemetry import trace
from openai import OpenAI

from .tools import TOOL_NAME, TOOL_SCHEMA, get_forecast
from .tracing import configure, flush

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
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    trace_id: str = ""
    model: str = ""


class Agent:
    def __init__(self, *, mode: str | None = None, fixture: bool | None = None, mutation: str | None = None):
        self.mode = mode or os.getenv("MODEL_MODE", "mock")
        self.fixture = (os.getenv("WEATHER_FIXTURE", "1") == "1") if fixture is None else fixture
        self.mutation = mutation
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.tracer = configure()
        if self.mode not in {"mock", "openai"}:
            raise ValueError("MODEL_MODE must be mock or openai")
        if self.mode == "openai":
            if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_BASE_URL"):
                raise RuntimeError("Set OPENAI_API_KEY or OPENAI_BASE_URL for MODEL_MODE=openai")
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY") or "local-proxy",
                base_url=os.getenv("OPENAI_BASE_URL") or None,
                timeout=60,
            )

    def run(self, question: str, *, recorded_tool_output: dict | None = None) -> RunResult:
        prompt = SYSTEM_PROMPT.format(today=date.today().isoformat())
        if self.mutation == "remove_units":
            prompt = prompt.replace("Always include units (°C, mm and %) when reporting numbers.\n", "")
        with self.tracer.start_as_current_span("weather_agent.run") as root:
            root.set_attribute("langfuse.observation.type", "agent")
            root.set_attribute("langfuse.trace.name", "weather-workshop")
            root.set_attribute("langfuse.observation.input", json.dumps(question))
            trace_id = f"{root.get_span_context().trace_id:032x}"
            if self.mode == "mock":
                result = self._run_mock(question, prompt, recorded_tool_output)
            else:
                result = self._run_openai(question, prompt)
            result.trace_id = trace_id
            result.model = "recorded-model" if self.mode == "mock" else self.model
            root.set_attribute("langfuse.observation.output", json.dumps(result.answer))
            root.set_attribute("workshop.tool_calls", len(result.tool_calls))
        flush()
        return result

    def _model_span(self, name: str):
        return self.tracer.start_as_current_span(name)

    def _run_mock(self, question: str, prompt: str, recorded_tool_output: dict | None) -> RunResult:
        with self._model_span("chat plan") as span:
            span.set_attribute("langfuse.observation.type", "generation")
            span.set_attribute("langfuse.observation.model.name", "recorded-model")
            span.set_attribute("langfuse.observation.input", json.dumps({"system": prompt, "user": question}))
            plan = self._mock_plan(question)
            span.set_attribute("langfuse.observation.output", json.dumps(plan))
        if "answer" in plan:
            return RunResult(question, plan["answer"])
        args = plan["arguments"]
        data = self._call_tool(args, recorded_tool_output)
        with self._model_span("chat answer") as span:
            span.set_attribute("langfuse.observation.type", "generation")
            span.set_attribute("langfuse.observation.model.name", "recorded-model")
            span.set_attribute("langfuse.observation.input", json.dumps(data))
            answer = self._mock_answer(question, data)
            if self.mutation == "remove_units":
                answer = answer.replace(" °C", "").replace(" mm", "").replace("%", "")
            span.set_attribute("langfuse.observation.output", json.dumps(answer))
        return RunResult(question, answer, [{"name": TOOL_NAME, "arguments": args, "output": data}])

    def _mock_plan(self, question: str) -> dict:
        q = question.casefold()
        if "paris" in q and "france" not in q and "texas" not in q:
            return {"answer": "Which Paris do you mean? Please include the country or state."}
        locations = {"berlin": "Berlin, Germany", "zurich": "Zurich, Switzerland", "oslo": "Oslo, Norway"}
        city = next((value for key, value in locations.items() if key in q), None)
        if not city:
            return {"answer": "Please tell me which city you mean."}
        if "three weeks" in q or "3 weeks" in q:
            target = date.today() + timedelta(days=21)
        elif "tomorrow" in q:
            target = date.today() + timedelta(days=1)
        elif "today" in q:
            target = date.today()
        else:
            match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", q)
            if not match:
                return {"answer": "Please give a date (for example, tomorrow)."}
            target = date.fromisoformat(match.group())
        return {"arguments": {"city": city, "date": target.isoformat()}}

    def _mock_answer(self, question: str, data: dict) -> str:
        if "error" in data:
            return f"I can't provide that forecast: {data['error']}."
        rain = data["precipitation_probability_pct"] >= 50
        lead = "Yes, take an umbrella" if rain else "An umbrella is probably unnecessary"
        return (f"{lead} in {data['city']} on {data['date']}: "
                f"{data['precipitation_probability_pct']}% chance of precipitation, "
                f"{data['precipitation_mm']} mm expected, and a high of "
                f"{data['temperature_max_c']} °C. ({data['source']})")

    def _call_tool(self, args: dict, recorded_tool_output: dict | None = None) -> dict:
        with self.tracer.start_as_current_span(f"execute_tool {TOOL_NAME}") as span:
            span.set_attribute("langfuse.observation.type", "tool")
            span.set_attribute("gen_ai.operation.name", "execute_tool")
            span.set_attribute("gen_ai.tool.name", TOOL_NAME)
            span.set_attribute("langfuse.observation.input", json.dumps(args))
            data = (dict(recorded_tool_output, date=args["date"])
                    if recorded_tool_output is not None else get_forecast(**args, fixture=self.fixture))
            span.set_attribute("langfuse.observation.output", json.dumps(data))
            return data

    def _run_openai(self, question: str, prompt: str) -> RunResult:
        messages: list[dict] = [{"role": "system", "content": prompt}, {"role": "user", "content": question}]
        called: list[dict] = []
        for step in range(3):
            with self._model_span("chat model") as span:
                span.set_attribute("langfuse.observation.type", "generation")
                span.set_attribute("langfuse.observation.model.name", self.model)
                span.set_attribute("langfuse.observation.input", json.dumps(messages))
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, tools=[TOOL_SCHEMA], tool_choice="auto", temperature=0,
                )
                msg = response.choices[0].message
                span.set_attribute("langfuse.observation.output", msg.model_dump_json())
                if response.usage:
                    span.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
                    span.set_attribute("gen_ai.usage.output_tokens", response.usage.completion_tokens)
            if not msg.tool_calls:
                return RunResult(question, msg.content or "No answer returned.", called)
            messages.append(msg.model_dump(exclude_none=True))
            for call in msg.tool_calls:
                try:
                    args = json.loads(call.function.arguments)
                    if call.function.name != TOOL_NAME or set(args) != {"city", "date"}:
                        raise ValueError("invalid tool call")
                    data = self._call_tool(args)
                except (ValueError, TypeError) as exc:
                    args, data = {}, {"error": str(exc)}
                called.append({"name": call.function.name, "arguments": args, "output": data})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(data)})
        return RunResult(question, "I could not finish this request within the tool-call limit.", called)


def build_agent(**kwargs) -> Agent:
    return Agent(**kwargs)


def ask(agent: Agent, question: str, thread_id: str = "default") -> str:
    return agent.run(question).answer
