# Exercise 1 — a golden set

A golden set is a small list of questions with the behaviour you expect. Run it
after every change to the agent.

```bash
cd exercises/01-golden-set
uv run python eval_golden.py --deepeval
```

## Files

| File | What it is |
|---|---|
| `golden.json` | The three test cases |
| `eval_golden.py` | Runs each case and checks the tool, its arguments and the answer |
| `weather_agent/agent.py` | The agent. `mock_plan` and `mock_answer` stand in for the LLM |
| `weather_agent/tools.py` | The `get_forecast` tool (recorded data in `weather_agent/forecast.json`) |
| `ask.py` | Ask one question: `uv run python ask.py "Will it rain in Oslo today?"` |

Each case in `golden.json` lists the expected tool, `expected_city`, a
`date_offset` in days from today (so the case still works next week), and words
the answer must contain (`answer_contains`).

## Task

1. Read `golden.json` and the `check` function in `eval_golden.py`.
2. Add three cases of your own:
   - an ambiguous city (for example, "Paris"),
   - a missing location,
   - a date in the past.

   When the agent should ask a clarifying question, leave out `expected_tool`,
   `expected_city` and `date_offset`. Put words from the clarification in
   `answer_contains`.
3. Run the suite again. Compare our checks with DeepEval's tool correctness
   report. DeepEval only sees tool *names*. Our checks also cover the arguments
   and the answer.

The mock model only knows Berlin, Zurich and Oslo. For other cities, use a real
model (`ask.py --model openai`, see the main README) or extend `mock_plan`.

<details>
<summary>Example solution</summary>

```json
{"id": "ambiguous-paris", "input": "Will it rain in Paris tomorrow?", "answer_contains": ["Which Paris"]},
{"id": "missing-city", "input": "Will it rain tomorrow?", "answer_contains": ["which city"]},
{"id": "past-date", "input": "Did it rain in Berlin yesterday?", "expected_tool": "get_forecast", "expected_city": "Berlin, Germany", "date_offset": -1, "answer_contains": ["outside the available 16-day forecast horizon"]}
```

The mock model does not understand "yesterday" yet, so the past-date case fails
until you teach `mock_plan` in `weather_agent/agent.py`:

```python
    elif "yesterday" in q:
        day = today - datetime.timedelta(days=1)
```

</details>
