# Exercise 4 — turn a trace into a test

Checkpoint: `git checkout ex4-start`.

Run `uv run --env-file .env python scripts/capture_failure.py p2`. Review the
generated `captured_case.json`: it contains the input, expected tool call,
recorded tool output, and source trace ID. Append it to
`fixtures/golden.json`, keeping valid JSON.

Run `scripts/eval_golden.py`. The new case should fail because the response
omits Fahrenheit. Fix `_mock_answer` in `example_agent/weather_agent/agent.py`
to convert Celsius to Fahrenheit when the user asks for it. Run the golden set
and mutation script again. The solution branch shows one implementation:

```bash
git switch solutions
uv run --env-file .env python scripts/eval_golden.py
git switch main
```

The trace ID and recorded output preserve the reason this case exists, while
the fixture makes it reproducible on another day.
