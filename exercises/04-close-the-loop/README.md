# Exercise 4 — turn a trace into a test

A failure found in production becomes a golden case, so it can never come back
unnoticed.

```bash
cd exercises/04-close-the-loop
uv run python production_batch.py
uv run python capture_failure.py p2
```

## Files

| File | What it is |
|---|---|
| `capture_failure.py` | Turns the flagged run from `batch.jsonl` into `captured_case.json` |
| `golden.json` | The golden set, including the `°C` check from exercise 2 |
| `eval_golden.py` | The golden set runner; replays `recorded_tool_output` when a case has one |
| `mutate.py` | The mutation test from exercise 2 |
| `weather_agent/agent.py` | The agent; `mock_answer` is what you will fix |

## Task

1. Read `captured_case.json`. It has the question, the expected tool call, the
   recorded tool output and the source trace ID.
2. Append it to the list in `golden.json` (keep the JSON valid).
3. Run `uv run python eval_golden.py`. The new case fails: the answer has no °F.
4. Fix `mock_answer` in `weather_agent/agent.py` so it converts Celsius to
   Fahrenheit when the user asks for it.
5. Run `eval_golden.py` and `mutate.py` again. Everything should pass, and the
   mutant should still be killed.

The trace ID explains why the case exists. The recorded tool output makes it
reproducible on any day.

<details>
<summary>Solution</summary>

In `mock_answer`, choose the temperature text before building the answer:

```python
    if "fahrenheit" in question.lower():
        temperature = f"{forecast['temperature_max_c'] * 9 / 5 + 32:.1f} °F"
    else:
        temperature = f"{forecast['temperature_max_c']} °C"
```

Then use `{temperature}` in place of `{forecast['temperature_max_c']} °C`.

</details>
