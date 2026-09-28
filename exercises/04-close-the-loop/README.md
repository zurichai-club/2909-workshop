# Exercise 4 — turn a trace into a test

## Run

```bash
cd exercises/04-close-the-loop
uv run python production_batch.py
uv run python capture_failure.py p2
uv run pytest -v test_golden.py
uv run python mutate.py
```

## Task

1. Turn the flagged `p2` run into a test (see "Your turn" in `test_golden.py`).
2. Fix `mock_answer` in `agent.py` so the test passes.

<details>
<summary>Solution</summary>

The test is at the bottom of `test_golden.py`. In `mock_answer`, choose the
temperature text before building the answer:

```python
    if "fahrenheit" in question.lower():
        temperature = f"{forecast['temperature_max_c'] * 9 / 5 + 32:.1f} °F"
    else:
        temperature = f"{forecast['temperature_max_c']} °C"
```

Then use `{temperature}` in place of `{forecast['temperature_max_c']} °C`.

</details>
