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
