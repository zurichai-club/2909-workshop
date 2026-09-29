# Exercise 1 — a golden set

## Run

```bash
cd exercises/01-golden-set
uv run pytest -v test_golden.py                     # with the mock model: no API key
AGENT_MODEL=openai uv run pytest -v test_golden.py  # with a real model: needs OPENAI_API_KEY
```

## Task

1. Write a test for a follow-up message: the user answers "Which Paris?" with
   "Paris, France" (see "Your turn" in `test_golden.py`).
2. Add tests for an ambiguous city, a missing city, and a date in the past.
3. Run the tests with the real model.
