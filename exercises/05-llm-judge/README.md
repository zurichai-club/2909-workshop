# Optional exercise 5 — LLM as a judge

## Run

```bash
cd exercises/05-llm-judge
export OPENAI_API_KEY=...
uv run pytest -v test_judge.py
```

## Task

1. Run the tests and find which judge catches the Fahrenheit answer.
2. Write your own judge with `GEval`: "the temperature is in the units the user
   asked for" (see "Your turn" in `test_judge.py`).
3. Change `JUDGE` to `GPTModel(model="gpt-5-nano")` and run again.
