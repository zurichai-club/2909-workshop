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

<details>
<summary>Solution</summary>

1. Task completion fails `test_fahrenheit` ("did not give the temperature in
   Fahrenheit as requested"). The other three judges pass it.
2. The `GEval` judge is at the bottom of `test_judge.py`. It catches the
   Fahrenheit answer on its own.
3. The cheaper judge lets the Fahrenheit answer pass: the judge model matters.

</details>
