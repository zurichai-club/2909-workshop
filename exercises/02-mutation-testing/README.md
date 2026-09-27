# Exercise 2 — test the tests

Mutation testing breaks the agent on purpose. If the golden set still passes,
the tests have a blind spot.

```bash
cd exercises/02-mutation-testing
uv run python mutate.py
```

## Files

| File | What it is |
|---|---|
| `mutate.py` | Removes the units rule from the system prompt and runs the golden set again |
| `eval_golden.py` | The golden set runner from exercise 1 (now takes a `system_prompt`) |
| `golden.json` | The three test cases |
| `weather_agent/agent.py` | The agent. At the end of `mock_answer`, the mock model obeys the units rule |

## Task

1. Run `mutate.py`. The mutant **survives**: no case checks for units, so the
   broken prompt goes unnoticed.
2. Add `"°C"` to `answer_contains` of one normal forecast case in `golden.json`.
3. Run `mutate.py` again. The mutant should be **killed** (score `1/1`) and the
   baseline must still pass.

The mutation changes the agent (its prompt), not the user's question. Typos or
paraphrases of the question would test robustness instead.

<details>
<summary>Solution</summary>

```json
{"id": "berlin-rain", ..., "answer_contains": ["Berlin", "take an umbrella", "°C"]}
```

</details>
