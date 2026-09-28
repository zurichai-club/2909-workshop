# Exercise 2 — test the tests

## Run

```bash
cd exercises/02-mutation-testing
uv run python mutate.py
```

## Task

1. Run `mutate.py`: the mutant (the prompt without its units rule) survives.
2. Add a check to `test_golden.py` so the mutant is killed.

<details>
<summary>Solution</summary>

In `test_berlin_rain`:

```python
    assert_test(test_case, [TOOL_CORRECTNESS, contains("Berlin"), contains("take an umbrella"), contains("°C")])
```

</details>
