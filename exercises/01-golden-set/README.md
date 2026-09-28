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

<details>
<summary>Solution</summary>

Step 1 is at the bottom of `test_golden.py`. Step 2:

```python
def test_ambiguous_paris():
    """"Paris" is ambiguous, so the agent should ask instead of calling the tool."""
    test_case = agent_test_case("Will it rain in Paris tomorrow?", expected_tools=[])
    assert_test(test_case, [TOOL_CORRECTNESS, contains("Which Paris")])


def test_missing_city():
    """No city at all."""
    test_case = agent_test_case("Will it rain tomorrow?", expected_tools=[])
    assert_test(test_case, [TOOL_CORRECTNESS, contains("which city")])


def test_past_date():
    """A past date; the tool should reject it."""
    test_case = agent_test_case(
        "Did it rain in Berlin yesterday?",
        expected_tools=[ToolCall(name="get_forecast",
                                 input_parameters={"city": "Berlin, Germany", "date": days_from_today(-1)})],
    )
    assert_test(test_case, [TOOL_CORRECTNESS, contains("outside the available 16-day forecast horizon")])
```

</details>
