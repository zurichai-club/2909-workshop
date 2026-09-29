# Solution — Exercise 1 — a golden set

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
