# Solution — Exercise 4 — turn a trace into a test

The test is at the bottom of `test_golden.py`. In `mock_answer`, choose the
temperature text before building the answer:

```python
    if "fahrenheit" in question.lower():
        temperature = f"{forecast['temperature_max_c'] * 9 / 5 + 32:.1f} °F"
    else:
        temperature = f"{forecast['temperature_max_c']} °C"
```

Then use `{temperature}` in place of `{forecast['temperature_max_c']} °C`.
