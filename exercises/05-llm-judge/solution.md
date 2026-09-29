# Solution — Optional exercise 5 — LLM as a judge

1. Task completion fails `test_fahrenheit` ("did not give the temperature in
   Fahrenheit as requested"). The other three judges pass it.
2. The `GEval` judge is at the bottom of `test_judge.py`. It catches the
   Fahrenheit answer on its own.
3. The cheaper judge lets the Fahrenheit answer pass: the judge model matters.
