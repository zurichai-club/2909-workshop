# Exercise 1 — a golden set

Checkpoint: `git checkout ex1-start`.

Run `uv run --env-file .env python scripts/eval_golden.py --deepeval`.
Inspect `fixtures/golden.json`. Its three cases specify expected tool calls,
city and date arguments, and answer properties. The dates are relative to the
current day so the cases remain usable next week. The tool response is a fixed
fixture in `fixtures/forecast.json`.

Add three cases of your own: an ambiguous city, a missing location, and a past
date. For clarification cases, omit `expected_tool`, `expected_city` and
`date_offset`, and set `answer_contains` to words in the desired clarification.
Run the suite again. Compare the deterministic checks with DeepEval's tool
correctness report. DeepEval only receives tool names; the plain Python checks
also cover arguments and answer requirements.

The `mock` model deliberately supports Berlin, Zurich and Oslo. For other
cities, use `--model openai` and a real model, or extend the recorded model for
your own edge case.
