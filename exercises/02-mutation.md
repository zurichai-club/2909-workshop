# Exercise 2 — test the tests

Checkpoint: `git checkout ex2-start`.

Run `uv run --env-file .env python scripts/mutate.py`. The custom mutation
removes the prompt rule about units. The initial golden set does not check for
units, so the mutant survives. That is a coverage gap in the evaluation suite.

Add `"°C"` to `answer_contains` on one normal forecast case in
`fixtures/golden.json`. Run `scripts/mutate.py` again. The mutant should now be
killed and the score should be `1/1`. The baseline must stay green.

The mutation script changes the agent prompt/behavior, not the user input.
Typos or paraphrases of the input would test agent robustness instead.
