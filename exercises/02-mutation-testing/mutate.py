"""Mutation testing: break the agent on purpose and check that the golden set notices."""

import sys

from eval_golden import run_golden_set
from weather_agent.agent import SYSTEM_PROMPT, UNITS_RULE

# The mutant: the same agent with the units rule removed from its prompt.
MUTANT_PROMPT = SYSTEM_PROMPT.replace(UNITS_RULE, "")

print("Baseline (original prompt):")
baseline_failures = run_golden_set(SYSTEM_PROMPT)

print("\nMutant (units rule removed):")
mutant_failures = run_golden_set(MUTANT_PROMPT)

# The mutant is "killed" when a test that passed on the baseline now fails.
if baseline_failures == 0 and mutant_failures > 0:
    print("\nMutation score: 1/1 -- killed. The golden set caught the broken prompt.")
else:
    print("\nMutation score: 0/1 -- survived. No golden case noticed the broken prompt.")

sys.exit(1 if baseline_failures else 0)
