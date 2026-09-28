"""Mutation testing: break the agent on purpose and check that the golden set notices."""

import os
import subprocess
import sys

from agent import SYSTEM_PROMPT, UNITS_RULE

# The mutant: the same agent with the units rule removed from its prompt.
MUTANT_PROMPT = SYSTEM_PROMPT.replace(UNITS_RULE, "")


def golden_set_passes(system_prompt):
    """Run test_golden.py with pytest, using this system prompt. True when every test passed."""
    env = dict(os.environ, SYSTEM_PROMPT=system_prompt)
    completed = subprocess.run([sys.executable, "-m", "pytest", "-q", "test_golden.py"], env=env)
    return completed.returncode == 0


print("Baseline (original prompt):", flush=True)
baseline_passes = golden_set_passes(SYSTEM_PROMPT)

print("\nMutant (units rule removed):", flush=True)
mutant_passes = golden_set_passes(MUTANT_PROMPT)

# The mutant is "killed" when the golden set passed on the baseline but fails on the mutant.
if baseline_passes and not mutant_passes:
    print("\nMutation score: 1/1 -- killed. The golden set caught the broken prompt.")
else:
    print("\nMutation score: 0/1 -- survived. No golden case noticed the broken prompt.")

sys.exit(0 if baseline_passes else 1)
