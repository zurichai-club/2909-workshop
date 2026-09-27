"""Each exercise folder has its own copy of the agent; check every starting point still runs."""

import os
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT / ".venv" / "bin" / "python")


def run(folder, *command):
    env = dict(os.environ, TRACE_FILE="")
    env.pop("LANGFUSE_PUBLIC_KEY", None)
    env.pop("LANGFUSE_SECRET_KEY", None)
    return subprocess.run([PYTHON, *command], cwd=ROOT / "exercises" / folder,
                          env=env, capture_output=True, text=True)


class ExerciseStartTest(unittest.TestCase):
    def test_golden_sets_pass(self):
        for folder in ["01-golden-set", "02-mutation-testing", "04-close-the-loop"]:
            completed = run(folder, "eval_golden.py")
            self.assertEqual(completed.returncode, 0, folder + "\n" + completed.stdout + completed.stderr)

    def test_exercise_2_mutant_survives(self):
        self.assertIn("survived", run("02-mutation-testing", "mutate.py").stdout)

    def test_exercise_4_mutant_is_killed(self):
        self.assertIn("killed", run("04-close-the-loop", "mutate.py").stdout)

    def test_production_batch_flags_p2(self):
        # Writes the ignored exercises/03-observability/batch.jsonl, as the exercise itself does.
        output = run("03-observability", "production_batch.py").stdout
        self.assertIn("p2: trace=", output)
        self.assertIn("flag=unit_mismatch", output)


if __name__ == "__main__":
    unittest.main()
