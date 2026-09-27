"""Guard the judge's token-prefix assumption."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exercises" / "05-llm-judge"))
from judge_helpers import shared_prefix_length


class JudgePromptTest(unittest.TestCase):
    def test_only_identical_tokens_are_cached(self):
        self.assertEqual(shared_prefix_length([[1, 2, 3, 7], [1, 2, 3, 8], [1, 2, 3, 9]]), 3)

    def test_identical_prompts_share_everything(self):
        self.assertEqual(shared_prefix_length([[1, 2], [1, 2]]), 2)


if __name__ == "__main__":
    unittest.main()
