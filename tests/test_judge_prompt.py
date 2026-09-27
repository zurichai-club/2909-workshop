"""Guard the benchmark's token-prefix and answer parsing assumptions."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from judge_cache_benchmark import label_of, longest_shared_prefix


class JudgePromptTest(unittest.TestCase):
    def test_only_identical_tokens_are_cached(self):
        self.assertEqual(longest_shared_prefix([[1, 2, 3, 7], [1, 2, 3, 8], [1, 2, 3, 9]]), [1, 2, 3])

    def test_only_explicit_yes_or_no_counts(self):
        self.assertEqual(label_of("YES."), "YES")
        self.assertEqual(label_of("No, it did not"), "NO")
        self.assertEqual(label_of("Maybe"), "INVALID")


if __name__ == "__main__":
    unittest.main()
