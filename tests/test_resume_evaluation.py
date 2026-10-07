"""标准答案集与字段级评测回归测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from joblens.evaluation import evaluate_resume_corpus  # noqa: E402


class ResumeEvaluationTests(unittest.TestCase):
    """验证十份标准答案样本达到第二阶段质量门槛。"""

    def test_gold_corpus_reaches_stage_two_quality_gate(self) -> None:
        """标准答案集的总体 Precision、Recall 和 F1 均应达到 0.95。"""
        report = evaluate_resume_corpus(
            PROJECT_ROOT / "data" / "ground_truth" / "resume_parser_gold.json",
            PROJECT_ROOT,
        )
        self.assertEqual(report["sample_count"], 10)
        self.assertEqual(report["successful_samples"], 10)
        self.assertGreaterEqual(report["overall"]["precision"], 0.95)
        self.assertGreaterEqual(report["overall"]["recall"], 0.95)
        self.assertGreaterEqual(report["overall"]["f1"], 0.95)


if __name__ == "__main__":
    unittest.main()
