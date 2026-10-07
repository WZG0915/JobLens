"""只使用 Python 标准库的阶段 0 冒烟测试。"""

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import app  # noqa: E402


class StageZeroSmokeTests(unittest.TestCase):
    """验证示例数据、岗位数量和敏感文件忽略规则。"""

    def test_sample_resume_is_readable(self) -> None:
        """阶段 0 示例简历必须存在且能够读取。"""
        content = app.load_text(app.SAMPLE_RESUME)
        self.assertIn("虚构人物", content)

    def test_exactly_thirty_job_descriptions_exist(self) -> None:
        """实验数据目录必须恰好包含三十份岗位描述。"""
        job_files = app.discover_job_files()
        self.assertEqual(len(job_files), 30)
        for job_file in job_files:
            self.assertTrue(app.load_text(job_file))

    def test_env_file_is_ignored(self) -> None:
        """包含密钥的 .env 文件必须被 Git 忽略。"""
        gitignore = app.load_text(PROJECT_ROOT / ".gitignore")
        self.assertIn(".env", gitignore.splitlines())


if __name__ == "__main__":
    unittest.main()
