"""Stage-0 smoke tests using only Python's standard library."""

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import app  # noqa: E402


class StageZeroSmokeTests(unittest.TestCase):
    def test_sample_resume_is_readable(self) -> None:
        content = app.load_text(app.SAMPLE_RESUME)
        self.assertIn("虚构人物", content)

    def test_exactly_five_job_descriptions_exist(self) -> None:
        job_files = app.discover_job_files()
        self.assertEqual(len(job_files), 5)
        for job_file in job_files:
            self.assertTrue(app.load_text(job_file))

    def test_env_file_is_ignored(self) -> None:
        gitignore = app.load_text(PROJECT_ROOT / ".gitignore")
        self.assertIn(".env", gitignore.splitlines())


if __name__ == "__main__":
    unittest.main()

