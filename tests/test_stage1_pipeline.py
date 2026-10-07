"""阶段 1 固定解析与匹配流程的自动测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from joblens.exceptions import DocumentReadError, StructuredOutputError  # noqa: E402
from joblens.io import read_document  # noqa: E402
from joblens.models import AnalysisReport, MatchStatus, ResumeProfile  # noqa: E402
from joblens.pipeline import (  # noqa: E402
    generate_report,
    match_requirements,
    parse_job_description,
    parse_resume,
)
from joblens.validation import validate_payload  # noqa: E402


SAMPLE_RESUME = PROJECT_ROOT / "data" / "resumes" / "sample_resume.md"
JOBS_DIR = PROJECT_ROOT / "data" / "jobs"


class StageOnePipelineTests(unittest.TestCase):
    """验证结构化解析、证据匹配、JSON输出和异常处理。"""
    def test_resume_is_parsed_into_structured_data(self) -> None:
        """阶段 1 简历入口应产出技能、项目和来源文件。"""
        resume = parse_resume(SAMPLE_RESUME)
        skill_names = {skill.name for skill in resume.skills}
        self.assertIn("Python", skill_names)
        self.assertIn("PyTorch", skill_names)
        self.assertEqual(len(resume.projects), 2)
        self.assertTrue(resume.source_file.endswith("sample_resume.md"))

    def test_all_thirty_jobs_are_parsed(self) -> None:
        """三十份岗位都应解析出公司、标题和至少五条要求。"""
        job_files = sorted(JOBS_DIR.glob("*.md"))
        self.assertEqual(len(job_files), 30)
        for job_file in job_files:
            job = parse_job_description(job_file)
            self.assertTrue(job.company)
            self.assertTrue(job.title)
            self.assertGreaterEqual(len(job.requirements), 5)

    def test_matching_keeps_requirement_and_resume_evidence_separate(self) -> None:
        """岗位要求原文与简历支持证据必须分开保存。"""
        resume = parse_resume(SAMPLE_RESUME)
        job = parse_job_description(JOBS_DIR / "01_baidu_llm_algorithm_graduate.md")
        report = match_requirements(resume, job)

        self.assertEqual(report.summary.total_requirements, len(job.requirements))
        self.assertTrue(all(item.requirement_evidence for item in report.matches))
        self.assertTrue(
            all(
                item.resume_evidence
                or item.status
                in {MatchStatus.insufficient_evidence, MatchStatus.not_matched}
                for item in report.matches
            )
        )
        self.assertGreater(report.summary.evidence_coverage, 0.0)
        framework_match = next(
            item for item in report.matches if "PyTorch" in item.requirement
        )
        self.assertEqual(framework_match.status, MatchStatus.matched)
        self.assertTrue(framework_match.resume_evidence)

    def test_generate_report_writes_valid_json(self) -> None:
        """固定流程应写出可以重新加载的合法 JSON 报告。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "report.json"
            report, saved_to = generate_report(
                resume_path=SAMPLE_RESUME,
                job_path=JOBS_DIR / "01_baidu_llm_algorithm_graduate.md",
                output_path=output,
            )
            loaded = AnalysisReport.model_validate_json(
                saved_to.read_text(encoding="utf-8")
            )
            self.assertEqual(loaded.job_title, report.job_title)

    def test_invalid_payload_has_a_readable_error(self) -> None:
        """无效结构化数据应产生带上下文的可读错误。"""
        invalid = {
            "target_roles": [],
            "education": [],
            "skills": "Python",
            "projects": [],
            "limitations": [],
        }
        with self.assertRaisesRegex(StructuredOutputError, "结构化输出无效"):
            validate_payload(ResumeProfile, invalid, "测试简历")

    def test_missing_document_has_a_readable_error(self) -> None:
        """读取不存在文件时应返回明确的文件错误。"""
        with self.assertRaisesRegex(DocumentReadError, "文件不存在"):
            read_document(PROJECT_ROOT / "data" / "missing.md")


if __name__ == "__main__":
    unittest.main()
