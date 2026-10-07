"""富结构简历与岗位解析器测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from pydantic import ValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from joblens.parsers import parse_job_document, parse_resume_document  # noqa: E402
from joblens.schemas import (  # noqa: E402
    JobParseMetadata,
    MatchMode,
    RequirementImportance,
    ResumeParseResult,
)
from joblens.schemas.resume import ResumeParseMetadata  # noqa: E402


class DetailedParserTests(unittest.TestCase):
    """验证富结构解析、证据位置和岗位匹配规则字段。"""

    def test_resume_parser_preserves_traceable_evidence(self) -> None:
        """技能字段的字符范围必须仍能指回标准化后的原文。"""
        result = parse_resume_document(PROJECT_ROOT / "data" / "resumes" / "sample_resume.md")
        self.assertTrue(result.success)
        self.assertIsNotNone(result.data)
        assert result.data is not None
        self.assertEqual(len(result.data.projects), 2)
        self.assertIn("Python", {skill.normalized_name for skill in result.data.skills})
        self.assertIn("尚未完成 RAG 或 Agent 项目", result.data.other_information)
        for skill in result.data.skills:
            for source in skill.source:
                self.assertEqual(
                    result.data.raw_text[source.start_char : source.end_char], source.text
                )

    def test_job_parser_extracts_requirement_logic(self) -> None:
        """岗位解析器应识别城市、日期、来源和关键词匹配规则。"""
        result = parse_job_document(
            PROJECT_ROOT / "data" / "jobs" / "04_baidu_llm_application_engineer.md"
        )
        self.assertTrue(result.success)
        assert result.data is not None
        self.assertEqual(result.data.company.name, "百度")
        self.assertEqual(result.data.basic_information.cities, ["上海市"])
        self.assertEqual(result.data.basic_information.published_date, "2026-07-21")
        self.assertIn("talent.baidu.com", result.data.basic_information.source_url or "")
        agent_requirement = next(
            item for item in result.data.requirements if "至少一个方向" in item.description
        )
        self.assertEqual(agent_requirement.match_mode, MatchMode.AT_LEAST)
        self.assertEqual(agent_requirement.minimum_match_count, 1)
        preferred = next(item for item in result.data.requirements if "Docker" in item.description)
        self.assertEqual(preferred.importance, RequirementImportance.PREFERRED)

    def test_all_sample_sources_parse_without_fatal_warnings(self) -> None:
        """三十份岗位样本都应解析成功并产生足够任职要求。"""
        for path in sorted((PROJECT_ROOT / "data" / "jobs").glob("*.md")):
            result = parse_job_document(path)
            self.assertTrue(result.success, path.name)
            self.assertIsNotNone(result.data)
            assert result.data is not None
            self.assertGreaterEqual(len(result.data.requirements), 5)

    def test_resume_parse_result_state_is_validated(self) -> None:
        """成功的简历解析结果缺少 data 时必须触发模型校验错误。"""
        with self.assertRaises(ValidationError):
            ResumeParseResult(
                success=True,
                metadata=ResumeParseMetadata(parser_name="unit_test"),
            )


if __name__ == "__main__":
    unittest.main()
