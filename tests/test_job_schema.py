"""岗位 JD 数据结构的基础校验测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from pydantic import ValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from joblens.schemas import (  # noqa: E402
    DateRange,
    JobData,
    JobParseMetadata,
    JobParseResult,
    JobRequirement,
    MatchMode,
    RequirementImportance,
    SourceReference,
)
from joblens.schemas.job import SourceReference as JobSourceReference  # noqa: E402
from joblens.schemas.resume import (  # noqa: E402
    DateRange as ResumeDateRange,
    SourceReference as ResumeSourceReference,
)


class JobSchemaTests(unittest.TestCase):
    """验证关键词逻辑和解析结果状态约束。"""

    def test_any_requirement_accepts_multiple_alternatives(self) -> None:
        """ANY 模式允许多个备选关键词且不要求最低命中数量。"""
        requirement = JobRequirement(
            id="req_001",
            description="了解 LangChain 或 LangGraph",
            importance=RequirementImportance.REQUIRED,
            normalized_keywords=["LangChain", "LangGraph"],
            match_mode=MatchMode.ANY,
        )
        self.assertEqual(requirement.match_mode, MatchMode.ANY)

    def test_resume_and_job_share_common_models(self) -> None:
        """简历与 JD 必须引用 common.py 中的同一组公共类型。"""
        self.assertIs(SourceReference, ResumeSourceReference)
        self.assertIs(SourceReference, JobSourceReference)
        self.assertIs(DateRange, ResumeDateRange)

    def test_at_least_requires_a_count(self) -> None:
        """AT_LEAST 模式缺少最低命中数量时应校验失败。"""
        with self.assertRaises(ValidationError):
            JobRequirement(
                id="req_002",
                description="至少掌握两种语言",
                normalized_keywords=["Python", "Java", "Go"],
                match_mode=MatchMode.AT_LEAST,
            )

    def test_successful_parse_result_requires_data(self) -> None:
        """岗位解析标记成功时必须同时提供结构化 data。"""
        metadata = JobParseMetadata(parser_name="unit_test")
        with self.assertRaises(ValidationError):
            JobParseResult(success=True, metadata=metadata)

        result = JobParseResult(
            success=True,
            data=JobData(raw_text="示例岗位描述"),
            metadata=metadata,
        )
        self.assertTrue(result.success)


if __name__ == "__main__":
    unittest.main()
