"""富结构证据匹配工具测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from joblens.schemas import (  # noqa: E402
    AvailabilityRequirement,
    CompanyInformation,
    ConstraintCheck,
    EvidenceMatchStatus,
    EvidenceMatchSummary,
    JobBasicInformation,
    JobData,
    JobEducationDegree,
    JobRequirement,
    MatchDecisionMethod,
    MatchMode,
    ProficiencyLevel,
    RequirementCategory,
    RequirementImportance,
    SourceReference,
)
from joblens.schemas.resume import (  # noqa: E402
    EducationDegree,
    EducationExperience,
    JobPreference,
    ProjectExperience,
    ResumeData,
    SkillCategory,
    SkillItem,
    SkillLevel,
)
from joblens.tools import (  # noqa: E402
    match_resume_job_data,
    match_resume_job_files,
    match_resume_to_job_directory,
)
from joblens.tools.evidence_matching_tool import SemanticEvidenceReview  # noqa: E402


def source(text: str, section: str = "测试") -> SourceReference:
    """为测试文本快速构造一条合法来源引用。"""
    return SourceReference(section=section, text=text, start_char=0, end_char=len(text), line_number=1)


def resume_with_python(level: SkillLevel = SkillLevel.FAMILIAR) -> ResumeData:
    """构造包含本科教育与 Python 技能的最小测试简历。"""
    evidence = "熟悉 Python，并在项目中开发数据处理脚本"
    return ResumeData(
        education=[
            EducationExperience(
                school="示例大学",
                major="计算机科学与技术",
                degree=EducationDegree.BACHELOR,
                source=[source("示例大学 计算机科学与技术 本科", "教育背景")],
            )
        ],
        skills=[
            SkillItem(
                name="Python",
                normalized_name="Python",
                category=SkillCategory.PROGRAMMING_LANGUAGE,
                level=level,
                source=[source(evidence, "专业技能")],
            )
        ],
        raw_text=evidence,
    )


def job_with_requirement(requirement: JobRequirement) -> JobData:
    """把单条要求包装成可供匹配器使用的最小岗位对象。"""
    return JobData(
        basic_information=JobBasicInformation(title="测试岗位"),
        company=CompanyInformation(name="测试公司"),
        requirements=[requirement],
        raw_text=requirement.description,
    )


class _AlwaysMatchReviewer:
    """测试替身：始终选择第一条候选证据并判定匹配。"""

    def review(self, requirement, candidates):
        """返回固定语义复核结果，避免单元测试访问真实模型。"""
        return SemanticEvidenceReview(
            status=EvidenceMatchStatus.MATCHED,
            evidence_ids=[candidates[0].evidence_id],
            explanation="候选证据在语义上能够支持要求。",
            confidence=0.88,
        )


class EvidenceMatchingToolTests(unittest.TestCase):
    """验证技能、约束、负向证据、批量排名和混合复核逻辑。"""

    def test_exact_skill_and_proficiency_are_matched(self) -> None:
        """技能名称和熟练度均满足时应判定完全匹配。"""
        requirement = JobRequirement(
            id="req_python",
            description="熟悉 Python",
            category=RequirementCategory.PROGRAMMING_LANGUAGE,
            importance=RequirementImportance.REQUIRED,
            normalized_keywords=["Python"],
            required_proficiency=ProficiencyLevel.FAMILIAR,
            source=[source("熟悉 Python", "任职要求")],
        )
        result = match_resume_job_data(resume_with_python(), job_with_requirement(requirement))
        self.assertTrue(result.success)
        self.assertEqual(result.matches[0].status, EvidenceMatchStatus.MATCHED)
        self.assertEqual(result.matches[0].matched_keywords, ["Python"])
        self.assertEqual(result.matches[0].evidence[0].source[0].text, "熟悉 Python，并在项目中开发数据处理脚本")

    def test_proficiency_gap_is_partial_match(self) -> None:
        """存在技能但熟练度不足时应判定部分匹配。"""
        requirement = JobRequirement(
            id="req_python",
            description="精通 Python",
            category=RequirementCategory.PROGRAMMING_LANGUAGE,
            importance=RequirementImportance.REQUIRED,
            normalized_keywords=["Python"],
            required_proficiency=ProficiencyLevel.EXPERT,
        )
        result = match_resume_job_data(resume_with_python(), job_with_requirement(requirement))
        self.assertEqual(result.matches[0].status, EvidenceMatchStatus.PARTIALLY_MATCHED)
        self.assertTrue(result.matches[0].needs_human_review)

    def test_any_and_all_keyword_logic(self) -> None:
        """验证 ANY 与 ALL 两种关键词组合逻辑。"""
        any_requirement = JobRequirement(
            id="req_any",
            description="熟悉 Python 或 Go",
            category=RequirementCategory.PROGRAMMING_LANGUAGE,
            normalized_keywords=["Python", "Go"],
            match_mode=MatchMode.ANY,
            required_proficiency=ProficiencyLevel.FAMILIAR,
        )
        all_requirement = any_requirement.model_copy(
            update={"id": "req_all", "description": "熟悉 Python 和 Go", "match_mode": MatchMode.ALL}
        )
        job = JobData(
            basic_information=JobBasicInformation(title="后端岗位"),
            company=CompanyInformation(name="测试公司"),
            requirements=[any_requirement, all_requirement],
            raw_text="测试岗位",
        )
        result = match_resume_job_data(resume_with_python(), job)
        self.assertEqual(result.matches[0].status, EvidenceMatchStatus.MATCHED)
        self.assertEqual(result.matches[1].status, EvidenceMatchStatus.PARTIALLY_MATCHED)
        self.assertEqual(result.matches[1].missing_keywords, ["Go"])

    def test_grouped_alternatives_require_fixed_and_one_optional_skill(self) -> None:
        """“固定技能以及多个备选之一”应要求至少命中两项。"""
        resume = ResumeData(
            skills=[
                SkillItem(
                    name="PyTorch",
                    normalized_name="PyTorch",
                    category=SkillCategory.AI_ML,
                    level=SkillLevel.FAMILIAR,
                    source=[source("熟悉 PyTorch", "专业技能")],
                )
            ],
            raw_text="熟悉 PyTorch",
        )
        requirement = JobRequirement(
            id="req_grouped",
            description="熟悉 PyTorch，以及 vLLM 或 SGLang",
            category=RequirementCategory.AI_ML,
            normalized_keywords=["PyTorch", "vLLM", "SGLang"],
            match_mode=MatchMode.ANY,
            required_proficiency=ProficiencyLevel.FAMILIAR,
        )
        result = match_resume_job_data(resume, job_with_requirement(requirement))
        self.assertEqual(result.matches[0].status, EvidenceMatchStatus.PARTIALLY_MATCHED)

    def test_requirement_without_keywords_or_constraints_is_not_auto_matched(self) -> None:
        """没有关键词或结构化约束的要求不能自动判定匹配。"""
        requirement = JobRequirement(
            id="req_generic",
            description="责任心强，积极主动",
            category=RequirementCategory.OTHER,
            importance=RequirementImportance.REQUIRED,
        )
        result = match_resume_job_data(resume_with_python(), job_with_requirement(requirement))
        self.assertEqual(result.matches[0].status, EvidenceMatchStatus.INSUFFICIENT_EVIDENCE)
        self.assertTrue(result.matches[0].needs_human_review)

    def test_project_technology_is_not_propagated_to_unrelated_source_lines(self) -> None:
        """项目技术不能传播到同一项目中无关的原文行。"""
        resume = ResumeData(
            projects=[
                ProjectExperience(
                    name="后端项目",
                    technologies=["Python"],
                    source=[
                        source("后端项目", "项目经历"),
                        source("使用 SQLite 保存数据", "项目经历"),
                    ],
                )
            ],
            raw_text="后端项目\n使用 SQLite 保存数据",
        )
        requirement = JobRequirement(
            id="req_python",
            description="熟悉 Python",
            category=RequirementCategory.PROGRAMMING_LANGUAGE,
            normalized_keywords=["Python"],
            required_proficiency=ProficiencyLevel.FAMILIAR,
        )
        result = match_resume_job_data(resume, job_with_requirement(requirement))
        self.assertEqual(result.matches[0].status, EvidenceMatchStatus.INSUFFICIENT_EVIDENCE)

    def test_failed_degree_hard_constraint_caps_score(self) -> None:
        """学历硬约束失败时总分必须受到上限限制。"""
        requirement = JobRequirement(
            id="req_degree",
            description="硕士及以上学历",
            category=RequirementCategory.EDUCATION,
            importance=RequirementImportance.REQUIRED,
            is_hard_constraint=True,
            minimum_degree=JobEducationDegree.MASTER,
        )
        result = match_resume_job_data(resume_with_python(), job_with_requirement(requirement))
        self.assertEqual(result.matches[0].status, EvidenceMatchStatus.NOT_MATCHED)
        self.assertEqual(result.summary.hard_constraints_failed, 1)
        self.assertLessEqual(result.summary.weighted_score, 59)
        self.assertFalse(result.matches[0].constraint_checks[0].passed)
        self.assertTrue(result.matches[0].evidence)

    def test_skill_mention_does_not_prove_work_experience(self) -> None:
        """技能列表中的简单提及不能证明拥有工作年限。"""
        requirement = JobRequirement(
            id="req_experience",
            description="具有 Python 开发经验",
            category=RequirementCategory.EXPERIENCE,
            importance=RequirementImportance.REQUIRED,
            normalized_keywords=["Python"],
        )
        result = match_resume_job_data(resume_with_python(), job_with_requirement(requirement))
        self.assertEqual(result.matches[0].status, EvidenceMatchStatus.INSUFFICIENT_EVIDENCE)

    def test_unknown_availability_is_not_guessed(self) -> None:
        """简历没有到岗信息时不得猜测满足可用性要求。"""
        requirement = JobRequirement(
            id="req_availability",
            description="每周到岗 4 天，连续实习 3 个月",
            category=RequirementCategory.AVAILABILITY,
            importance=RequirementImportance.REQUIRED,
            is_hard_constraint=True,
        )
        job = job_with_requirement(requirement).model_copy(
            update={"availability": AvailabilityRequirement(minimum_days_per_week=4, minimum_months=3)}
        )
        resume = resume_with_python().model_copy(update={"job_preference": JobPreference(target_roles=["Python 实习生"])})
        result = match_resume_job_data(resume, job)
        self.assertEqual(result.matches[0].status, EvidenceMatchStatus.INSUFFICIENT_EVIDENCE)
        self.assertEqual(result.summary.hard_constraints_unknown, 1)

    def test_explicit_negative_evidence_is_not_matched(self) -> None:
        """明确的负向证据不能被当作正向能力证据。"""
        resume = ResumeData(other_information=["尚未完成 RAG 项目"], raw_text="尚未完成 RAG 项目")
        requirement = JobRequirement(
            id="req_rag",
            description="具备 RAG 项目经验",
            category=RequirementCategory.PROJECT,
            importance=RequirementImportance.REQUIRED,
            normalized_keywords=["RAG"],
        )
        result = match_resume_job_data(resume, job_with_requirement(requirement))
        self.assertEqual(result.matches[0].status, EvidenceMatchStatus.NOT_MATCHED)
        self.assertEqual(result.matches[0].evidence[0].polarity.value, "negative")

    def test_hybrid_reviewer_can_only_refine_non_hard_result(self) -> None:
        """混合复核器只能细化非硬约束结果。"""
        requirement = JobRequirement(
            id="req_python",
            description="精通 Python",
            category=RequirementCategory.PROGRAMMING_LANGUAGE,
            importance=RequirementImportance.REQUIRED,
            normalized_keywords=["Python"],
            required_proficiency=ProficiencyLevel.EXPERT,
        )
        result = match_resume_job_data(
            resume_with_python(),
            job_with_requirement(requirement),
            mode="hybrid",
            reviewer=_AlwaysMatchReviewer(),
        )
        self.assertEqual(result.matches[0].status, EvidenceMatchStatus.MATCHED)
        self.assertEqual(result.matches[0].decision_method, MatchDecisionMethod.HYBRID)

    def test_summary_rejects_inconsistent_counts(self) -> None:
        """汇总状态数量不一致时 Pydantic 模型必须拒绝。"""
        with self.assertRaises(ValidationError):
            EvidenceMatchSummary(
                total_requirements=1,
                matched=1,
                partially_matched=1,
                insufficient_evidence=0,
                not_matched=0,
                hard_constraints_total=0,
                hard_constraints_passed=0,
                hard_constraints_failed=0,
                hard_constraints_unknown=0,
                evidence_coverage=1,
                weighted_score=100,
                review_required=0,
            )

    def test_file_and_batch_tools_work_with_project_data(self) -> None:
        """单文件和批量匹配入口都能处理项目内真实数据。"""
        resume_path = PROJECT_ROOT / "data" / "resumes" / "sample_resume.md"
        job_path = PROJECT_ROOT / "data" / "jobs" / "04_baidu_llm_application_engineer.md"
        with tempfile.TemporaryDirectory() as temp_dir:
            single_path = Path(temp_dir) / "single.json"
            single = match_resume_job_files(resume_path, job_path, output_path=single_path)
            self.assertTrue(single.success)
            self.assertTrue(single_path.exists())
            self.assertGreater(len(single.matches), 0)

            batch = match_resume_to_job_directory(
                resume_path,
                PROJECT_ROOT / "data" / "jobs",
                output_dir=Path(temp_dir) / "batch",
            )
            self.assertTrue(batch.success)
            self.assertEqual(batch.job_count, 30)
            self.assertEqual(batch.successful_jobs, 30)
            self.assertEqual(len(batch.rankings), 30)
            self.assertEqual([item.rank for item in batch.rankings], list(range(1, 31)))


if __name__ == "__main__":
    unittest.main()
