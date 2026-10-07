"""简历—岗位证据匹配工具的数据模型。"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import Field, model_validator

from .common import SourceReference, StrictBaseModel
from .job import RequirementCategory, RequirementImportance


class EvidenceMatchStatus(str, Enum):
    """单条岗位要求与简历证据的关系。"""

    MATCHED = "matched"
    PARTIALLY_MATCHED = "partially_matched"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NOT_MATCHED = "not_matched"


class MatchDecisionMethod(str, Enum):
    """当前结论来自规则、模型复核还是二者组合。"""

    RULE = "rule"
    HYBRID = "hybrid"
    LLM_REVIEW = "llm_review"


class EvidenceSection(str, Enum):
    """简历证据所属区域。"""

    BASIC_INFORMATION = "basic_information"
    JOB_PREFERENCE = "job_preference"
    EDUCATION = "education"
    SKILL = "skill"
    WORK_EXPERIENCE = "work_experience"
    PROJECT = "project"
    CERTIFICATE = "certificate"
    AWARD = "award"
    LANGUAGE = "language"
    PUBLICATION = "publication"
    OTHER = "other"


class EvidencePolarity(str, Enum):
    """证据是支持要求还是明确说明能力不足。"""

    POSITIVE = "positive"
    NEGATIVE = "negative"


class EvidenceCandidate(StrictBaseModel):
    """可以被匹配结论引用的一条真实简历证据。"""

    evidence_id: str = Field(min_length=1)
    section: EvidenceSection
    text: str = Field(min_length=1)
    polarity: EvidencePolarity = EvidencePolarity.POSITIVE
    matched_keywords: list[str] = Field(default_factory=list)
    match_score: float = Field(default=0.0, ge=0.0, le=1.0)
    proficiency_score: Optional[int] = Field(default=None, ge=0, le=4)
    years_of_experience: Optional[float] = Field(default=None, ge=0)
    source: list[SourceReference] = Field(default_factory=list)


class ConstraintCheck(StrictBaseModel):
    """学历、专业、年限或到岗条件等结构化约束判断。"""

    name: str = Field(min_length=1)
    expected: str = Field(min_length=1)
    actual: Optional[str] = None
    passed: Optional[bool] = None
    is_hard_constraint: bool = False
    explanation: str = Field(min_length=1)


class RequirementEvidenceMatch(StrictBaseModel):
    """一条岗位要求的完整证据匹配结论。"""

    requirement_id: str = Field(min_length=1)
    requirement: str = Field(min_length=1)
    category: RequirementCategory
    importance: RequirementImportance
    is_hard_constraint: bool = False
    status: EvidenceMatchStatus
    decision_method: MatchDecisionMethod = MatchDecisionMethod.RULE
    confidence: float = Field(ge=0.0, le=1.0)
    matched_keywords: list[str] = Field(default_factory=list)
    missing_keywords: list[str] = Field(default_factory=list)
    evidence: list[EvidenceCandidate] = Field(default_factory=list)
    constraint_checks: list[ConstraintCheck] = Field(default_factory=list)
    requirement_source: list[SourceReference] = Field(default_factory=list)
    explanation: str = Field(min_length=1)
    needs_human_review: bool = False


class EvidenceMatchSummary(StrictBaseModel):
    """整份匹配报告的可比较指标。"""

    total_requirements: int = Field(ge=0)
    matched: int = Field(ge=0)
    partially_matched: int = Field(ge=0)
    insufficient_evidence: int = Field(ge=0)
    not_matched: int = Field(ge=0)
    hard_constraints_total: int = Field(ge=0)
    hard_constraints_passed: int = Field(ge=0)
    hard_constraints_failed: int = Field(ge=0)
    hard_constraints_unknown: int = Field(ge=0)
    evidence_coverage: float = Field(ge=0.0, le=1.0)
    weighted_score: float = Field(ge=0.0, le=100.0)
    review_required: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> "EvidenceMatchSummary":
        """校验状态数量及硬约束数量能与总数相互对应。"""
        status_total = (
            self.matched
            + self.partially_matched
            + self.insufficient_evidence
            + self.not_matched
        )
        if status_total != self.total_requirements:
            raise ValueError("匹配状态数量之和必须等于 total_requirements")
        hard_total = (
            self.hard_constraints_passed
            + self.hard_constraints_failed
            + self.hard_constraints_unknown
        )
        if hard_total != self.hard_constraints_total:
            raise ValueError("硬约束状态数量之和必须等于 hard_constraints_total")
        return self


class EvidenceMatchResult(StrictBaseModel):
    """证据匹配工具对单个岗位的标准输出。"""

    success: bool
    schema_version: str = "1.0"
    resume_source: Optional[str] = None
    job_source: Optional[str] = None
    job_company: Optional[str] = None
    job_title: Optional[str] = None
    matches: list[RequirementEvidenceMatch] = Field(default_factory=list)
    summary: Optional[EvidenceMatchSummary] = None
    recommendations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    processing_time_ms: Optional[float] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_result_state(self) -> "EvidenceMatchResult":
        """成功结果必须有 summary，失败结果必须有错误信息。"""
        if self.success and self.summary is None:
            raise ValueError("匹配成功时 summary 不能为空")
        if not self.success and not self.errors:
            raise ValueError("匹配失败时必须提供 errors")
        return self


class RankedJobMatch(StrictBaseModel):
    """批量匹配中的岗位排名条目。"""

    rank: int = Field(ge=1)
    job_source: str = Field(min_length=1)
    job_company: str = Field(min_length=1)
    job_title: str = Field(min_length=1)
    weighted_score: float = Field(ge=0.0, le=100.0)
    hard_constraints_failed: int = Field(ge=0)
    review_required: int = Field(ge=0)
    result_file: Optional[str] = None


class BatchEvidenceMatchResult(StrictBaseModel):
    """一份简历匹配多个岗位时的汇总结果。"""

    success: bool
    resume_source: str = Field(min_length=1)
    job_count: int = Field(ge=0)
    successful_jobs: int = Field(ge=0)
    failed_jobs: int = Field(ge=0)
    rankings: list[RankedJobMatch] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    processing_time_ms: Optional[float] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_job_counts(self) -> "BatchEvidenceMatchResult":
        """校验成功岗位数与失败岗位数之和等于岗位总数。"""
        if self.successful_jobs + self.failed_jobs != self.job_count:
            raise ValueError("成功和失败岗位数之和必须等于 job_count")
        return self
