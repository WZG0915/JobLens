"""本地解析与大模型解析共同使用的 Pydantic 数据结构。"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """拒绝未定义字段，避免错误的大模型输出被悄悄接受。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SkillLevel(str, Enum):
    """简历中能够由原文证据支持的技能熟练程度。"""
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"


class RequirementImportance(str, Enum):
    """岗位要求是必需条件还是优先/加分条件。"""
    required = "required"
    preferred = "preferred"


class MatchStatus(str, Enum):
    """一条岗位要求与简历证据之间的匹配状态。"""
    matched = "matched"
    partially_matched = "partially_matched"
    insufficient_evidence = "insufficient_evidence"
    not_matched = "not_matched"


class EducationRecord(StrictModel):
    """阶段 1 使用的一条教育经历及对应原文证据。"""

    description: str = Field(min_length=1)
    evidence: str = Field(min_length=1)


class SkillRecord(StrictModel):
    """阶段 1 使用的一项技能、熟练度和证据列表。"""

    name: str = Field(min_length=1)
    level: SkillLevel
    evidence: list[str] = Field(min_length=1)


class ProjectRecord(StrictModel):
    """阶段 1 使用的一条项目经历。"""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    technologies: list[str]
    evidence: list[str] = Field(min_length=1)


class ResumeProfile(StrictModel):
    """从一份简历中提取出的统一结构。"""
    target_roles: list[str]
    education: list[EducationRecord]
    skills: list[SkillRecord]
    projects: list[ProjectRecord]
    limitations: list[str]
    source_file: str = ""


class JobRequirement(StrictModel):
    """阶段 1 使用的一条岗位要求及其匹配关键词。"""

    description: str = Field(min_length=1)
    importance: RequirementImportance
    keywords: list[str]


class JobDescription(StrictModel):
    """从一份岗位 JD 中提取出的统一结构。"""
    company: str = Field(min_length=1)
    title: str = Field(min_length=1)
    responsibilities: list[str]
    requirements: list[JobRequirement] = Field(min_length=1)
    source_file: str = ""


class RequirementMatch(StrictModel):
    """单条岗位要求的证据匹配结果。"""
    requirement: str = Field(min_length=1)
    importance: RequirementImportance
    status: MatchStatus
    matched_keywords: list[str]
    requirement_evidence: str = Field(min_length=1)
    resume_evidence: list[str]
    explanation: str = Field(min_length=1)


class ReportSummary(StrictModel):
    """整份报告中各匹配状态的数量和证据覆盖率。"""
    total_requirements: int = Field(ge=0)
    matched: int = Field(ge=0)
    partially_matched: int = Field(ge=0)
    insufficient_evidence: int = Field(ge=0)
    not_matched: int = Field(ge=0)
    evidence_coverage: float = Field(ge=0.0, le=1.0)


class AnalysisReport(StrictModel):
    """阶段 1 最终写入 JSON 文件的分析报告。"""
    resume_source: str
    job_source: str
    job_company: str
    job_title: str
    matches: list[RequirementMatch]
    summary: ReportSummary
    recommendations: list[str]
