"""JobLens 岗位 JD 解析数据模型。

本模块只定义数据结构，不负责读取或解析 JD。
所有岗位解析器最终都应该返回 JobParseResult。
"""

from enum import Enum
from typing import Optional

from pydantic import Field, model_validator

from .common import SourceReference, StrictBaseModel


class EmploymentType(str, Enum):
    """用工类型。"""

    INTERNSHIP = "internship"
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    CAMPUS_RECRUITMENT = "campus_recruitment"
    OTHER = "other"
    UNKNOWN = "unknown"


class WorkMode(str, Enum):
    """办公形式。"""

    ON_SITE = "on_site"
    REMOTE = "remote"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class ExperienceLevel(str, Enum):
    """岗位面向的经验等级。"""

    INTERN = "intern"
    ENTRY = "entry"
    JUNIOR = "junior"
    MIDDLE = "middle"
    SENIOR = "senior"
    EXPERT = "expert"
    UNKNOWN = "unknown"


class EducationDegree(str, Enum):
    """岗位要求的最低学历。"""

    HIGH_SCHOOL = "high_school"
    ASSOCIATE = "associate"
    BACHELOR = "bachelor"
    MASTER = "master"
    DOCTOR = "doctor"
    OTHER = "other"
    UNKNOWN = "unknown"


class RequirementCategory(str, Enum):
    """岗位要求类别，用于选择不同的证据匹配策略。"""

    PROGRAMMING_LANGUAGE = "programming_language"
    FRAMEWORK = "framework"
    DATABASE = "database"
    AI_ML = "ai_ml"
    TOOL = "tool"
    PLATFORM = "platform"
    EDUCATION = "education"
    MAJOR = "major"
    EXPERIENCE = "experience"
    PROJECT = "project"
    LANGUAGE = "language"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    SOFT_SKILL = "soft_skill"
    CERTIFICATE = "certificate"
    AVAILABILITY = "availability"
    OTHER = "other"


class RequirementImportance(str, Enum):
    """岗位要求是必需条件、优先条件，还是原文无法确认。"""

    REQUIRED = "required"
    PREFERRED = "preferred"
    UNKNOWN = "unknown"


class MatchMode(str, Enum):
    """一条要求内多个关键词之间的逻辑关系。"""

    ALL = "all"
    ANY = "any"
    AT_LEAST = "at_least"


class ProficiencyLevel(str, Enum):
    """JD 对技能熟练程度的要求。"""

    KNOW = "know"
    FAMILIAR = "familiar"
    PROFICIENT = "proficient"
    EXPERT = "expert"
    UNKNOWN = "unknown"


class SalaryPeriod(str, Enum):
    """薪资计算周期。"""

    DAILY = "daily"
    MONTHLY = "monthly"
    YEARLY = "yearly"
    UNKNOWN = "unknown"


class CompanyInformation(StrictBaseModel):
    """JD 中明确提供的公司信息。"""

    name: Optional[str] = None
    industry: Optional[str] = None
    company_size: Optional[str] = None
    financing_stage: Optional[str] = None
    description: Optional[str] = None
    source: list[SourceReference] = Field(default_factory=list)


class SalaryRange(StrictBaseModel):
    """结构化薪资；无法确定数值时只保留 raw_text。"""

    minimum: Optional[float] = Field(default=None, ge=0)
    maximum: Optional[float] = Field(default=None, ge=0)
    currency: str = "CNY"
    period: SalaryPeriod = SalaryPeriod.UNKNOWN
    months_per_year: Optional[int] = Field(default=None, ge=1, le=24)
    raw_text: str = ""

    @model_validator(mode="after")
    def validate_salary_range(self) -> "SalaryRange":
        """同时存在上下限时，最高薪资不能低于最低薪资。"""
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.maximum < self.minimum
        ):
            raise ValueError("maximum 不能小于 minimum")
        return self


class JobBasicInformation(StrictBaseModel):
    """岗位名称、类型、地点和发布时间等基本信息。"""

    title: str = ""
    department: Optional[str] = None
    employment_type: EmploymentType = EmploymentType.UNKNOWN
    experience_level: ExperienceLevel = ExperienceLevel.UNKNOWN
    work_mode: WorkMode = WorkMode.UNKNOWN
    cities: list[str] = Field(default_factory=list)
    address: Optional[str] = None
    salary: Optional[SalaryRange] = None
    source_url: Optional[str] = None
    published_date: Optional[str] = None
    source: list[SourceReference] = Field(default_factory=list)


class JobResponsibility(StrictBaseModel):
    """一条岗位职责；职责不自动等同于候选人硬性要求。"""

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    keywords: list[str] = Field(default_factory=list)
    normalized_keywords: list[str] = Field(default_factory=list)
    source: list[SourceReference] = Field(default_factory=list)


class JobRequirement(StrictBaseModel):
    """一条可以独立与简历证据进行匹配的岗位要求。"""

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    category: RequirementCategory = RequirementCategory.OTHER
    importance: RequirementImportance = RequirementImportance.UNKNOWN
    is_hard_constraint: bool = False
    keywords: list[str] = Field(default_factory=list)
    normalized_keywords: list[str] = Field(default_factory=list)
    match_mode: MatchMode = MatchMode.ALL
    minimum_match_count: Optional[int] = Field(default=None, ge=1)
    required_proficiency: ProficiencyLevel = ProficiencyLevel.UNKNOWN
    minimum_years: Optional[float] = Field(default=None, ge=0)
    minimum_degree: Optional[EducationDegree] = None
    accepted_majors: list[str] = Field(default_factory=list)
    evidence_expectation: Optional[str] = None
    source: list[SourceReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_match_rule(self) -> "JobRequirement":
        """检查“至少满足几项”的配置是否完整且合理。"""
        keyword_count = len(self.normalized_keywords or self.keywords)
        if self.match_mode == MatchMode.AT_LEAST:
            if self.minimum_match_count is None:
                raise ValueError("AT_LEAST 模式必须设置 minimum_match_count")
            if self.minimum_match_count > keyword_count:
                raise ValueError("minimum_match_count 不能超过关键词数量")
        elif self.minimum_match_count is not None:
            raise ValueError("只有 AT_LEAST 模式可以设置 minimum_match_count")
        return self


class AvailabilityRequirement(StrictBaseModel):
    """实习天数、持续时间和到岗日期等要求。"""

    minimum_days_per_week: Optional[int] = Field(default=None, ge=1, le=7)
    minimum_months: Optional[int] = Field(default=None, ge=1)
    start_date: Optional[str] = None
    graduation_years: list[int] = Field(default_factory=list)
    raw_text: str = ""
    source: list[SourceReference] = Field(default_factory=list)


class JobBenefit(StrictBaseModel):
    """一项岗位福利。"""

    name: str = Field(min_length=1)
    description: Optional[str] = None
    source: list[SourceReference] = Field(default_factory=list)


class JobData(StrictBaseModel):
    """解析完成后的完整结构化岗位 JD。"""

    schema_version: str = "1.0"
    basic_information: JobBasicInformation = Field(
        default_factory=JobBasicInformation
    )
    company: CompanyInformation = Field(default_factory=CompanyInformation)
    summary: Optional[str] = None
    responsibilities: list[JobResponsibility] = Field(default_factory=list)
    requirements: list[JobRequirement] = Field(default_factory=list)
    availability: Optional[AvailabilityRequirement] = None
    benefits: list[JobBenefit] = Field(default_factory=list)
    other_information: list[str] = Field(default_factory=list)
    raw_text: str = Field(min_length=1)


class JobParseWarning(StrictBaseModel):
    """不阻止解析继续进行的 JD 数据质量问题。"""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    section: Optional[str] = None


class JobParseMetadata(StrictBaseModel):
    """解析器、来源和耗时等运行信息。"""

    parser_name: str = Field(min_length=1)
    parser_version: str = "1.0"
    source_name: Optional[str] = None
    source_url: Optional[str] = None
    character_count: int = Field(default=0, ge=0)
    line_count: int = Field(default=0, ge=0)
    processing_time_ms: Optional[float] = Field(default=None, ge=0)


class JobParseResult(StrictBaseModel):
    """岗位 JD 解析工具的标准返回类型。"""

    success: bool
    data: Optional[JobData] = None
    warnings: list[JobParseWarning] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    metadata: JobParseMetadata

    @model_validator(mode="after")
    def validate_result_state(self) -> "JobParseResult":
        """成功结果必须携带数据，失败结果必须说明原因。"""
        if self.success and self.data is None:
            raise ValueError("解析成功时 data 不能为空")
        if not self.success and not self.errors:
            raise ValueError("解析失败时 errors 不能为空")
        return self
