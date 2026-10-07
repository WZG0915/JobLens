"""JobLens 简历解析数据模型。

本模块只定义数据结构，不负责读取文件或解析简历。
所有简历解析器最终都应该返回 ResumeParseResult。
"""

from enum import Enum
from typing import Optional

from pydantic import Field, model_validator

from .common import DateRange, SourceReference, StrictBaseModel


# ============================================================
# 枚举类型
# ============================================================


class EducationDegree(str, Enum):
    """学历类型。"""

    HIGH_SCHOOL = "high_school"
    ASSOCIATE = "associate"
    BACHELOR = "bachelor"
    MASTER = "master"
    DOCTOR = "doctor"
    OTHER = "other"
    UNKNOWN = "unknown"


class SkillCategory(str, Enum):
    """技能类别。"""

    PROGRAMMING_LANGUAGE = "programming_language"
    FRAMEWORK = "framework"
    DATABASE = "database"
    AI_ML = "ai_ml"
    TOOL = "tool"
    PLATFORM = "platform"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    SOFT_SKILL = "soft_skill"
    OTHER = "other"


class SkillLevel(str, Enum):
    """技能熟练程度。

    只有当简历中明确出现相关描述时才填写。
    例如“熟悉Python”可以解析为 familiar。
    不允许解析器仅凭项目经历擅自判断为 expert。
    """

    BEGINNER = "beginner"
    FAMILIAR = "familiar"
    PROFICIENT = "proficient"
    EXPERT = "expert"
    UNKNOWN = "unknown"


class LanguageLevel(str, Enum):
    """语言能力等级。"""

    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    NATIVE = "native"
    UNKNOWN = "unknown"


class ExperienceType(str, Enum):
    """工作或实践经历类型。"""

    INTERNSHIP = "internship"
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    FREELANCE = "freelance"
    CAMPUS = "campus"
    VOLUNTEER = "volunteer"
    OTHER = "other"
    UNKNOWN = "unknown"


# ============================================================
# 基本信息与求职意向
# ============================================================


class ContactInformation(StrictBaseModel):
    """联系方式。

    这些字段属于个人敏感信息，写入日志时应进行脱敏。
    """

    phone: Optional[str] = None
    email: Optional[str] = None
    city: Optional[str] = None

    github: Optional[str] = None
    personal_website: Optional[str] = None
    linkedin: Optional[str] = None
    other_links: list[str] = Field(default_factory=list)


class BasicInformation(StrictBaseModel):
    """候选人基本信息。"""

    name: Optional[str] = None
    contact: ContactInformation = Field(
        default_factory=ContactInformation
    )

    current_status: Optional[str] = Field(
        default=None,
        description="例如：在校生、应届毕业生、已工作",
    )

    expected_graduation_date: Optional[str] = Field(
        default=None,
        description="预计毕业时间，推荐格式：YYYY-MM",
    )

    source: list[SourceReference] = Field(
        default_factory=list,
        description="基本信息对应的原文位置",
    )


class JobPreference(StrictBaseModel):
    """求职意向。"""

    target_roles: list[str] = Field(
        default_factory=list,
        description="目标岗位，例如：AI应用开发实习生",
    )

    target_cities: list[str] = Field(default_factory=list)

    employment_types: list[str] = Field(
        default_factory=list,
        description="例如：实习、全职、远程",
    )

    available_start_date: Optional[str] = None

    days_per_week: Optional[int] = Field(
        default=None,
        ge=1,
        le=7,
        description="每周可实习天数",
    )

    internship_duration_months: Optional[int] = Field(
        default=None,
        ge=1,
        description="可以连续实习的月数",
    )

    source: list[SourceReference] = Field(default_factory=list)


# ============================================================
# 教育经历
# ============================================================


class EducationExperience(StrictBaseModel):
    """一段教育经历。"""

    school: str = Field(
        default="",
        description="学校名称",
    )

    college: Optional[str] = Field(
        default=None,
        description="学院名称",
    )

    major: Optional[str] = Field(
        default=None,
        description="专业名称",
    )

    degree: EducationDegree = EducationDegree.UNKNOWN

    date_range: DateRange = Field(default_factory=DateRange)

    gpa: Optional[str] = Field(
        default=None,
        description="保留简历中的GPA表达，例如：3.7/4.0",
    )

    ranking: Optional[str] = Field(
        default=None,
        description="例如：专业前10%",
    )

    relevant_courses: list[str] = Field(
        default_factory=list,
        description="与岗位相关的课程",
    )

    descriptions: list[str] = Field(
        default_factory=list,
        description="教育经历中的其他说明",
    )

    source: list[SourceReference] = Field(
        default_factory=list,
        description="该教育经历对应的原文",
    )


# ============================================================
# 技能
# ============================================================


class SkillItem(StrictBaseModel):
    """一项技能。

    name 保存简历中的名称；
    normalized_name 保存标准化名称，供匹配工具使用。

    例如：
    name="python"
    normalized_name="Python"
    """

    name: str = Field(
        min_length=1,
        description="简历中的技能名称",
    )

    normalized_name: str = Field(
        min_length=1,
        description="标准化后的技能名称",
    )

    category: SkillCategory = SkillCategory.OTHER

    level: SkillLevel = SkillLevel.UNKNOWN

    years_of_experience: Optional[float] = Field(
        default=None,
        ge=0,
        description="只有简历明确说明或能够可靠计算时才填写",
    )

    aliases: list[str] = Field(
        default_factory=list,
        description="该技能在简历中出现过的其他名称",
    )

    source: list[SourceReference] = Field(
        default_factory=list,
        description="该技能在简历中的证据",
    )


class SkillGroup(StrictBaseModel):
    """简历中的一组技能。

    例如：
    编程语言：Python、Java
    数据库：MySQL、SQLite
    """

    group_name: str
    skills: list[SkillItem] = Field(default_factory=list)
    source: list[SourceReference] = Field(default_factory=list)


# ============================================================
# 工作与实习经历
# ============================================================


class WorkExperience(StrictBaseModel):
    """一段工作、实习或者校园实践经历。"""

    organization: str = Field(
        default="",
        description="公司、实验室或组织名称",
    )

    department: Optional[str] = None

    position: str = Field(
        default="",
        description="职位名称",
    )

    experience_type: ExperienceType = ExperienceType.UNKNOWN

    location: Optional[str] = None

    date_range: DateRange = Field(default_factory=DateRange)

    responsibilities: list[str] = Field(
        default_factory=list,
        description="主要职责",
    )

    achievements: list[str] = Field(
        default_factory=list,
        description="成果或者可量化贡献",
    )

    technologies: list[str] = Field(
        default_factory=list,
        description="这段经历中明确使用的技术",
    )

    source: list[SourceReference] = Field(
        default_factory=list,
        description="该经历对应的简历原文",
    )


# ============================================================
# 项目经历
# ============================================================


class ProjectExperience(StrictBaseModel):
    """一个项目经历。"""

    name: str = Field(
        default="",
        description="项目名称",
    )

    role: Optional[str] = Field(
        default=None,
        description="本人在项目中的角色",
    )

    date_range: DateRange = Field(default_factory=DateRange)

    project_type: Optional[str] = Field(
        default=None,
        description="例如：个人项目、课程设计、科研项目、企业项目",
    )

    background: Optional[str] = Field(
        default=None,
        description="项目背景或需要解决的问题",
    )

    description: Optional[str] = Field(
        default=None,
        description="项目总体介绍",
    )

    responsibilities: list[str] = Field(
        default_factory=list,
        description="候选人具体负责的工作",
    )

    technologies: list[str] = Field(
        default_factory=list,
        description="项目中明确使用的技术",
    )

    achievements: list[str] = Field(
        default_factory=list,
        description="项目结果和可量化成果",
    )

    repository_url: Optional[str] = None
    demo_url: Optional[str] = None

    source: list[SourceReference] = Field(
        default_factory=list,
        description="该项目对应的完整原文证据",
    )


# ============================================================
# 证书、奖项和语言能力
# ============================================================


class Certificate(StrictBaseModel):
    """证书信息。"""

    name: str
    issuing_organization: Optional[str] = None
    issue_date: Optional[str] = None
    credential_id: Optional[str] = None

    source: list[SourceReference] = Field(default_factory=list)


class Award(StrictBaseModel):
    """比赛、奖学金或荣誉奖项。"""

    name: str
    level: Optional[str] = Field(
        default=None,
        description="例如：国家级、省级、校级",
    )

    issuer: Optional[str] = None
    date: Optional[str] = None
    description: Optional[str] = None

    source: list[SourceReference] = Field(default_factory=list)


class LanguageAbility(StrictBaseModel):
    """语言能力。"""

    language: str
    level: LanguageLevel = LanguageLevel.UNKNOWN

    examination: Optional[str] = Field(
        default=None,
        description="例如：CET-6、IELTS",
    )

    score: Optional[str] = None

    source: list[SourceReference] = Field(default_factory=list)


# ============================================================
# 论文和开源成果：第一版可以暂不解析
# ============================================================


class Publication(StrictBaseModel):
    """论文、专利或技术文章。"""

    title: str
    publication_type: str = Field(
        default="other",
        description="例如：paper、patent、article",
    )

    authors: list[str] = Field(default_factory=list)
    publisher: Optional[str] = None
    publication_date: Optional[str] = None
    url: Optional[str] = None

    source: list[SourceReference] = Field(default_factory=list)


# ============================================================
# 完整结构化简历
# ============================================================


class ResumeData(StrictBaseModel):
    """解析完成后的结构化简历。"""

    schema_version: str = Field(
        default="1.0",
        description="数据结构版本",
    )

    basic_information: BasicInformation = Field(
        default_factory=BasicInformation
    )

    job_preference: Optional[JobPreference] = None

    professional_summary: Optional[str] = Field(
        default=None,
        description="简历中原本存在的个人总结，不由解析器自行生成",
    )

    education: list[EducationExperience] = Field(
        default_factory=list
    )

    skill_groups: list[SkillGroup] = Field(
        default_factory=list
    )

    skills: list[SkillItem] = Field(
        default_factory=list,
        description="去重并标准化后的完整技能列表",
    )

    work_experiences: list[WorkExperience] = Field(
        default_factory=list
    )

    projects: list[ProjectExperience] = Field(
        default_factory=list
    )

    certificates: list[Certificate] = Field(
        default_factory=list
    )

    awards: list[Award] = Field(
        default_factory=list
    )

    languages: list[LanguageAbility] = Field(
        default_factory=list
    )

    publications: list[Publication] = Field(
        default_factory=list
    )

    other_information: list[str] = Field(
        default_factory=list,
        description="暂时无法分类但可能有价值的信息",
    )

    raw_text: str = Field(
        min_length=1,
        description="经过基础清理后的完整简历文本",
    )


# ============================================================
# 解析过程信息
# ============================================================


class ParseWarning(StrictBaseModel):
    """解析过程中发现的非致命问题。"""

    code: str = Field(
        description="警告代码，例如：MISSING_SKILLS_SECTION",
    )

    message: str = Field(
        description="给开发者或用户看的中文说明",
    )

    section: Optional[str] = None


class ResumeParseMetadata(StrictBaseModel):
    """解析过程元数据。"""

    parser_name: str = Field(
        description="解析器名称，例如：rule_based_parser",
    )

    parser_version: str = "1.0"

    source_file_name: Optional[str] = None

    source_file_type: Optional[str] = Field(
        default=None,
        description="例如：txt、pdf、docx",
    )

    character_count: int = Field(default=0, ge=0)
    line_count: int = Field(default=0, ge=0)

    page_count: Optional[int] = Field(
        default=None,
        ge=1,
        description="PDF 页数；文本文件为 null",
    )

    extraction_method: Optional[str] = Field(
        default=None,
        description="例如：pdfplumber_text_layer、utf8_text",
    )

    empty_page_numbers: list[int] = Field(
        default_factory=list,
        description="没有提取到文本的 PDF 页码",
    )

    ocr_page_numbers: list[int] = Field(
        default_factory=list,
        description="通过 OCR 回退成功识别的 PDF 页码",
    )

    layout_page_numbers: list[int] = Field(
        default_factory=list,
        description="使用坐标版面恢复的 PDF 页码",
    )

    processing_time_ms: Optional[float] = Field(
        default=None,
        ge=0,
    )


class ResumeParseResult(StrictBaseModel):
    """简历解析工具的最终返回类型。"""

    success: bool

    data: Optional[ResumeData] = None

    warnings: list[ParseWarning] = Field(
        default_factory=list
    )

    errors: list[str] = Field(
        default_factory=list
    )

    metadata: ResumeParseMetadata

    @model_validator(mode="after")
    def validate_result_state(self) -> "ResumeParseResult":
        """成功结果必须携带数据，失败结果必须给出错误原因。"""
        if self.success and self.data is None:
            raise ValueError("解析成功时 data 不能为空")
        if not self.success and not self.errors:
            raise ValueError("解析失败时 errors 不能为空")
        return self
