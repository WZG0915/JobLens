"""轻量报告模型的本地与 OpenAI 抽取实现。"""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from joblens.config import load_llm_settings
from joblens.exceptions import ProviderConfigurationError, ProviderResponseError
from joblens.models import (
    EducationRecord,
    JobDescription,
    JobRequirement,
    ProjectRecord,
    RequirementImportance,
    ResumeProfile,
    SkillLevel,
    SkillRecord,
)
from joblens.parsers.job_parser import parse_job_text
from joblens.parsers.resume_parser import parse_resume_text
from joblens.prompts import JOB_SYSTEM_PROMPT, RESUME_SYSTEM_PROMPT
from joblens.schemas.job import RequirementImportance as DetailedImportance
from joblens.schemas.resume import SkillLevel as DetailedSkillLevel
from joblens.validation import validate_payload


ModelT = TypeVar("ModelT", bound=BaseModel)


def extract_resume_local(text: str, source_file: str = "") -> ResumeProfile:
    """复用富结构解析器，再适配到阶段 1 的稳定报告模型。"""
    parsed = parse_resume_text(text, source_name=source_file or None)
    if not parsed.success or parsed.data is None:
        raise ProviderResponseError("本地简历解析失败：" + "；".join(parsed.errors))
    data = parsed.data
    level_map = {
        DetailedSkillLevel.BEGINNER: SkillLevel.beginner,
        DetailedSkillLevel.FAMILIAR: SkillLevel.intermediate,
        DetailedSkillLevel.PROFICIENT: SkillLevel.advanced,
        DetailedSkillLevel.EXPERT: SkillLevel.advanced,
        DetailedSkillLevel.UNKNOWN: SkillLevel.beginner,
    }
    return ResumeProfile(
        target_roles=data.job_preference.target_roles if data.job_preference else [],
        education=[
            EducationRecord(
                description="，".join(
                    item
                    for item in (record.school, record.major, record.degree.value)
                    if item and item != "unknown"
                ) or "未分类教育经历",
                evidence=record.source[0].text if record.source else record.school,
            )
            for record in data.education
        ],
        skills=[
            SkillRecord(
                name=skill.normalized_name,
                level=level_map[skill.level],
                evidence=[source.text for source in skill.source],
            )
            for skill in data.skills
        ],
        projects=[
            ProjectRecord(
                name=project.name,
                description=project.description or project.name,
                technologies=project.technologies,
                evidence=[source.text for source in project.source],
            )
            for project in data.projects
        ],
        limitations=data.other_information,
        source_file=source_file,
    )


def extract_job_local(text: str, source_file: str = "") -> JobDescription:
    """复用富结构岗位解析器，再适配为阶段 1 的轻量岗位模型。"""
    parsed = parse_job_text(text, source_name=source_file or None)
    if not parsed.success or parsed.data is None:
        raise ProviderResponseError("本地岗位解析失败：" + "；".join(parsed.errors))
    data = parsed.data
    importance_map = {
        DetailedImportance.REQUIRED: RequirementImportance.required,
        DetailedImportance.PREFERRED: RequirementImportance.preferred,
        DetailedImportance.UNKNOWN: RequirementImportance.required,
    }
    return JobDescription(
        company=data.company.name or "未知公司",
        title=data.basic_information.title or "未知岗位",
        responsibilities=[item.description for item in data.responsibilities],
        requirements=[
            JobRequirement(
                description=item.description,
                importance=importance_map[item.importance],
                keywords=item.normalized_keywords or item.keywords,
            )
            for item in data.requirements
        ],
        source_file=source_file,
    )


def _extract_openai(
    text: str,
    model_type: type[ModelT],
    system_prompt: str,
    context: str,
) -> ModelT:
    """使用 OpenAI Structured Outputs 抽取指定 Pydantic 模型。"""
    # API Key、模型名等敏感配置仅从环境变量加载，不接受简历内容覆盖。
    settings = load_llm_settings()
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ProviderConfigurationError(
            "OpenAI 模式需要可选依赖，请安装：pip install -e .[llm]"
        ) from exc
    kwargs = {"api_key": settings.api_key}
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    client = OpenAI(**kwargs)
    try:
        response = client.responses.parse(
            model=settings.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请解析以下{context}：\n\n{text}"},
            ],
            text_format=model_type,
            store=False,
        )
    except Exception as exc:  # SDK 的异常层级会随版本变化，统一转成领域错误。
        raise ProviderResponseError(f"{context}的模型调用失败：{exc}") from exc
    parsed = getattr(response, "output_parsed", None)
    if parsed is None:
        raise ProviderResponseError(f"模型没有返回可用的{context}结构化结果")
    return validate_payload(model_type, parsed, context)


def extract_resume_openai(text: str, source_file: str = "") -> ResumeProfile:
    """调用外部模型解析简历，并补回本地来源文件路径。"""
    result = _extract_openai(text, ResumeProfile, RESUME_SYSTEM_PROMPT, "简历")
    return result.model_copy(update={"source_file": source_file})


def extract_job_openai(text: str, source_file: str = "") -> JobDescription:
    """调用外部模型解析岗位 JD，并补回本地来源文件路径。"""
    result = _extract_openai(text, JobDescription, JOB_SYSTEM_PROMPT, "岗位 JD")
    return result.model_copy(update={"source_file": source_file})
