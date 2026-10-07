"""从结构化简历生成多个互补的岗位检索查询。"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

from joblens.exceptions import DocumentParseError
from joblens.parsers import parse_resume_document
from joblens.schemas.common import SourceReference
from joblens.schemas.resume import ResumeData

from .schemas import ResumeMultiQueryResult, ResumeQueryType, ResumeSearchQuery


def _clean_text(value: str | None) -> str:
    """清除 Markdown 引用、列表符号和首尾空白。"""
    return re.sub(r"^\s*(?:[>•●▪*\-]\s*)+", "", value or "").strip()


def _unique(values: Iterable[str | None]) -> list[str]:
    """去除空值和大小写重复项。"""
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean_text(value)
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _unique_details(values: Iterable[str | None]) -> list[str]:
    """删除重复项以及已经被更长描述完整包含的短句。"""
    unique_values = _unique(values)
    return [
        value
        for index, value in enumerate(unique_values)
        if not any(
            index != other_index and value in other
            for other_index, other in enumerate(unique_values)
        )
    ]


def _skill_is_positive(skill) -> bool:
    """过滤“不会、未完成、只有概念性了解”等明确负向技能证据。"""
    negative_pattern = re.compile(
        r"尚未|未完成|没有|不会|不熟悉|仅了解|只有概念|概念性了解|无.{0,6}经验"
    )
    return not any(negative_pattern.search(source.text) for source in skill.source)


def _references(values: Iterable[SourceReference]) -> list[SourceReference]:
    """按来源位置去重原文引用。"""
    result: list[SourceReference] = []
    seen: set[tuple[object, ...]] = set()
    for value in values:
        key = (value.section, value.text, value.start_char, value.end_char, value.line_number)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def generate_resume_queries(
    resume: ResumeData,
    *,
    max_queries: int = 10,
    max_projects: int = 3,
    max_work_experiences: int = 2,
) -> ResumeMultiQueryResult:
    """按求职意向、技能、项目、工作和教育等视角生成查询。

    这里只使用已经解析且可追溯的简历字段，不把姓名、手机或邮箱放入向量
    查询。确定性生成便于测试；后续可在此结果之上增加可选的LLM改写层。
    """
    if max_queries < 1:
        raise ValueError("max_queries 必须大于 0")
    if max_projects < 0 or max_work_experiences < 0:
        raise ValueError("项目和工作查询数量不能为负数")

    candidates: list[tuple[ResumeQueryType, str, list[str], float, list[SourceReference]]] = []
    preference = resume.job_preference
    if preference:
        parts: list[str] = []
        if preference.target_roles:
            parts.append("目标岗位：" + "、".join(preference.target_roles))
        if preference.target_cities:
            parts.append("期望城市：" + "、".join(preference.target_cities))
        if preference.employment_types:
            parts.append("工作类型：" + "、".join(preference.employment_types))
        if parts:
            candidates.append(
                (
                    ResumeQueryType.TARGET_ROLE,
                    "；".join(parts),
                    _unique([*preference.target_roles, *preference.target_cities]),
                    1.0,
                    preference.source,
                )
            )

    if resume.professional_summary:
        candidates.append(
            (
                ResumeQueryType.SUMMARY,
                "候选人概况：" + _clean_text(resume.professional_summary),
                [],
                0.9,
                [],
            )
        )

    positive_skills = [skill for skill in resume.skills if _skill_is_positive(skill)]
    skills = _unique(skill.normalized_name or skill.name for skill in positive_skills)
    if skills:
        candidates.append(
            (
                ResumeQueryType.SKILLS,
                "核心技能与技术栈：" + "、".join(skills[:30]),
                skills[:30],
                1.0,
                _references(source for skill in positive_skills for source in skill.source),
            )
        )

    for project in resume.projects[:max_projects]:
        technologies = _unique(project.technologies)
        details = _unique_details(
            [
                project.background,
                project.description,
                *project.responsibilities,
                *project.achievements,
            ]
        )
        parts = [f"项目：{project.name or '未命名项目'}"]
        if project.role:
            parts.append(f"角色：{project.role}")
        if technologies:
            parts.append("技术：" + "、".join(technologies))
        if details:
            parts.append("经历：" + "；".join(details[:4]))
        candidates.append(
            (
                ResumeQueryType.PROJECT,
                "；".join(parts),
                technologies,
                0.95,
                project.source,
            )
        )

    for work in resume.work_experiences[:max_work_experiences]:
        technologies = _unique(work.technologies)
        details = _unique_details([*work.responsibilities, *work.achievements])
        parts = [f"工作经历：{work.position or '未注明职位'}"]
        if work.organization:
            parts.append(f"组织：{work.organization}")
        if technologies:
            parts.append("技术：" + "、".join(technologies))
        if details:
            parts.append("职责成果：" + "；".join(details[:4]))
        candidates.append(
            (
                ResumeQueryType.WORK_EXPERIENCE,
                "；".join(parts),
                technologies,
                0.95,
                work.source,
            )
        )

    education_parts: list[str] = []
    education_keywords: list[str] = []
    education_sources: list[SourceReference] = []
    degree_labels = {
        "high_school": "高中",
        "associate": "大专",
        "bachelor": "本科",
        "master": "硕士",
        "doctor": "博士",
        "other": "其他学历",
    }
    for education in resume.education[:2]:
        values = _unique(
            [
                degree_labels.get(education.degree.value),
                education.major,
                education.school,
                *education.relevant_courses,
            ]
        )
        if values:
            education_parts.append("、".join(values))
            education_keywords.extend(values)
            education_sources.extend(education.source)
    if education_parts:
        candidates.append(
            (
                ResumeQueryType.EDUCATION,
                "教育背景：" + "；".join(education_parts),
                _unique(education_keywords),
                0.75,
                _references(education_sources),
            )
        )

    if preference:
        availability: list[str] = []
        if preference.days_per_week is not None:
            availability.append(f"每周可到岗{preference.days_per_week}天")
        if preference.internship_duration_months is not None:
            availability.append(f"可连续实习{preference.internship_duration_months}个月")
        if preference.available_start_date:
            availability.append(f"可到岗日期{preference.available_start_date}")
        if availability:
            candidates.append(
                (
                    ResumeQueryType.AVAILABILITY,
                    "到岗条件：" + "；".join(availability),
                    availability,
                    0.8,
                    preference.source,
                )
            )

    # 极简简历可能没有被解析到任何专项字段，此时使用截断后的原文保证可检索。
    if not candidates:
        candidates.append(
            (
                ResumeQueryType.FALLBACK,
                "候选人简历：" + resume.raw_text[:1200],
                [],
                0.6,
                [],
            )
        )

    queries: list[ResumeSearchQuery] = []
    seen_texts: set[str] = set()
    for query_type, text, keywords, weight, sources in candidates:
        cleaned = text.strip()
        key = cleaned.casefold()
        if not cleaned or key in seen_texts:
            continue
        seen_texts.add(key)
        queries.append(
            ResumeSearchQuery(
                query_id=f"query_{len(queries) + 1:03d}",
                query_type=query_type,
                text=cleaned,
                keywords=_unique(keywords),
                weight=weight,
                source_references=_references(sources),
            )
        )
        if len(queries) >= max_queries:
            break

    return ResumeMultiQueryResult(
        resume_hash=hashlib.sha256(resume.raw_text.encode("utf-8")).hexdigest(),
        queries=queries,
    )


def generate_resume_queries_from_document(
    path: str | Path,
    **options: int,
) -> ResumeMultiQueryResult:
    """解析 PDF/Markdown/TXT 简历，并生成多查询。"""
    result = parse_resume_document(path)
    if not result.success or result.data is None:
        detail = "；".join(result.errors) or "解析器未返回简历数据"
        raise DocumentParseError(f"生成检索查询前简历解析失败：{detail}")
    return generate_resume_queries(result.data, **options)
