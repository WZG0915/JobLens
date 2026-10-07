"""把结构化岗位 JD 转换为可检索的语义块。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from joblens.exceptions import DocumentParseError, DocumentReadError
from joblens.parsers import parse_job_document
from joblens.schemas.common import SourceReference
from joblens.schemas.job import JobData, SalaryPeriod

from .schemas import (
    JobChunk,
    JobChunkingResult,
    JobChunkSection,
    JobCorpusChunkingResult,
)


SUPPORTED_JOB_SUFFIXES = {".md", ".markdown", ".txt"}


def _unique_strings(values: Iterable[str | None]) -> list[str]:
    """按首次出现顺序去重，并忽略空字符串。"""
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = (value or "").strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _unique_references(values: Iterable[SourceReference]) -> list[SourceReference]:
    """去除同一来源字段产生的重复原文引用。"""
    result: list[SourceReference] = []
    seen: set[tuple[object, ...]] = set()
    for value in values:
        key = (
            value.section,
            value.text,
            value.start_char,
            value.end_char,
            value.line_number,
        )
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _content_hash(job: JobData) -> str:
    """根据解析器输出的标准化原文计算稳定哈希。"""
    return hashlib.sha256(job.raw_text.encode("utf-8")).hexdigest()


def _salary_text(job: JobData) -> str | None:
    """把薪资模型转换为适合检索的简短文本。"""
    salary = job.basic_information.salary
    if salary is None:
        return None
    if salary.raw_text:
        return salary.raw_text
    if salary.minimum is None and salary.maximum is None:
        return None
    period_labels = {
        SalaryPeriod.DAILY: "每天",
        SalaryPeriod.MONTHLY: "每月",
        SalaryPeriod.YEARLY: "每年",
        SalaryPeriod.UNKNOWN: "",
    }
    if salary.minimum is not None and salary.maximum is not None:
        amount = f"{salary.minimum:g}-{salary.maximum:g} {salary.currency}"
    else:
        value = salary.minimum if salary.minimum is not None else salary.maximum
        amount = f"{value:g} {salary.currency}"
    return f"{amount}{period_labels[salary.period]}"


def _basic_text(job: JobData) -> str:
    """把岗位基本字段组织成可读文本，不把空字段写进索引。"""
    basic = job.basic_information
    company = job.company
    pairs: list[tuple[str, str | None]] = [
        ("岗位名称", basic.title),
        ("公司", company.name),
        ("部门", basic.department),
        ("城市", "、".join(basic.cities) if basic.cities else None),
        ("地址", basic.address),
        ("用工类型", basic.employment_type.value),
        ("经验等级", basic.experience_level.value),
        ("办公方式", basic.work_mode.value),
        ("薪资", _salary_text(job)),
        ("行业", company.industry),
        ("公司规模", company.company_size),
        ("融资阶段", company.financing_stage),
        ("公司简介", company.description),
    ]
    # unknown 只是解析器默认值，不应成为检索语义的一部分。
    lines = [
        f"{label}：{value}"
        for label, value in pairs
        if value and value != "unknown"
    ]
    return "\n".join(lines)


def _availability_text(job: JobData) -> str:
    """优先保留原文时间要求，缺失时再由结构化字段生成。"""
    availability = job.availability
    if availability is None:
        return ""
    if availability.raw_text:
        return availability.raw_text
    values: list[str] = []
    if availability.minimum_days_per_week is not None:
        values.append(f"每周至少到岗 {availability.minimum_days_per_week} 天")
    if availability.minimum_months is not None:
        values.append(f"至少连续实习 {availability.minimum_months} 个月")
    if availability.start_date:
        values.append(f"到岗日期：{availability.start_date}")
    if availability.graduation_years:
        values.append("毕业年份：" + "、".join(map(str, availability.graduation_years)))
    return "；".join(values)


def chunk_job(
    job: JobData,
    *,
    source_path: str | Path,
    job_id: str | None = None,
) -> JobChunkingResult:
    """按业务语义分块，而不是按固定字符数切断岗位内容。

    分块粒度：一个基本信息块、一个摘要块、每条职责一个块、每条要求
    一个块、一个到岗要求块、每项福利和其他信息各一个块。
    """
    path = Path(source_path).resolve()
    resolved_job_id = (job_id or path.stem).strip()
    if not resolved_job_id:
        raise ValueError("job_id 不能为空")

    source_name = str(path)
    source_hash = _content_hash(job)
    title = job.basic_information.title
    company = job.company.name
    chunks: list[JobChunk] = []

    def append_chunk(
        *,
        suffix: str,
        section: JobChunkSection,
        text: str,
        keywords: Iterable[str | None] = (),
        source_references: Iterable[SourceReference] = (),
        responsibility_id: str | None = None,
        requirement_id: str | None = None,
        requirement_category=None,
        requirement_importance=None,
        is_hard_constraint: bool = False,
    ) -> None:
        """补齐公共元数据并跳过空块。"""
        cleaned_text = text.strip()
        if not cleaned_text:
            return
        chunks.append(
            JobChunk(
                chunk_id=f"{resolved_job_id}:{suffix}",
                job_id=resolved_job_id,
                source_path=source_name,
                source_hash=source_hash,
                section=section,
                text=cleaned_text,
                title=title,
                company=company,
                keywords=_unique_strings(keywords),
                source_references=_unique_references(source_references),
                responsibility_id=responsibility_id,
                requirement_id=requirement_id,
                requirement_category=requirement_category,
                requirement_importance=requirement_importance,
                is_hard_constraint=is_hard_constraint,
            )
        )

    basic_text = _basic_text(job)
    append_chunk(
        suffix="basic",
        section=JobChunkSection.BASIC,
        text=basic_text,
        keywords=[title, company, *job.basic_information.cities, job.company.industry],
        source_references=[*job.basic_information.source, *job.company.source],
    )

    if job.summary:
        append_chunk(
            suffix="summary",
            section=JobChunkSection.SUMMARY,
            text=job.summary,
        )

    for responsibility in job.responsibilities:
        append_chunk(
            suffix=f"responsibility:{responsibility.id}",
            section=JobChunkSection.RESPONSIBILITY,
            text=responsibility.description,
            keywords=responsibility.normalized_keywords or responsibility.keywords,
            source_references=responsibility.source,
            responsibility_id=responsibility.id,
        )

    for requirement in job.requirements:
        append_chunk(
            suffix=f"requirement:{requirement.id}",
            section=JobChunkSection.REQUIREMENT,
            text=requirement.description,
            keywords=requirement.normalized_keywords or requirement.keywords,
            source_references=requirement.source,
            requirement_id=requirement.id,
            requirement_category=requirement.category,
            requirement_importance=requirement.importance,
            is_hard_constraint=requirement.is_hard_constraint,
        )

    if job.availability is not None:
        append_chunk(
            suffix="availability",
            section=JobChunkSection.AVAILABILITY,
            text=_availability_text(job),
            keywords=["到岗要求", "实习时间", *map(str, job.availability.graduation_years)],
            source_references=job.availability.source,
        )

    for index, benefit in enumerate(job.benefits, start=1):
        text = benefit.name
        if benefit.description:
            text = f"{benefit.name}：{benefit.description}"
        append_chunk(
            suffix=f"benefit:{index:03d}",
            section=JobChunkSection.BENEFIT,
            text=text,
            keywords=[benefit.name],
            source_references=benefit.source,
        )

    for index, information in enumerate(job.other_information, start=1):
        append_chunk(
            suffix=f"other:{index:03d}",
            section=JobChunkSection.OTHER,
            text=information,
        )

    return JobChunkingResult(
        job_id=resolved_job_id,
        source_path=source_name,
        source_hash=source_hash,
        chunks=chunks,
    )


def chunk_job_document(path: str | Path, *, job_id: str | None = None) -> JobChunkingResult:
    """读取并解析一份岗位文件，然后执行结构化分块。"""
    document_path = Path(path)
    result = parse_job_document(document_path)
    if not result.success or result.data is None:
        detail = "；".join(result.errors) or "解析器未返回岗位数据"
        raise DocumentParseError(f"岗位分块前解析失败：{detail}")
    return chunk_job(result.data, source_path=document_path, job_id=job_id)


def chunk_job_corpus(directory: str | Path) -> JobCorpusChunkingResult:
    """批量分块目录内的 Markdown/TXT 岗位文件。"""
    source_directory = Path(directory).resolve()
    if not source_directory.exists():
        raise DocumentReadError(f"岗位目录不存在：{source_directory}")
    if not source_directory.is_dir():
        raise DocumentReadError(f"岗位路径不是目录：{source_directory}")

    paths = sorted(
        path
        for path in source_directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_JOB_SUFFIXES
    )
    if not paths:
        raise DocumentReadError(f"岗位目录中没有可分块的 Markdown/TXT 文件：{source_directory}")
    jobs = [chunk_job_document(path) for path in paths]
    return JobCorpusChunkingResult(
        source_directory=str(source_directory),
        jobs=jobs,
    )
