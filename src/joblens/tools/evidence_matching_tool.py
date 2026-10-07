"""简历—岗位证据匹配工具。

本地模式只使用确定性规则。混合模式仍由规则处理硬约束，仅把需要
语义判断的候选证据交给模型复核，并且模型只能返回已有 evidence_id。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from joblens.config import load_llm_settings
from joblens.exceptions import JobLensError, ProviderConfigurationError, ProviderResponseError
from joblens.parsers import parse_job_document, parse_resume_document
from joblens.parsers.common_parser import SKILL_DEFINITIONS, unique
from joblens.schemas import JobData, ResumeData, SourceReference
from joblens.schemas.job import (
    EducationDegree as JobEducationDegree,
    JobRequirement,
    MatchMode,
    ProficiencyLevel,
    RequirementCategory,
    RequirementImportance,
)
from joblens.schemas.match import (
    BatchEvidenceMatchResult,
    ConstraintCheck,
    EvidenceCandidate,
    EvidenceMatchResult,
    EvidenceMatchStatus,
    EvidenceMatchSummary,
    EvidencePolarity,
    EvidenceSection,
    MatchDecisionMethod,
    RankedJobMatch,
    RequirementEvidenceMatch,
)
from joblens.schemas.resume import EducationDegree as ResumeEducationDegree
from joblens.schemas.resume import SkillLevel


NEGATIVE_PATTERN = re.compile(
    r"没有|尚未|未完成|未接触|不熟悉|不会|欠缺|缺少"
)


CONCEPT_ALIASES: dict[str, tuple[str, ...]] = {
    "LLM API": ("大模型 api", "大模型api", "模型 api", "模型api", "llm api"),
    "Transformer": ("transformer", "transformers"),
    "PaddlePaddle": ("paddlepaddle", "paddle"),
    "SFT": ("sft", "监督微调"),
    "RLHF": ("rlhf", "人类反馈强化学习"),
    "reinforcement learning": ("强化学习", "reinforcement learning"),
    "model evaluation": ("模型评测", "模型评估", "测试集", "效果评估"),
    "model alignment": ("模型对齐", "对齐相关", "对齐方法"),
    "LLM application": ("大模型应用", "ai 应用", "ai应用", "模型应用开发"),
    "internship experience": ("实习经历", "正式实习", "internship experience"),
    "tool calling": ("工具调用", "tool calling", "function calling", "函数调用"),
    "multi-agent": ("多智能体", "multi-agent", "multi agent"),
    "machine learning": ("机器学习", "machine learning"),
    "deep learning": ("深度学习", "deep learning"),
    "data structures": ("数据结构", "data structures"),
    "algorithms": ("算法", "算法基础", "算法设计", "algorithms"),
    "distributed systems": ("分布式系统", "distributed systems"),
    "cloud computing": ("云计算", "cloud computing"),
    "containerization": ("容器化", "容器运行时", "containerization"),
    "GPU": ("gpu", "显存", "cuda"),
    "parallel training": ("并行训练", "并行策略", "tp / pp / dp", "tp/pp/dp"),
    "inference optimization": ("推理优化", "推理加速", "kv cache", "pagedattention"),
    "TGI": ("tgi",),
    "SGLang": ("sglang",),
    "TensorRT-LLM": ("tensorrt-llm", "tensorrt llm"),
    "gRPC": ("grpc",),
    "Kubernetes": ("kubernetes", "k8s"),
    "communication": ("沟通", "表达能力", "communication"),
    "collaboration": ("协作", "合作能力", "团队合作", "collaboration"),
    "problem solving": ("问题分析", "解决问题", "问题定位", "problem solving"),
    "paper": ("论文", "学术会议", "acl", "neurips", "icml", "iclr", "cvpr"),
}


def _key(value: str) -> str:
    """生成忽略大小写、空白和常见符号的稳定比较键。"""
    return re.sub(r"[^a-z0-9+#\u4e00-\u9fff]", "", value.casefold())


_ALIASES_BY_CANONICAL: dict[str, tuple[str, ...]] = {
    definition.canonical: (definition.canonical, *definition.aliases)
    for definition in SKILL_DEFINITIONS
}
for _canonical_name, _aliases in CONCEPT_ALIASES.items():
    existing = _ALIASES_BY_CANONICAL.get(_canonical_name, ())
    _ALIASES_BY_CANONICAL[_canonical_name] = tuple(unique([*existing, _canonical_name, *_aliases]))

_CANONICAL_BY_KEY = {
    _key(alias): canonical
    for canonical, aliases in _ALIASES_BY_CANONICAL.items()
    for alias in aliases
}


def _alias_present(alias: str, text: str) -> bool:
    """判断别名是否出现在文本中，并保护 Go、R 等短词不被误命中。"""
    lowered = text.casefold()
    escaped = re.escape(alias.casefold())
    if re.fullmatch(r"[a-z0-9+#. /-]+", alias.casefold()):
        if len(alias) <= 2 and alias.casefold() not in {"c++"}:
            return bool(re.search(rf"(?<![a-z0-9_.+#-]){escaped}(?![a-z0-9_.+#-])", lowered))
        return bool(re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", lowered))
    return alias.casefold() in lowered


def _canonical(value: str) -> str:
    """把技能或概念别名映射为标准名称。"""
    return _CANONICAL_BY_KEY.get(_key(value), value.strip())


def _terms_in_text(text: str) -> set[str]:
    """提取文本中出现的全部标准技能和概念词。"""
    return {
        canonical
        for canonical, aliases in _ALIASES_BY_CANONICAL.items()
        if any(_alias_present(alias, text) for alias in aliases)
    }


def _dedupe_sources(sources: list[SourceReference]) -> list[SourceReference]:
    """按章节、位置和文本去重来源引用，同时保持原顺序。"""
    seen: set[tuple[str, int | None, int | None, str]] = set()
    result: list[SourceReference] = []
    for source in sources:
        key = (source.section, source.start_char, source.end_char, source.text)
        if key not in seen:
            seen.add(key)
            result.append(source)
    return result


def _proficiency_from_text(text: str, fallback: int = 1) -> int:
    """将精通、掌握、熟悉、了解等措辞映射为 0—4 的等级。"""
    if re.search(r"精通|专家", text):
        return 4
    if re.search(r"熟练掌握|熟练使用|掌握", text):
        return 3
    if re.search(r"熟悉|能够使用|能使用", text):
        return 2
    if re.search(r"了解|基础|学习中|正在学习|概念", text):
        return 1
    if re.search(r"实现|开发|训练|部署|优化|构建|负责|使用", text):
        return max(fallback, 2)
    return fallback


SKILL_LEVEL_SCORE = {
    SkillLevel.UNKNOWN: 1,
    SkillLevel.BEGINNER: 1,
    SkillLevel.FAMILIAR: 2,
    SkillLevel.PROFICIENT: 3,
    SkillLevel.EXPERT: 4,
}

REQUIRED_PROFICIENCY_SCORE = {
    ProficiencyLevel.UNKNOWN: 1,
    ProficiencyLevel.KNOW: 1,
    ProficiencyLevel.FAMILIAR: 2,
    ProficiencyLevel.PROFICIENT: 3,
    ProficiencyLevel.EXPERT: 4,
}


@dataclass(frozen=True)
class _EvidenceRecord:
    """匹配器内部统一证据记录，不直接暴露为外部 Schema。"""

    evidence_id: str
    section: EvidenceSection
    text: str
    terms: frozenset[str]
    polarity: EvidencePolarity
    proficiency: int
    years: float | None
    source: tuple[SourceReference, ...]


def _month_value(value: str | None, *, current: bool = False) -> int | None:
    """把 YYYY 或 YYYY-MM 转成便于计算月份差的整数。"""
    if current:
        today = date.today()
        return today.year * 12 + today.month
    if not value:
        return None
    match = re.fullmatch(r"(\d{4})(?:-(\d{2}))?", value)
    if not match:
        return None
    return int(match.group(1)) * 12 + int(match.group(2) or 6)


def _duration_years(start: str | None, end: str | None, is_current: bool) -> float | None:
    """根据起止年月估算经历年数；信息不足时返回 None。"""
    start_month = _month_value(start)
    end_month = _month_value(end, current=is_current)
    if start_month is None or end_month is None or end_month < start_month:
        return None
    return round((end_month - start_month + 1) / 12, 2)


def _source_texts(sources: list[SourceReference], fallback: list[str]) -> list[tuple[str, list[SourceReference]]]:
    """优先使用可追溯来源文本，否则使用调用方提供的回退文本。"""
    if sources:
        return [(source.text, [source]) for source in _dedupe_sources(sources)]
    return [(text, []) for text in unique([item for item in fallback if item.strip()])]


def _build_evidence(resume: ResumeData) -> list[_EvidenceRecord]:
    """把简历各章节展开为统一、可检索且可追溯的证据集合。"""
    records: list[_EvidenceRecord] = []
    counter = 0

    def add(
        section: EvidenceSection,
        text: str,
        *,
        terms: set[str] | None = None,
        polarity: EvidencePolarity | None = None,
        proficiency: int = 1,
        years: float | None = None,
        sources: list[SourceReference] | None = None,
    ) -> None:
        """向证据集合添加一条记录，并集中处理默认值与去重。"""
        nonlocal counter
        cleaned = text.strip()
        if not cleaned:
            return
        counter += 1
        detected_polarity = polarity or (
            EvidencePolarity.NEGATIVE if NEGATIVE_PATTERN.search(cleaned) else EvidencePolarity.POSITIVE
        )
        detected_terms = {_canonical(term) for term in (terms or set()) if term.strip()}
        detected_terms.update(_terms_in_text(cleaned))
        records.append(
            _EvidenceRecord(
                evidence_id=f"ev_{counter:04d}",
                section=section,
                text=cleaned,
                terms=frozenset(detected_terms),
                polarity=detected_polarity,
                proficiency=0 if detected_polarity == EvidencePolarity.NEGATIVE else proficiency,
                years=years,
                source=tuple(_dedupe_sources(sources or [])),
            )
        )

    if resume.job_preference:
        preference = resume.job_preference
        preference_terms = set(preference.target_roles + preference.target_cities + preference.employment_types)
        fallback = [" / ".join(preference.target_roles)] if preference.target_roles else []
        for text, sources in _source_texts(preference.source, fallback):
            add(EvidenceSection.JOB_PREFERENCE, text, terms=preference_terms, sources=sources)

    for item in resume.education:
        terms = {
            item.school,
            item.major or "",
            item.degree.value,
            *item.relevant_courses,
        }
        fallback = [
            "，".join(value for value in (item.school, item.major, item.degree.value) if value),
            *item.descriptions,
        ]
        for text, sources in _source_texts(item.source, fallback):
            add(EvidenceSection.EDUCATION, text, terms=terms, proficiency=2, sources=sources)

    for item in resume.skills:
        base_terms = {item.name, item.normalized_name, *item.aliases}
        base_level = SKILL_LEVEL_SCORE[item.level]
        for text, sources in _source_texts(item.source, [item.name]):
            add(
                EvidenceSection.SKILL,
                text,
                terms=base_terms,
                proficiency=_proficiency_from_text(text, base_level),
                years=item.years_of_experience,
                sources=sources,
            )

    for item in resume.work_experiences:
        years = _duration_years(
            item.date_range.start_date,
            item.date_range.end_date,
            item.date_range.is_current,
        )
        terms = {item.organization, item.position}
        fallback = [
            " | ".join(value for value in (item.organization, item.position) if value),
            *item.responsibilities,
            *item.achievements,
        ]
        for text, sources in _source_texts(item.source, fallback):
            add(
                EvidenceSection.WORK_EXPERIENCE,
                text,
                terms=terms,
                proficiency=_proficiency_from_text(text, 2),
                years=years,
                sources=sources,
            )

    for item in resume.projects:
        years = _duration_years(
            item.date_range.start_date,
            item.date_range.end_date,
            item.date_range.is_current,
        )
        terms = {item.name, item.project_type or "项目", "项目"}
        fallback = [item.name, item.description or "", *item.responsibilities, *item.achievements]
        for text, sources in _source_texts(item.source, fallback):
            add(
                EvidenceSection.PROJECT,
                text,
                terms=terms,
                proficiency=_proficiency_from_text(text, 2),
                years=years,
                sources=sources,
            )

    for item in resume.certificates:
        for text, sources in _source_texts(item.source, [item.name]):
            add(EvidenceSection.CERTIFICATE, text, terms={item.name}, proficiency=2, sources=sources)
    for item in resume.awards:
        for text, sources in _source_texts(item.source, [item.name, item.description or ""]):
            add(EvidenceSection.AWARD, text, terms={item.name, item.level or ""}, proficiency=2, sources=sources)
    for item in resume.languages:
        terms = {item.language, item.examination or "", item.level.value}
        for text, sources in _source_texts(item.source, [item.language, item.examination or ""]):
            add(EvidenceSection.LANGUAGE, text, terms=terms, proficiency=2, sources=sources)
    for item in resume.publications:
        for text, sources in _source_texts(item.source, [item.title]):
            add(EvidenceSection.PUBLICATION, text, terms={item.title, "论文"}, proficiency=2, sources=sources)
    for text in resume.other_information:
        add(EvidenceSection.OTHER, text, proficiency=_proficiency_from_text(text, 1))
    return records


def _requirement_keywords(requirement: JobRequirement) -> list[str]:
    """合并岗位解析器关键词与概念规则，得到用于匹配的标准词。"""
    generic = {
        "经验", "能力", "方向", "项目", "专业", "学历", "要求", "计算", "方法", "技术",
    }
    parsed: list[str] = []
    for value in (requirement.normalized_keywords or requirement.keywords):
        if not value.strip():
            continue
        canonical = _canonical(value)
        is_known = _key(value) in _CANONICAL_BY_KEY or _key(canonical) in {
            _key(item) for item in _ALIASES_BY_CANONICAL
        }
        is_clear_technical_token = bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9+#. -]{1,30}", value.strip()))
        if canonical not in generic and (is_known or is_clear_technical_token):
            parsed.append(canonical)
    concepts = sorted(_terms_in_text(requirement.description))
    # 教育、专业和可用性由结构化约束处理，不把整句回退词当作技能。
    if requirement.category in {
        RequirementCategory.EDUCATION,
        RequirementCategory.MAJOR,
        RequirementCategory.AVAILABILITY,
    }:
        return unique(concepts)
    return unique([*parsed, *concepts])


def _record_matches_keyword(record: _EvidenceRecord, keyword: str) -> bool:
    """判断一条证据是否支持指定关键词或其标准别名。"""
    canonical = _canonical(keyword)
    if canonical in record.terms:
        return True
    aliases = _ALIASES_BY_CANONICAL.get(canonical, (keyword,))
    return any(_alias_present(alias, record.text) for alias in aliases)


def _matching_category(requirement: JobRequirement) -> RequirementCategory:
    """修正解析阶段难以区分的“实习经历”和“到岗条件”。"""
    if (
        requirement.category == RequirementCategory.AVAILABILITY
        and re.search(r"经历|经验", requirement.description)
        and not re.search(r"每周|到岗|连续实习\s*\d|可实习|毕业时间", requirement.description)
    ):
        return RequirementCategory.EXPERIENCE
    return requirement.category


def _record_allowed_for_category(record: _EvidenceRecord, category: RequirementCategory) -> bool:
    """限制不同要求类别可使用的证据章节，避免跨章节错误证明。"""
    if record.polarity == EvidencePolarity.NEGATIVE:
        return True
    allowed = {
        RequirementCategory.EDUCATION: {EvidenceSection.EDUCATION},
        RequirementCategory.MAJOR: {EvidenceSection.EDUCATION},
        RequirementCategory.EXPERIENCE: {
            EvidenceSection.WORK_EXPERIENCE,
            EvidenceSection.PROJECT,
        },
        RequirementCategory.PROJECT: {EvidenceSection.PROJECT},
        RequirementCategory.LANGUAGE: {EvidenceSection.LANGUAGE},
        RequirementCategory.CERTIFICATE: {EvidenceSection.CERTIFICATE},
        RequirementCategory.AVAILABILITY: {EvidenceSection.JOB_PREFERENCE},
        RequirementCategory.SOFT_SKILL: {
            EvidenceSection.WORK_EXPERIENCE,
            EvidenceSection.PROJECT,
            EvidenceSection.AWARD,
            EvidenceSection.OTHER,
        },
    }
    return record.section in allowed.get(category, set(EvidenceSection))


def _degree_rank(value: ResumeEducationDegree | JobEducationDegree) -> int:
    """将不同枚举中的学历转换成可比较的等级数字。"""
    mapping = {
        "unknown": 0,
        "other": 0,
        "high_school": 1,
        "associate": 2,
        "bachelor": 3,
        "master": 4,
        "doctor": 5,
    }
    return mapping.get(value.value, 0)


def _education_check(requirement: JobRequirement, resume: ResumeData) -> ConstraintCheck | None:
    """检查简历最高学历是否满足岗位最低学历。"""
    if requirement.minimum_degree is None:
        return None
    actual = max(resume.education, key=lambda item: _degree_rank(item.degree), default=None)
    passed = None if actual is None else _degree_rank(actual.degree) >= _degree_rank(requirement.minimum_degree)
    actual_text = actual.degree.value if actual else None
    return ConstraintCheck(
        name="minimum_degree",
        expected=requirement.minimum_degree.value,
        actual=actual_text,
        passed=passed,
        is_hard_constraint=requirement.is_hard_constraint,
        explanation=(
            "简历学历满足最低要求。"
            if passed is True
            else "简历学历低于最低要求。"
            if passed is False
            else "简历中没有可用于判断的学历证据。"
        ),
    )


def _major_check(requirement: JobRequirement, resume: ResumeData) -> ConstraintCheck | None:
    """检查教育经历中的专业是否落在岗位接受范围。"""
    if not requirement.accepted_majors:
        return None
    actual_majors = unique([item.major for item in resume.education if item.major])
    if not actual_majors:
        passed: bool | None = None
    else:
        passed = any(
            _key(expected) in _key(actual) or _key(actual) in _key(expected)
            for expected in requirement.accepted_majors
            for actual in actual_majors
        )
    return ConstraintCheck(
        name="accepted_major",
        expected="、".join(requirement.accepted_majors),
        actual="、".join(actual_majors) or None,
        passed=passed,
        is_hard_constraint=requirement.is_hard_constraint,
        explanation=(
            "简历专业与岗位接受专业匹配。"
            if passed is True
            else "简历专业与岗位列出的专业不匹配。"
            if passed is False
            else "简历中没有可用于判断的专业证据。"
        ),
    )


def _experience_check(requirement: JobRequirement, resume: ResumeData) -> ConstraintCheck | None:
    """根据工作经历日期估算年限并检查最低经验要求。"""
    if requirement.minimum_years is None:
        return None
    durations = [
        duration
        for item in resume.work_experiences
        if (
            duration := _duration_years(
                item.date_range.start_date,
                item.date_range.end_date,
                item.date_range.is_current,
            )
        )
        is not None
    ]
    explicit_skill_years = [item.years_of_experience for item in resume.skills if item.years_of_experience]
    actual_years = max([*durations, *explicit_skill_years], default=None)
    passed = None if actual_years is None else actual_years >= requirement.minimum_years
    return ConstraintCheck(
        name="minimum_years",
        expected=f"{requirement.minimum_years:g} 年",
        actual=f"{actual_years:g} 年" if actual_years is not None else None,
        passed=passed,
        is_hard_constraint=requirement.is_hard_constraint,
        explanation=(
            "简历中的明确经验年限满足要求。"
            if passed is True
            else "简历中的明确经验年限低于要求。"
            if passed is False
            else "简历没有足够日期或年限信息进行判断。"
        ),
    )


def _availability_checks(requirement: JobRequirement, resume: ResumeData, job: JobData) -> list[ConstraintCheck]:
    """检查每周到岗、实习时长和毕业年份等可用性约束。"""
    if requirement.category != RequirementCategory.AVAILABILITY or job.availability is None:
        return []
    expected = job.availability
    actual = resume.job_preference
    checks: list[ConstraintCheck] = []
    if expected.minimum_days_per_week is not None:
        actual_days = actual.days_per_week if actual else None
        passed = None if actual_days is None else actual_days >= expected.minimum_days_per_week
        checks.append(
            ConstraintCheck(
                name="minimum_days_per_week",
                expected=f"每周 {expected.minimum_days_per_week} 天",
                actual=f"每周 {actual_days} 天" if actual_days else None,
                passed=passed,
                is_hard_constraint=requirement.is_hard_constraint,
                explanation=(
                    "每周可到岗天数满足要求。"
                    if passed is True
                    else "每周可到岗天数不足。"
                    if passed is False
                    else "简历未说明每周可到岗天数。"
                ),
            )
        )
    if expected.minimum_months is not None:
        actual_months = actual.internship_duration_months if actual else None
        passed = None if actual_months is None else actual_months >= expected.minimum_months
        checks.append(
            ConstraintCheck(
                name="minimum_internship_months",
                expected=f"连续 {expected.minimum_months} 个月",
                actual=f"连续 {actual_months} 个月" if actual_months else None,
                passed=passed,
                is_hard_constraint=requirement.is_hard_constraint,
                explanation=(
                    "可实习时长满足要求。"
                    if passed is True
                    else "可实习时长不足。"
                    if passed is False
                    else "简历未说明可连续实习时长。"
                ),
            )
        )
    return checks


def _project_check(requirement: JobRequirement, resume: ResumeData) -> ConstraintCheck | None:
    """检查明确要求项目或作品时，简历是否存在项目经历。"""
    if requirement.category != RequirementCategory.PROJECT:
        return None
    passed = bool(resume.projects)
    return ConstraintCheck(
        name="project_evidence",
        expected="至少一项可核验项目经历",
        actual=f"{len(resume.projects)} 项项目" if resume.projects else None,
        passed=passed,
        is_hard_constraint=requirement.is_hard_constraint,
        explanation="简历包含项目经历。" if passed else "简历未识别到项目经历。",
    )


def _constraint_checks(requirement: JobRequirement, resume: ResumeData, job: JobData) -> list[ConstraintCheck]:
    """按要求类别组合学历、专业、经验、到岗和项目检查。"""
    checks = [
        _education_check(requirement, resume),
        _major_check(requirement, resume),
        _experience_check(requirement, resume),
        _project_check(requirement, resume),
    ]
    return [check for check in checks if check is not None] + _availability_checks(requirement, resume, job)


def _needed_keyword_count(requirement: JobRequirement, keyword_count: int) -> int:
    """根据 ALL、ANY、AT_LEAST 规则计算需要命中的关键词数量。"""
    if keyword_count == 0:
        return 0
    if requirement.match_mode == MatchMode.ANY:
        if "以及" in requirement.description and "或" in requirement.description and keyword_count > 1:
            return 2
        return 1
    if requirement.match_mode == MatchMode.AT_LEAST:
        return requirement.minimum_match_count or 1
    return keyword_count


def _candidate_from_record(
    record: _EvidenceRecord,
    matched_keywords: list[str],
    keyword_count: int,
) -> EvidenceCandidate:
    """把内部证据记录转换为公开候选证据，并计算关键词覆盖分。"""
    score = min(1.0, len(matched_keywords) / max(keyword_count, 1))
    if record.polarity == EvidencePolarity.NEGATIVE:
        score = max(score, 0.8)
    return EvidenceCandidate(
        evidence_id=record.evidence_id,
        section=record.section,
        text=record.text,
        polarity=record.polarity,
        matched_keywords=matched_keywords,
        match_score=round(score, 3),
        proficiency_score=record.proficiency,
        years_of_experience=record.years,
        source=list(record.source),
    )


def _rule_match_requirement(
    requirement: JobRequirement,
    resume: ResumeData,
    job: JobData,
    evidence_records: list[_EvidenceRecord],
) -> RequirementEvidenceMatch:
    """使用关键词、熟练度和结构化约束判断一条岗位要求。"""
    keywords = _requirement_keywords(requirement)
    matching_category = _matching_category(requirement)
    required_level = REQUIRED_PROFICIENCY_SCORE[requirement.required_proficiency]
    candidates: list[EvidenceCandidate] = []
    positive_keywords: set[str] = set()
    level_keywords: set[str] = set()
    negative_keywords: set[str] = set()

    for record in evidence_records:
        if not _record_allowed_for_category(record, matching_category):
            continue
        matched = [keyword for keyword in keywords if _record_matches_keyword(record, keyword)]
        if not matched:
            continue
        candidate = _candidate_from_record(record, matched, len(keywords))
        candidates.append(candidate)
        if record.polarity == EvidencePolarity.NEGATIVE:
            negative_keywords.update(matched)
        else:
            positive_keywords.update(matched)
            if record.proficiency >= required_level:
                level_keywords.update(matched)

    checks = _constraint_checks(requirement, resume, job)
    if checks:
        section_by_category = {
            RequirementCategory.EDUCATION: EvidenceSection.EDUCATION,
            RequirementCategory.MAJOR: EvidenceSection.EDUCATION,
            RequirementCategory.EXPERIENCE: EvidenceSection.WORK_EXPERIENCE,
            RequirementCategory.PROJECT: EvidenceSection.PROJECT,
            RequirementCategory.AVAILABILITY: EvidenceSection.JOB_PREFERENCE,
        }
        evidence_section = section_by_category.get(matching_category)
        if evidence_section is not None:
            existing_ids = {item.evidence_id for item in candidates}
            for record in evidence_records:
                if record.section != evidence_section or record.evidence_id in existing_ids:
                    continue
                candidates.append(_candidate_from_record(record, [], max(len(keywords), 1)))
                existing_ids.add(record.evidence_id)
    failed_hard = any(check.is_hard_constraint and check.passed is False for check in checks)
    unknown_hard = any(check.is_hard_constraint and check.passed is None for check in checks)
    failed_check = any(check.passed is False for check in checks)
    all_known_checks_pass = bool(checks) and all(check.passed is True for check in checks)
    needed = _needed_keyword_count(requirement, len(keywords))
    keyword_satisfied = needed == 0 or len(positive_keywords) >= needed
    level_satisfied = needed == 0 or len(level_keywords) >= needed
    matched_keywords = [keyword for keyword in keywords if keyword in positive_keywords]
    missing_keywords = [keyword for keyword in keywords if keyword not in positive_keywords]

    if failed_hard:
        status = EvidenceMatchStatus.NOT_MATCHED
        confidence = 0.98
        explanation = "至少一项岗位硬约束有明确证据表明不满足。"
    elif (keywords or checks) and keyword_satisfied and level_satisfied and not failed_check and not unknown_hard:
        status = EvidenceMatchStatus.MATCHED
        confidence = 0.95 if keywords else 0.9
        explanation = "简历中的可追溯证据覆盖了该要求及其结构化约束。"
    elif not keywords and all_known_checks_pass:
        status = EvidenceMatchStatus.MATCHED
        confidence = 0.92
        explanation = "结构化简历字段满足该要求。"
    elif positive_keywords or any(check.passed is True for check in checks):
        status = EvidenceMatchStatus.PARTIALLY_MATCHED
        confidence = 0.72
        if keyword_satisfied and not level_satisfied:
            explanation = "找到对应技能证据，但明确熟练程度尚未达到岗位要求。"
        elif failed_check:
            explanation = "找到相关证据，但至少一项非硬约束未满足。"
        else:
            explanation = "找到部分相关证据，但关键词覆盖或结构化条件尚不完整。"
    elif negative_keywords:
        status = EvidenceMatchStatus.NOT_MATCHED
        confidence = 0.9
        explanation = "简历明确说明该能力或经历目前不足，且没有找到正向证据。"
    else:
        status = EvidenceMatchStatus.INSUFFICIENT_EVIDENCE
        confidence = 0.4 if unknown_hard else 0.3
        explanation = "简历中没有找到足以支持或否定该要求的原文证据。"

    # 同一原文可能由多个技能条目产生，按文本和来源位置去重并保留信息更完整者。
    deduped: dict[tuple[str, EvidencePolarity], EvidenceCandidate] = {}
    for candidate in sorted(candidates, key=lambda item: (item.match_score, len(item.matched_keywords)), reverse=True):
        key = (candidate.text, candidate.polarity)
        if key not in deduped:
            deduped[key] = candidate
    selected = list(deduped.values())[:8]
    ambiguous_category = requirement.category in {RequirementCategory.OTHER, RequirementCategory.SOFT_SKILL}
    needs_review = bool(
        status in {EvidenceMatchStatus.PARTIALLY_MATCHED, EvidenceMatchStatus.INSUFFICIENT_EVIDENCE}
        and (selected or ambiguous_category or unknown_hard)
    )
    return RequirementEvidenceMatch(
        requirement_id=requirement.id,
        requirement=requirement.description,
        category=requirement.category,
        importance=requirement.importance,
        is_hard_constraint=requirement.is_hard_constraint,
        status=status,
        confidence=confidence,
        matched_keywords=matched_keywords,
        missing_keywords=missing_keywords,
        evidence=selected,
        constraint_checks=checks,
        requirement_source=requirement.source,
        explanation=explanation,
        needs_human_review=needs_review,
    )


class SemanticEvidenceReview(BaseModel):
    """外部复核器的受限输出；不得返回自由生成的证据文本。"""

    model_config = ConfigDict(extra="forbid")

    status: EvidenceMatchStatus
    evidence_ids: list[str] = Field(default_factory=list)
    explanation: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class EvidenceReviewer(Protocol):
    """可插拔语义复核器协议，测试和真实模型均可实现该接口。"""

    def review(self, requirement: JobRequirement, candidates: list[EvidenceCandidate]) -> SemanticEvidenceReview:
        """复核一条要求，只允许引用传入的候选证据 ID。"""


class OpenAIEvidenceReviewer:
    """可选的 OpenAI Structured Outputs 语义复核器。"""

    def __init__(self) -> None:
        """读取环境配置并创建 OpenAI 客户端。"""
        settings = load_llm_settings()
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderConfigurationError(
                "混合匹配模式需要可选依赖，请安装：pip install -e .[llm]"
            ) from exc
        kwargs = {"api_key": settings.api_key}
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        self._client = OpenAI(**kwargs)
        self._model = settings.model

    def review(self, requirement: JobRequirement, candidates: list[EvidenceCandidate]) -> SemanticEvidenceReview:
        """让模型仅在给定候选 ID 中选择证据并返回结构化结论。"""
        payload = {
            "requirement": requirement.model_dump(mode="json"),
            "candidate_evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "section": item.section.value,
                    "text": item.text,
                    "polarity": item.polarity.value,
                }
                for item in candidates
            ],
        }
        try:
            response = self._client.responses.parse(
                model=self._model,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "你是严格的求职证据审查器。只能根据给定候选证据判断，不得补写、"
                            "推测或改写证据。evidence_ids 只能选择输入中真实存在的 ID。"
                            "证据仅间接相关时返回 partially_matched；没有证据时返回 insufficient_evidence。"
                        ),
                    },
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                text_format=SemanticEvidenceReview,
                store=False,
            )
        except Exception as exc:
            raise ProviderResponseError(f"证据语义复核失败：{exc}") from exc
        reviewed = getattr(response, "output_parsed", None)
        if reviewed is None:
            raise ProviderResponseError("证据语义复核没有返回结构化结果")
        allowed_ids = {item.evidence_id for item in candidates}
        if any(evidence_id not in allowed_ids for evidence_id in reviewed.evidence_ids):
            raise ProviderResponseError("证据语义复核返回了不存在的 evidence_id")
        return reviewed


def _apply_semantic_review(
    match: RequirementEvidenceMatch,
    requirement: JobRequirement,
    reviewer: EvidenceReviewer,
) -> RequirementEvidenceMatch:
    """将语义复核结果安全合并到规则结果，且不覆盖硬约束失败。"""
    if not match.needs_human_review or not match.evidence:
        return match
    review = reviewer.review(requirement, match.evidence)
    # 模型不能覆盖规则已经确认的硬约束失败。
    hard_failed = any(check.is_hard_constraint and check.passed is False for check in match.constraint_checks)
    if hard_failed:
        return match
    selected_ids = set(review.evidence_ids)
    evidence = [item for item in match.evidence if item.evidence_id in selected_ids]
    return match.model_copy(
        update={
            "status": review.status,
            "decision_method": MatchDecisionMethod.HYBRID,
            "confidence": review.confidence,
            "evidence": evidence,
            "explanation": review.explanation,
            "needs_human_review": review.confidence < 0.75,
        }
    )


def _summary(matches: list[RequirementEvidenceMatch]) -> EvidenceMatchSummary:
    """汇总匹配数量、证据覆盖率、硬约束状态和加权得分。"""
    counts = {status: sum(item.status == status for item in matches) for status in EvidenceMatchStatus}
    hard = [item for item in matches if item.is_hard_constraint]
    hard_failed = sum(item.status == EvidenceMatchStatus.NOT_MATCHED for item in hard)
    hard_passed = sum(item.status == EvidenceMatchStatus.MATCHED for item in hard)
    hard_unknown = len(hard) - hard_failed - hard_passed
    covered = sum(bool(item.evidence) or any(check.passed is not None for check in item.constraint_checks) for item in matches)
    status_score = {
        EvidenceMatchStatus.MATCHED: 1.0,
        EvidenceMatchStatus.PARTIALLY_MATCHED: 0.5,
        EvidenceMatchStatus.INSUFFICIENT_EVIDENCE: 0.0,
        EvidenceMatchStatus.NOT_MATCHED: 0.0,
    }
    weights = [
        (3.0 if item.is_hard_constraint else 2.0 if item.importance == RequirementImportance.REQUIRED else 1.0)
        for item in matches
    ]
    denominator = sum(weights)
    weighted = (
        sum(weight * status_score[item.status] for item, weight in zip(matches, weights)) / denominator * 100
        if denominator
        else 0.0
    )
    if hard_failed:
        weighted = min(weighted, 59.0)
    return EvidenceMatchSummary(
        total_requirements=len(matches),
        matched=counts[EvidenceMatchStatus.MATCHED],
        partially_matched=counts[EvidenceMatchStatus.PARTIALLY_MATCHED],
        insufficient_evidence=counts[EvidenceMatchStatus.INSUFFICIENT_EVIDENCE],
        not_matched=counts[EvidenceMatchStatus.NOT_MATCHED],
        hard_constraints_total=len(hard),
        hard_constraints_passed=hard_passed,
        hard_constraints_failed=hard_failed,
        hard_constraints_unknown=hard_unknown,
        evidence_coverage=round(covered / len(matches), 4) if matches else 0.0,
        weighted_score=round(weighted, 2),
        review_required=sum(item.needs_human_review for item in matches),
    )


def _recommendations(matches: list[RequirementEvidenceMatch]) -> list[str]:
    """针对未完全满足的必需要求生成简历改进建议。"""
    recommendations: list[str] = []
    for item in matches:
        if item.importance != RequirementImportance.REQUIRED or item.status == EvidenceMatchStatus.MATCHED:
            continue
        if item.status == EvidenceMatchStatus.NOT_MATCHED:
            prefix = "优先补足硬性差距" if item.is_hard_constraint else "补足能力差距"
        elif item.status == EvidenceMatchStatus.PARTIALLY_MATCHED:
            prefix = "补充更强证据"
        else:
            prefix = "补充可核验证据"
        recommendations.append(f"{prefix}：{item.requirement}")
    if not recommendations:
        recommendations.append("核心要求已有证据覆盖；建议继续补充量化成果、项目链接和职责边界。")
    return unique(recommendations)


def match_resume_job_data(
    resume: ResumeData,
    job: JobData,
    *,
    resume_source: str | None = None,
    job_source: str | None = None,
    mode: str = "local",
    reviewer: EvidenceReviewer | None = None,
) -> EvidenceMatchResult:
    """匹配已经解析好的富结构简历与岗位数据。"""
    started = time.perf_counter()
    if mode not in {"local", "hybrid"}:
        raise ValueError("mode 仅支持 local 或 hybrid")
    if not job.requirements:
        return EvidenceMatchResult(
            success=False,
            resume_source=resume_source,
            job_source=job_source,
            job_company=job.company.name,
            job_title=job.basic_information.title,
            errors=["岗位中没有可匹配的任职要求"],
        )
    evidence_records = _build_evidence(resume)
    matches = [
        _rule_match_requirement(requirement, resume, job, evidence_records)
        for requirement in job.requirements
    ]
    warnings: list[str] = []
    if not evidence_records:
        warnings.append("简历中没有构建出可匹配证据")
    if mode == "hybrid":
        active_reviewer = reviewer or OpenAIEvidenceReviewer()
        reviewed_matches: list[RequirementEvidenceMatch] = []
        for requirement, match in zip(job.requirements, matches):
            try:
                reviewed_matches.append(_apply_semantic_review(match, requirement, active_reviewer))
            except ProviderResponseError as exc:
                warnings.append(f"{requirement.id} 模型复核失败，保留规则结果：{exc}")
                reviewed_matches.append(match)
        matches = reviewed_matches
    elapsed_ms = (time.perf_counter() - started) * 1000
    return EvidenceMatchResult(
        success=True,
        resume_source=resume_source,
        job_source=job_source,
        job_company=job.company.name,
        job_title=job.basic_information.title,
        matches=matches,
        summary=_summary(matches),
        recommendations=_recommendations(matches),
        warnings=unique(warnings),
        processing_time_ms=elapsed_ms,
    )


def match_resume_job_files(
    resume_path: str | Path,
    job_path: str | Path,
    *,
    mode: str = "local",
    output_path: str | Path | None = None,
    reviewer: EvidenceReviewer | None = None,
) -> EvidenceMatchResult:
    """解析两个文件并生成一份证据匹配报告。"""
    resume_file = Path(resume_path).expanduser().resolve()
    job_file = Path(job_path).expanduser().resolve()
    try:
        resume_result = parse_resume_document(resume_file)
        job_result = parse_job_document(job_file)
        if not resume_result.success or resume_result.data is None:
            raise ValueError("简历解析失败：" + "；".join(resume_result.errors))
        if not job_result.success or job_result.data is None:
            raise ValueError("岗位解析失败：" + "；".join(job_result.errors))
        result = match_resume_job_data(
            resume_result.data,
            job_result.data,
            resume_source=str(resume_file),
            job_source=str(job_file),
            mode=mode,
            reviewer=reviewer,
        )
        parser_warnings = [warning.message for warning in resume_result.warnings]
        parser_warnings.extend(warning.message for warning in job_result.warnings)
        if parser_warnings:
            result = result.model_copy(update={"warnings": unique([*result.warnings, *parser_warnings])})
    except (JobLensError, OSError, ValueError) as exc:
        result = EvidenceMatchResult(
            success=False,
            resume_source=str(resume_file),
            job_source=str(job_file),
            errors=[str(exc)],
        )
    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result


def match_resume_to_job_directory(
    resume_path: str | Path,
    jobs_dir: str | Path,
    *,
    mode: str = "local",
    output_dir: str | Path | None = None,
    reviewer: EvidenceReviewer | None = None,
) -> BatchEvidenceMatchResult:
    """用一份简历批量匹配目录中的 Markdown/TXT 岗位并按得分排名。"""
    started = time.perf_counter()
    resume_file = Path(resume_path).expanduser().resolve()
    directory = Path(jobs_dir).expanduser().resolve()
    files = sorted([*directory.glob("*.md"), *directory.glob("*.markdown"), *directory.glob("*.txt")])
    destination_dir = Path(output_dir).expanduser().resolve() if output_dir else None
    if destination_dir:
        destination_dir.mkdir(parents=True, exist_ok=True)
    ranked: list[tuple[EvidenceMatchResult, Path | None]] = []
    errors: list[str] = []
    for job_file in files:
        output_path = destination_dir / f"{job_file.stem}.json" if destination_dir else None
        result = match_resume_job_files(
            resume_file,
            job_file,
            mode=mode,
            output_path=output_path,
            reviewer=reviewer,
        )
        if result.success and result.summary is not None:
            ranked.append((result, output_path))
        else:
            errors.append(f"{job_file.name}: {'；'.join(result.errors)}")
    ranked.sort(
        key=lambda pair: (
            pair[0].summary.hard_constraints_failed if pair[0].summary else 999,
            -(pair[0].summary.weighted_score if pair[0].summary else 0),
            pair[0].job_title or "",
        )
    )
    rankings = [
        RankedJobMatch(
            rank=index,
            job_source=result.job_source or "",
            job_company=result.job_company or "未知公司",
            job_title=result.job_title or "未知岗位",
            weighted_score=result.summary.weighted_score,
            hard_constraints_failed=result.summary.hard_constraints_failed,
            review_required=result.summary.review_required,
            result_file=str(output_path) if output_path else None,
        )
        for index, (result, output_path) in enumerate(ranked, start=1)
        if result.summary is not None
    ]
    elapsed_ms = (time.perf_counter() - started) * 1000
    return BatchEvidenceMatchResult(
        success=not errors,
        resume_source=str(resume_file),
        job_count=len(files),
        successful_jobs=len(ranked),
        failed_jobs=len(errors),
        rankings=rankings,
        errors=errors,
        processing_time_ms=elapsed_ms,
    )
