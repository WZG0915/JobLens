"""可复现的岗位要求—简历证据匹配。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from joblens.models import (
    AnalysisReport,
    JobDescription,
    MatchStatus,
    ReportSummary,
    RequirementImportance,
    RequirementMatch,
    ResumeProfile,
    SkillLevel,
)
from joblens.parsers.common_parser import SKILL_DEFINITIONS, unique


@dataclass(frozen=True)
class EvidenceItem:
    """阶段 1 匹配器使用的一条轻量简历证据。"""

    terms: frozenset[str]
    text: str
    level: int = 2


def _key(value: str) -> str:
    """生成忽略大小写和常见符号的比较键。"""
    return re.sub(r"[^a-z0-9+#\u4e00-\u9fff]", "", value.casefold())


_CANONICAL_BY_ALIAS = {
    _key(alias): definition.canonical
    for definition in SKILL_DEFINITIONS
    for alias in (definition.canonical, *definition.aliases)
}


def _canonical(value: str) -> str:
    """把技能别名转换成词表中的标准名称。"""
    return _CANONICAL_BY_ALIAS.get(_key(value), value.strip())


def _resume_evidence(resume: ResumeProfile) -> list[EvidenceItem]:
    """将技能、项目、教育和限制信息展开为统一证据列表。"""
    level_value = {
        SkillLevel.beginner: 1,
        SkillLevel.intermediate: 2,
        SkillLevel.advanced: 3,
    }
    evidence: list[EvidenceItem] = []
    for skill in resume.skills:
        canonical = _canonical(skill.name)
        for text in skill.evidence:
            evidence.append(EvidenceItem(frozenset({_key(canonical)}), text, level_value[skill.level]))
    for project in resume.projects:
        terms = {_key(_canonical(item)) for item in project.technologies}
        terms.update({_key("项目"), _key("完整项目")})
        if "课程" in project.name:
            terms.add(_key("课程项目"))
        for text in project.evidence:
            evidence.append(EvidenceItem(frozenset(terms), text, 2))
    for education in resume.education:
        evidence.append(
            EvidenceItem(
                frozenset({_key("教育"), _key("学历"), _key(education.description)}),
                education.evidence,
                2,
            )
        )
    for limitation in resume.limitations:
        # “只有概念性了解 Docker”仍是低等级正向证据；“没有/尚未”则只作为差距。
        if re.search(r"了解|概念", limitation) and not re.search(r"没有|尚未|不会|未完成", limitation):
            terms = {
                _key(definition.canonical)
                for definition in SKILL_DEFINITIONS
                if any(alias.casefold() in limitation.casefold() for alias in definition.aliases)
            }
            if terms:
                evidence.append(EvidenceItem(frozenset(terms), limitation, 1))
    return evidence


def _required_level(requirement: str) -> int:
    """根据岗位措辞估算要求的熟练程度等级。"""
    if re.search(r"精通|专家", requirement):
        return 3
    if re.search(r"熟练掌握|熟练使用|掌握", requirement):
        return 3
    if re.search(r"熟悉", requirement):
        return 2
    return 1


def _keyword_matches(keyword: str, evidence: EvidenceItem) -> bool:
    """判断岗位关键词是否被一条简历证据覆盖。"""
    canonical_key = _key(_canonical(keyword))
    keyword_key = _key(keyword)
    if canonical_key in evidence.terms or keyword_key in evidence.terms:
        return True
    evidence_key = _key(evidence.text)
    return bool(keyword_key and len(keyword_key) >= 2 and keyword_key in evidence_key)


def _is_project_requirement(description: str) -> bool:
    """判断岗位要求是否明确需要项目或作品。"""
    return bool(re.search(r"项目|作品", description))


def _match_one(
    requirement,
    resume: ResumeProfile,
    evidence_items: list[EvidenceItem],
) -> RequirementMatch:
    """匹配一条岗位要求，返回状态、关键词、证据和解释。"""
    keywords = unique([_canonical(item) for item in requirement.keywords if item.strip()])
    requirement_level = _required_level(requirement.description)
    matches_by_keyword: dict[str, list[EvidenceItem]] = {}
    for keyword in keywords:
        found = [item for item in evidence_items if _keyword_matches(keyword, item)]
        if found:
            matches_by_keyword[keyword] = found

    if _is_project_requirement(requirement.description) and resume.projects:
        project_evidence = [
            EvidenceItem(frozenset({_key("项目")}), text, 2)
            for project in resume.projects
            for text in project.evidence
        ]
        if not keywords:
            matches_by_keyword["项目"] = project_evidence

    matched_keywords = list(matches_by_keyword)
    resume_evidence = unique(
        [item.text for items in matches_by_keyword.values() for item in items]
    )
    enough_level = {
        keyword
        for keyword, items in matches_by_keyword.items()
        if any(item.level >= requirement_level for item in items)
    }
    needed = len(keywords)
    if "至少" in requirement.description:
        count_match = re.search(r"至少\s*(\d+)", requirement.description)
        needed = int(count_match.group(1)) if count_match else 1
    elif "或" in requirement.description and len(keywords) > 1:
        needed = 1
    if _is_project_requirement(requirement.description):
        needed = 1

    limitation_text = "\n".join(resume.limitations)
    explicitly_limited = [keyword for keyword in keywords if _key(keyword) in _key(limitation_text)]
    if needed > 0 and len(enough_level) >= needed:
        status = MatchStatus.matched
        explanation = "简历中的原文证据覆盖了该要求。"
    elif matched_keywords:
        status = MatchStatus.partially_matched
        explanation = "找到相关证据，但关键词覆盖或明确熟练程度尚未完全达到要求。"
    elif explicitly_limited:
        status = MatchStatus.not_matched
        explanation = "简历明确说明该能力仍是当前不足，且未找到正向项目或技能证据。"
    else:
        status = MatchStatus.insufficient_evidence
        explanation = "简历中没有找到足以支持或否定该要求的原文证据。"

    return RequirementMatch(
        requirement=requirement.description,
        importance=requirement.importance,
        status=status,
        matched_keywords=matched_keywords,
        requirement_evidence=requirement.description,
        resume_evidence=resume_evidence,
        explanation=explanation,
    )


def match_resume_to_job(resume: ResumeProfile, job: JobDescription) -> AnalysisReport:
    """逐项匹配，报告证据覆盖而非臆测录用概率。"""
    evidence_items = _resume_evidence(resume)
    matches = [_match_one(requirement, resume, evidence_items) for requirement in job.requirements]
    counts = {status: sum(item.status == status for item in matches) for status in MatchStatus}
    covered = sum(bool(item.resume_evidence) for item in matches)
    total = len(matches)
    recommendations = [
        f"补充或强化“{item.requirement}”的可核验证据。"
        for item in matches
        if item.importance == RequirementImportance.required
        and item.status != MatchStatus.matched
    ]
    if not recommendations:
        recommendations.append("核心要求已有证据覆盖；建议继续补充可量化成果和项目链接。")
    return AnalysisReport(
        resume_source=resume.source_file,
        job_source=job.source_file,
        job_company=job.company,
        job_title=job.title,
        matches=matches,
        summary=ReportSummary(
            total_requirements=total,
            matched=counts[MatchStatus.matched],
            partially_matched=counts[MatchStatus.partially_matched],
            insufficient_evidence=counts[MatchStatus.insufficient_evidence],
            not_matched=counts[MatchStatus.not_matched],
            evidence_coverage=(covered / total if total else 0.0),
        ),
        recommendations=unique(recommendations),
    )
