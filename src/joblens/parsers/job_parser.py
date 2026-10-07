"""面向中文 Markdown/TXT 岗位 JD 的确定性解析器。"""

from __future__ import annotations

import re
import time
from pathlib import Path

from joblens.io import read_document
from joblens.parsers.common_parser import (
    LocatedLine,
    clean_text,
    content_lines,
    content_text,
    extract_skills,
    first_h1,
    iter_located_lines,
    section_lines,
    unique,
)
from joblens.schemas.job import (
    AvailabilityRequirement,
    CompanyInformation,
    EducationDegree,
    EmploymentType,
    ExperienceLevel,
    JobBasicInformation,
    JobBenefit,
    JobData,
    JobParseMetadata,
    JobParseResult,
    JobParseWarning,
    JobRequirement,
    JobResponsibility,
    MatchMode,
    ProficiencyLevel,
    RequirementCategory,
    RequirementImportance,
    SalaryPeriod,
    SalaryRange,
    WorkMode,
)


RESPONSIBILITY_SECTIONS = {
    "工作内容", "岗位职责", "工作职责", "职位描述", "工作描述", "职责",
    "job responsibilities", "responsibilities",
}
REQUIREMENT_SECTIONS = {
    "任职要求", "岗位要求", "职位要求", "任职资格", "任职条件", "要求",
    "requirements", "qualifications",
}
BENEFIT_SECTIONS = {
    "福利", "职位福利", "福利待遇", "我们提供", "岗位亮点", "benefits",
}


def _top_content_lines(text: str) -> list[LocatedLine]:
    """返回第一个二级标题前的非标题内容，供基础字段抽取。"""
    result: list[LocatedLine] = []
    offset = 0
    for number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if stripped.startswith("## "):
            break
        if not stripped or stripped.startswith("# "):
            offset += len(raw_line) + 1
            continue
        start = offset + raw_line.find(stripped)
        result.append(
            LocatedLine(
                section="基本信息",
                text=stripped,
                line_number=number,
                start_char=start,
                end_char=start + len(stripped),
            )
        )
        offset += len(raw_line) + 1
    return result


def _labeled_value(text: str, *labels: str) -> tuple[str | None, list]:
    """从顶部基本信息中读取“标签：值”，并保留原文来源。"""
    label_pattern = "|".join(re.escape(label) for label in labels)
    for line in _top_content_lines(text):
        match = re.match(rf"^(?:{label_pattern})\s*[：:]\s*(.+?)\s*$", line.text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            return value, [line.reference(value)]
    return None, []


def _split_cities(value: str | None) -> list[str]:
    """把多个工作城市按常见中英文分隔符拆分并去重。"""
    if not value:
        return []
    return unique(
        [
            item.strip()
            for item in re.split(r"[、,，/|｜]", value)
            if item.strip()
        ]
    )


def _header_parts(text: str) -> tuple[str, str, list]:
    """从一级标题解析公司和岗位名称。"""
    heading = first_h1(text)
    if heading is None:
        return "", "", []
    value = content_text(heading)
    match = re.match(r"^(.+?)\s*[—–|｜]\s*(.+)$", value)
    if match:
        return match.group(1).strip(), match.group(2).strip(), [heading.reference(value)]
    return "", value, [heading.reference(value)]


def _employment_type(title: str, text: str) -> EmploymentType:
    """根据标题和正文关键词判断实习、校招、全职等用工类型。"""
    combined = f"{title} {text}".lower()
    if "实习" in combined:
        return EmploymentType.INTERNSHIP
    if "校招" in combined or "校园招聘" in combined:
        return EmploymentType.CAMPUS_RECRUITMENT
    if "兼职" in combined:
        return EmploymentType.PART_TIME
    if "合同" in combined:
        return EmploymentType.CONTRACT
    if "全职" in combined:
        return EmploymentType.FULL_TIME
    return EmploymentType.UNKNOWN


def _experience_level(title: str, text: str = "") -> ExperienceLevel:
    """根据岗位措辞和最低年限推断经验等级。"""
    combined = f"{title} {text}".lower()
    if "实习" in combined or "intern" in combined:
        return ExperienceLevel.INTERN
    if any(value in combined for value in ("应届", "校招", "初级", "助理", "junior", "graduate")):
        return ExperienceLevel.ENTRY
    if any(value in combined for value in ("高级", "资深", "senior")):
        return ExperienceLevel.SENIOR
    if "专家" in combined or "expert" in combined:
        return ExperienceLevel.EXPERT
    years = re.search(r"(\d+)\s*[-~—–至]\s*(\d+)\s*年", combined)
    if years:
        minimum = int(years.group(1))
        if minimum >= 5:
            return ExperienceLevel.SENIOR
        if minimum >= 3:
            return ExperienceLevel.MIDDLE
        if minimum >= 1:
            return ExperienceLevel.JUNIOR
    return ExperienceLevel.UNKNOWN


def _work_mode(text: str) -> WorkMode:
    """识别现场、远程或混合办公模式。"""
    if "混合办公" in text:
        return WorkMode.HYBRID
    if "远程" in text:
        return WorkMode.REMOTE
    if "现场办公" in text or "坐班" in text:
        return WorkMode.ON_SITE
    return WorkMode.UNKNOWN


def _salary(text: str) -> SalaryRange | None:
    """解析薪资上下限、计薪周期和年薪月数。"""
    salary_line, _ = _labeled_value(text, "薪资", "薪酬", "工资", "薪资范围")
    search_text = salary_line or text
    matches = re.finditer(
        r"(?P<minimum>\d+(?:\.\d+)?)\s*[-~—–至]\s*(?P<maximum>\d+(?:\.\d+)?)\s*"
        r"(?P<unit>[kK千万元]*)\s*(?:元)?\s*(?P<period>/天|/日|/月|/年|每天|每月|每年)?",
        search_text,
    )
    match = next(
        (
            candidate
            for candidate in matches
            if candidate.group("unit") or candidate.group("period")
        ),
        None,
    )
    if not match:
        return None
    factor = 1.0
    unit = match.group("unit").lower()
    if "k" in unit or "千" in unit:
        factor = 1000.0
    elif "万" in unit:
        factor = 10000.0
    period_text = match.group("period") or ""
    period = SalaryPeriod.UNKNOWN
    if "天" in period_text or "日" in period_text:
        period = SalaryPeriod.DAILY
    elif "月" in period_text:
        period = SalaryPeriod.MONTHLY
    elif "年" in period_text:
        period = SalaryPeriod.YEARLY
    months_match = re.search(r"(?:[·x×*]|\s)(\d{2})\s*薪", search_text, re.IGNORECASE)
    return SalaryRange(
        minimum=float(match.group("minimum")) * factor,
        maximum=float(match.group("maximum")) * factor,
        period=period,
        months_per_year=int(months_match.group(1)) if months_match else None,
        raw_text=match.group(0),
    )


def _parse_responsibilities(text: str) -> list[JobResponsibility]:
    """将岗位职责章节的每一行转换为带来源的职责对象。"""
    lines = content_lines(section_lines(text, RESPONSIBILITY_SECTIONS))
    return [
        JobResponsibility(
            id=f"resp_{index:03d}",
            description=content_text(line),
            keywords=[definition.canonical for definition in extract_skills(content_text(line))],
            normalized_keywords=[definition.canonical for definition in extract_skills(content_text(line))],
            source=[line.reference(content_text(line))],
        )
        for index, line in enumerate(lines, start=1)
    ]


def _importance(value: str) -> RequirementImportance:
    """区分必需要求和优先/加分要求。"""
    if re.search(r"优先|加分|更佳|preferred", value, re.IGNORECASE):
        return RequirementImportance.PREFERRED
    return RequirementImportance.REQUIRED


def _proficiency(value: str) -> ProficiencyLevel:
    """从“了解、熟悉、掌握、精通”等词推断熟练度。"""
    if re.search(r"精通|专家", value):
        return ProficiencyLevel.EXPERT
    if re.search(r"熟练掌握|熟练使用|掌握", value):
        return ProficiencyLevel.PROFICIENT
    if re.search(r"熟悉", value):
        return ProficiencyLevel.FAMILIAR
    if re.search(r"了解|基础", value):
        return ProficiencyLevel.KNOW
    return ProficiencyLevel.UNKNOWN


def _category(value: str, skill_categories: list[str]) -> RequirementCategory:
    """根据规则和技能类别判断岗位要求所属类别。"""
    if re.search(r"学历|本科|硕士|博士|大专", value):
        return RequirementCategory.EDUCATION
    if "专业" in value and re.search(r"计算机|软件|数学|统计|电子|自动化", value):
        return RequirementCategory.MAJOR
    if re.search(r"实习|到岗|每周|连续.*月|毕业", value):
        return RequirementCategory.AVAILABILITY
    if re.search(r"项目|作品", value):
        return RequirementCategory.PROJECT
    if re.search(r"英语|雅思|托福|CET|语言能力", value, re.IGNORECASE):
        return RequirementCategory.LANGUAGE
    if re.search(r"经验|年", value) and not skill_categories:
        return RequirementCategory.EXPERIENCE
    if skill_categories:
        order = [
            "programming_language",
            "framework",
            "database",
            "ai_ml",
            "tool",
            "platform",
            "domain_knowledge",
            "soft_skill",
        ]
        selected = next((item for item in order if item in skill_categories), "other")
        return RequirementCategory(selected)
    return RequirementCategory.OTHER


def _fallback_keywords(value: str, category: RequirementCategory) -> list[str]:
    """技能词典未命中时，从要求原文中提取少量回退关键词。"""
    keyword_rules = {
        RequirementCategory.PROJECT: ("课程项目", "个人项目", "完整项目", "项目"),
        RequirementCategory.EDUCATION: ("博士", "硕士", "本科", "大专"),
        RequirementCategory.MAJOR: ("计算机", "软件工程", "人工智能", "数学", "统计"),
        RequirementCategory.LANGUAGE: ("英语", "CET-4", "CET-6", "雅思", "托福"),
        RequirementCategory.AVAILABILITY: ("每周", "连续实习", "到岗", "毕业"),
        RequirementCategory.EXPERIENCE: ("经验",),
    }
    found = [keyword for keyword in keyword_rules.get(category, ()) if keyword.lower() in value.lower()]
    if found:
        return found
    phrases = re.findall(r"[A-Za-z][A-Za-z0-9+#.-]*|[\u4e00-\u9fff]{2,8}", value)
    stop_words = {"熟悉", "了解", "掌握", "能够", "具备", "优先", "基础", "良好", "至少", "方向", "一个"}
    return [phrase for phrase in phrases if phrase not in stop_words][:4]


def _minimum_degree(value: str) -> EducationDegree | None:
    """识别岗位要求中的最低学历。"""
    for label, degree in (
        ("博士", EducationDegree.DOCTOR),
        ("硕士", EducationDegree.MASTER),
        ("本科", EducationDegree.BACHELOR),
        ("大专", EducationDegree.ASSOCIATE),
        ("高中", EducationDegree.HIGH_SCHOOL),
    ):
        if label in value:
            return degree
    return None


def _atomic_requirement_parts(value: str) -> list[str]:
    """只在明显的新约束动词处拆分，避免把技术枚举错误切开。"""
    parts = re.split(
        r"[；;]|，(?=(?:必须|需要|具备|具有|能够|能|熟悉|了解|掌握|有|对|计算机|软件|数学|统计))",
        value,
    )
    return [part.strip(" ，,；;") for part in parts if part.strip(" ，,；;")]


def _parse_requirements(text: str) -> list[JobRequirement]:
    """拆分并结构化任职要求，生成匹配模式、硬约束和来源证据。"""
    lines = content_lines(section_lines(text, REQUIREMENT_SECTIONS))
    requirements: list[JobRequirement] = []
    atomic_items = [
        (line, part)
        for line in lines
        for part in _atomic_requirement_parts(content_text(line))
    ]
    for index, (line, value) in enumerate(atomic_items, start=1):
        definitions = extract_skills(value)
        category = _category(value, [definition.category for definition in definitions])
        keywords = unique([definition.canonical for definition in definitions])
        if not keywords:
            keywords = unique(_fallback_keywords(value, category))
        match_mode = MatchMode.ALL
        minimum_count = None
        if re.search(r"至少(?:一个|一项|1个|1项)", value):
            match_mode = MatchMode.AT_LEAST
            minimum_count = 1
        elif "或" in value and len(keywords) > 1:
            match_mode = MatchMode.ANY
        years_match = re.search(r"(\d+(?:\.\d+)?)\s*年", value)
        accepted_majors = [
            major
            for major in ("计算机", "软件工程", "人工智能", "数学", "统计", "电子信息", "自动化")
            if major in value
        ]
        importance = _importance(value)
        requirements.append(
            JobRequirement(
                id=f"req_{index:03d}",
                description=value,
                category=category,
                importance=importance,
                is_hard_constraint=bool(
                    importance == RequirementImportance.REQUIRED
                    and ("必须" in value or category in {RequirementCategory.EDUCATION, RequirementCategory.AVAILABILITY})
                ),
                keywords=keywords,
                normalized_keywords=keywords,
                match_mode=match_mode,
                minimum_match_count=minimum_count,
                required_proficiency=_proficiency(value),
                minimum_years=float(years_match.group(1)) if years_match else None,
                minimum_degree=_minimum_degree(value),
                accepted_majors=accepted_majors,
                evidence_expectation=f"简历中应提供能够支持“{value}”的原文证据",
                source=[line.reference(value)],
            )
        )
    return requirements


def _parse_availability(text: str) -> AvailabilityRequirement | None:
    """解析每周到岗天数、实习月数和毕业年份等时间约束。"""
    days = re.search(r"每周\s*(?:至少)?\s*(?:到岗|出勤)?\s*(\d)\s*天", text)
    months = re.search(r"(?:连续)?(?:实习)?\s*(\d+)\s*个月", text)
    years = [int(year) for year in re.findall(r"(20\d{2})\s*届", text)]
    if not (days or months or years):
        return None
    fragments = [match.group(0) for match in (days, months) if match] + [f"{year}届" for year in years]
    evidence_lines = [
        line.reference(content_text(line))
        for line in iter_located_lines(text)
        if any(fragment in line.text for fragment in fragments)
    ]
    return AvailabilityRequirement(
        minimum_days_per_week=int(days.group(1)) if days else None,
        minimum_months=int(months.group(1)) if months else None,
        graduation_years=unique(years),
        raw_text="；".join(fragments),
        source=evidence_lines,
    )


def _parse_benefits(text: str) -> list[JobBenefit]:
    """解析福利章节并为每项福利保留来源。"""
    lines = content_lines(section_lines(text, BENEFIT_SECTIONS))
    return [
        JobBenefit(name=content_text(line), source=[line.reference(content_text(line))])
        for line in lines
    ]


def parse_job_text(
    text: str,
    source_name: str | None = None,
    source_url: str | None = None,
) -> JobParseResult:
    """把岗位原文解析为职责、原子要求、匹配规则和来源证据。"""
    started = time.perf_counter()
    cleaned = clean_text(text)
    if not cleaned:
        return JobParseResult(
            success=False,
            errors=["岗位描述为空"],
            metadata=JobParseMetadata(
                parser_name="markdown_rule_job_parser",
                source_name=source_name,
                source_url=source_url,
            ),
        )
    company, title, header_source = _header_parts(cleaned)
    city, city_source = _labeled_value(cleaned, "城市", "工作城市", "地点")
    address, address_source = _labeled_value(cleaned, "地址", "工作地点", "详细地址")
    published_date, date_source = _labeled_value(cleaned, "发布日期", "发布时间")
    department, department_source = _labeled_value(cleaned, "部门", "所属部门")
    document_url, url_source = _labeled_value(cleaned, "来源链接", "原始链接")
    effective_source_url = source_url or document_url
    industry, industry_source = _labeled_value(cleaned, "行业", "所属行业")
    company_size, size_source = _labeled_value(cleaned, "公司规模", "规模")
    financing_stage, financing_source = _labeled_value(cleaned, "融资阶段", "融资情况")
    responsibilities = _parse_responsibilities(cleaned)
    requirements = _parse_requirements(cleaned)
    warnings: list[JobParseWarning] = []
    if not company:
        warnings.append(JobParseWarning(code="MISSING_COMPANY", message="未从标题识别到公司名称"))
    if not title:
        warnings.append(JobParseWarning(code="MISSING_TITLE", message="未从一级标题识别到岗位名称"))
    if not responsibilities:
        warnings.append(JobParseWarning(code="MISSING_RESPONSIBILITIES", message="未识别到岗位职责"))
    if not requirements:
        warnings.append(JobParseWarning(code="MISSING_REQUIREMENTS", message="未识别到任职要求"))

    data = JobData(
        basic_information=JobBasicInformation(
            title=title,
            department=department,
            employment_type=_employment_type(title, cleaned),
            experience_level=_experience_level(title, cleaned),
            work_mode=_work_mode(cleaned),
            cities=_split_cities(city),
            address=address,
            salary=_salary(cleaned),
            source_url=effective_source_url,
            published_date=published_date,
            source=(
                header_source
                + city_source
                + address_source
                + date_source
                + department_source
                + url_source
            ),
        ),
        company=CompanyInformation(
            name=company or None,
            industry=industry,
            company_size=company_size,
            financing_stage=financing_stage,
            source=header_source + industry_source + size_source + financing_source,
        ),
        responsibilities=responsibilities,
        requirements=requirements,
        availability=_parse_availability(cleaned),
        benefits=_parse_benefits(cleaned),
        raw_text=cleaned,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    return JobParseResult(
        success=True,
        data=data,
        warnings=warnings,
        metadata=JobParseMetadata(
            parser_name="markdown_rule_job_parser",
            parser_version="2.0",
            source_name=source_name,
            source_url=effective_source_url,
            character_count=len(cleaned),
            line_count=len(cleaned.splitlines()),
            processing_time_ms=elapsed_ms,
        ),
    )


def parse_job_document(path: str | Path) -> JobParseResult:
    """读取 Markdown/TXT 岗位文件并调用文本解析入口。"""
    document_path = Path(path)
    return parse_job_text(read_document(document_path), source_name=str(document_path))
