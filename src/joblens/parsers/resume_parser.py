"""面向中文 PDF、Markdown 和 TXT 简历的确定性解析器。"""

from __future__ import annotations

import re
import time
from pathlib import Path

from joblens.io import PdfDocumentText, read_document, read_pdf_document
from joblens.parsers.common_parser import (
    LocatedLine,
    clean_text,
    content_lines,
    content_text,
    extract_skills,
    first_h1,
    iter_located_lines,
    normalize_heading,
    section_lines,
    split_list,
    unique,
)
from joblens.schemas.resume import (
    Award,
    BasicInformation,
    Certificate,
    ContactInformation,
    DateRange,
    EducationDegree,
    EducationExperience,
    ExperienceType,
    JobPreference,
    LanguageAbility,
    LanguageLevel,
    ParseWarning,
    ProjectExperience,
    Publication,
    ResumeData,
    ResumeParseMetadata,
    ResumeParseResult,
    SkillCategory,
    SkillGroup,
    SkillItem,
    SkillLevel,
    WorkExperience,
)


TARGET_SECTIONS = {
    "求职目标", "求职意向", "期望岗位", "职业目标", "求职方向",
    "career objective", "objective", "job objective",
}
EDUCATION_SECTIONS = {
    "教育经历", "教育背景", "学术背景", "学历",
    "education", "academic background",
}
SKILL_SECTIONS = {
    "技能",
    "专业技能",
    "技能清单",
    "技术栈",
    "技能储备",
    "荣誉技能",
    "开源、竞赛与专业能力",
    "开源竞赛与专业能力",
    "skills",
    "technical skills",
    "tech stack",
}
WORK_SECTIONS = {
    "工作经历",
    "实习经历",
    "实践经历",
    "校园经历",
    "实习与项目经历",
    "experience",
    "work experience",
    "employment",
    "internship experience",
}
PROJECT_SECTIONS = {
    "项目经历", "项目经验", "个人项目", "课程项目", "实习与项目经历",
    "projects", "project experience", "personal projects",
}
PUBLICATION_SECTIONS = {
    "论文与专利", "论文", "科研成果", "学术成果", "发表成果",
    "publications", "research", "research outputs",
}
SUMMARY_SECTIONS = {"总结", "个人总结", "个人简介", "专业概述", "summary", "profile"}
LIMITATION_SECTIONS = {"当前不足", "不足", "待提升", "自我评价"}
CERTIFICATE_SECTIONS = {"证书", "专业证书", "certifications", "certificates"}
AWARD_SECTIONS = {"奖项", "荣誉奖项", "荣誉", "awards", "honors"}


SECTION_CANONICAL: dict[str, str] = {}
for aliases, canonical in (
    (TARGET_SECTIONS, "求职意向"),
    (EDUCATION_SECTIONS, "教育背景"),
    ({"工作经历"}, "工作经历"),
    ({"实习经历", "实践经历"}, "实习经历"),
    ({"校园经历"}, "校园经历"),
    ({"实习与项目经历"}, "实习与项目经历"),
    (PROJECT_SECTIONS - {"实习与项目经历"}, "项目经历"),
    (SKILL_SECTIONS, "专业技能"),
    (PUBLICATION_SECTIONS, "论文与专利"),
    (SUMMARY_SECTIONS, "总结"),
    (LIMITATION_SECTIONS, "自我评价"),
    (CERTIFICATE_SECTIONS, "证书"),
    (AWARD_SECTIONS, "荣誉奖项"),
):
    for alias in aliases:
        SECTION_CANONICAL[normalize_heading(alias)] = canonical


YEAR_MONTH_PATTERN = (
    r"(?:(?:19|20)\d{2}|20XX)"
    r"(?:\s*年\s*(?:1[0-2]|0?[1-9])\s*月|[./-]\s*(?:1[0-2]|0?[1-9]))?"
)
DATE_RANGE_RE = re.compile(
    rf"(?P<start>{YEAR_MONTH_PATTERN})\s*(?:—|–|~|至|-)\s*"
    rf"(?P<end>{YEAR_MONTH_PATTERN}|至今|现在|目前|present)",
    re.IGNORECASE,
)
METRIC_RE = re.compile(r"\d|提升|降低|减少|增长|覆盖|准确率|完成|实现|节省|优化")
ROLE_RE = re.compile(
    r"工程师|实习生|研究员|分析师|开发|负责人|成员|部长|运营|维护者|管理员|intern|engineer|researcher",
    re.IGNORECASE,
)
ORGANIZATION_RE = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9·（）() XXx]+?(?:有限公司|股份有限公司|公司|"
    r"实验室|研究院|大学|学院|学生会|委员会|中心|部门|大厂)"
)


def _canonical_section_name(value: str, *, allow_composite: bool = True) -> str | None:
    """将中英文、编号和组合章节标题统一为解析器内部名称。"""
    raw = re.sub(r"^#{1,6}\s+", "", value).strip()
    raw = re.sub(r"^(?:\d+[.)、-]?|[一二三四五六七八九十]+[、.])\s*", "", raw)
    candidates = [raw]
    if allow_composite:
        candidates.extend(re.split(r"\s*(?:/|｜|\||&|·)\s*", raw))
    for candidate in candidates:
        canonical = SECTION_CANONICAL.get(normalize_heading(candidate))
        if canonical:
            return canonical
    return None


def _strip_markdown_inline(value: str) -> str:
    """移除 Markdown 链接、强调和代码标记，仅保留可解析文本。"""
    text = re.sub(r"!\[([^]]*)\]\([^)]+\)", r"\1", value)
    text = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r"\1 \2", text)
    text = re.sub(r"(?:\*\*|__|~~|`)", "", text)
    return text.strip()


def _table_cells(line: str) -> list[str]:
    """把一行 Markdown 表格拆成清理后的单元格。"""
    return [_strip_markdown_inline(cell.strip()) for cell in line.strip().strip("|").split("|")]


def _is_table_separator(line: str) -> bool:
    """判断一行是否为 Markdown 表头下方的横线分隔行。"""
    cells = _table_cells(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)


def _table_value(headers: list[str], row: list[str], aliases: set[str]) -> str:
    """根据多个可能的列名，从表格行中读取对应单元格。"""
    normalized_aliases = {normalize_heading(alias) for alias in aliases}
    for index, header in enumerate(headers):
        if normalize_heading(header) in normalized_aliases and index < len(row):
            return row[index].strip()
    return ""


def _markdown_table_to_lines(section: str, headers: list[str], rows: list[list[str]]) -> list[str]:
    """把常见简历表格恢复成与普通 Markdown 相同的章节条目。"""
    output: list[str] = []
    for row in rows:
        if section == "教育背景":
            school = _table_value(headers, row, {"学校", "院校", "school", "university"})
            if school:
                major = _table_value(headers, row, {"专业", "major"})
                degree = _table_value(headers, row, {"学历", "学位", "degree"})
                date = _table_value(headers, row, {"时间", "日期", "date", "duration"})
                output.append("- " + " | ".join(value for value in (school, major, degree, date) if value))
                continue
            gpa = _table_value(headers, row, {"gpa", "绩点"})
            ranking = _table_value(headers, row, {"排名", "专业排名", "ranking"})
            courses = _table_value(headers, row, {"课程", "主修课程", "核心课程", "courses"})
            details = []
            if gpa:
                details.append(f"GPA：{gpa}")
            if ranking:
                details.append(f"专业排名：{ranking}")
            if courses:
                details.append(f"主修课程：{courses}")
            if details:
                output.append("- " + "；".join(details))
        elif section in {"工作经历", "实习经历", "校园经历", "实习与项目经历"}:
            organization = _table_value(headers, row, {"公司", "单位", "组织", "company", "organization"})
            position = _table_value(headers, row, {"职位", "岗位", "position", "role"})
            date = _table_value(headers, row, {"时间", "日期", "date", "duration"})
            description = _table_value(headers, row, {"工作内容", "职责", "描述", "description"})
            if organization or position:
                output.append("### " + " | ".join(value for value in (organization, position, date) if value))
                if description:
                    output.append(f"- {description}")
        elif section == "项目经历":
            name = _table_value(headers, row, {"项目", "项目名称", "project", "name"})
            role = _table_value(headers, row, {"角色", "职责", "role"})
            date = _table_value(headers, row, {"时间", "日期", "date", "duration"})
            description = _table_value(headers, row, {"项目内容", "描述", "description"})
            if name:
                output.append("### " + " | ".join(value for value in (name, role, date) if value))
                if description:
                    output.append(f"- {description}")
        elif section == "专业技能":
            values = [value for value in row if value and value not in headers]
            if values:
                output.append("- " + "：".join(values))
        else:
            output.append("- " + " | ".join(value for value in row if value))
    return output


def _prepare_markdown_resume_text(text: str) -> str:
    """规范 Markdown 标题、内联标记和表格，同时保留可追踪文本。"""
    source_lines = clean_text(text).splitlines()
    result: list[str] = []
    current_section = ""
    index = 0
    while index < len(source_lines):
        line = source_lines[index]
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if heading:
            value = _strip_markdown_inline(heading.group(2))
            if len(heading.group(1)) == 2:
                canonical = _canonical_section_name(value)
                if canonical:
                    current_section = canonical
                    result.append(f"## {canonical}")
                else:
                    current_section = value
                    result.append(f"## {value}")
            else:
                result.append(f"{heading.group(1)} {value}")
            index += 1
            continue
        if (
            "|" in line
            and index + 1 < len(source_lines)
            and _is_table_separator(source_lines[index + 1])
        ):
            headers = _table_cells(line)
            rows: list[list[str]] = []
            index += 2
            while index < len(source_lines) and "|" in source_lines[index]:
                if source_lines[index].strip() and not _is_table_separator(source_lines[index]):
                    rows.append(_table_cells(source_lines[index]))
                index += 1
            result.extend(_markdown_table_to_lines(current_section, headers, rows))
            continue
        result.append(_strip_markdown_inline(line))
        index += 1
    return clean_text("\n".join(result))


def _is_bullet(value: str) -> bool:
    """判断文本行是否以常见项目符号或数字序号开头。"""
    return bool(re.match(r"^\s*(?:[-*+•●▪◦]|\d+[.)、])\s*", value))


def _has_date_range(value: str) -> bool:
    """判断文本中是否包含可识别的起止日期。"""
    return DATE_RANGE_RE.search(value) is not None


def _looks_like_name(value: str) -> bool:
    """使用长度、字符和排除规则判断顶部文本是否可能是姓名。"""
    text = value.strip()
    if not text or len(text) > 40:
        return False
    if normalize_heading(text) in SECTION_CANONICAL:
        return False
    if re.search(r"@|电话|手机|邮箱|email|mobile|github|http|\d{4}", text, re.IGNORECASE):
        return False
    if "/" in text or "|" in text or "·" in text:
        return False
    if re.fullmatch(r"[\u4e00-\u9fff·]{2,8}", text):
        return True
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z .'-]{2,30}", text))


def _looks_like_project_heading(value: str, next_value: str = "") -> bool:
    """结合当前行和下一行判断当前行是否像项目条目标题。"""
    text = value.strip()
    if not text or _is_bullet(text):
        return False
    if re.fullmatch(r"(?:项目)?负责人|核心成员|项目成员|成员", text):
        return False
    if re.match(r"^(?:项目背景|项目简介|项目概述|核心职责|我的职责|背景|效果)[：:]", text):
        return False
    if _has_date_range(text):
        return True
    if re.search(r"(?:项目经历|Demo Project)\s*$", text, re.IGNORECASE):
        return True
    return bool(
        len(text) <= 140
        and (
            re.match(
                r"^(?:项目简介|项目概述|生成链路|检索链路|背景)[：:]",
                next_value,
            )
            or re.fullmatch(r"(?:项目)?负责人|核心成员|项目成员|成员", next_value)
            or re.match(r"^(?:https?://|github\.com/)", next_value, re.IGNORECASE)
        )
    )


def _prepare_pdf_resume_text(text: str) -> str:
    """把 PDF 视觉行恢复为解析器可追踪的轻量 Markdown 结构。"""
    cleaned = clean_text(text)
    source_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.splitlines()]
    lines = [line for line in source_lines if line]
    name_index: int | None = None
    for index, line in enumerate(lines[:8]):
        if _looks_like_name(line):
            name_index = index
            break

    result: list[str] = []
    current_section = ""
    section_entry_count = 0
    consumed_indexes: set[int] = set()
    for index, line in enumerate(lines):
        if index in consumed_indexes:
            continue
        normalized = normalize_heading(line)
        canonical = _canonical_section_name(line, allow_composite=False)
        if not canonical and normalized.endswith(normalize_heading("岗位适配能力")):
            canonical = "自我评价"
        if canonical:
            current_section = canonical
            section_entry_count = 0
            result.append(f"## {canonical}")
            continue
        if index == name_index:
            result.append(f"# {line}")
            continue

        next_value = lines[index + 1] if index + 1 < len(lines) else ""
        as_heading = False
        if current_section in WORK_SECTIONS and not _is_bullet(line):
            if _has_date_range(line):
                as_heading = True
            elif (
                next_value
                and _has_date_range(next_value)
                and ORGANIZATION_RE.search(line)
                and ROLE_RE.search(line)
            ):
                line = f"{line} {next_value}"
                consumed_indexes.add(index + 1)
                as_heading = True
        if (
            not as_heading
            and current_section in PROJECT_SECTIONS
            and _looks_like_project_heading(line, next_value)
        ):
            as_heading = True
        elif current_section == "论文与专利" and not _is_bullet(line):
            as_heading = bool(
                "虚拟链接" in line
                or (section_entry_count == 0 and next_value and _is_bullet(next_value))
            )

        if as_heading:
            result.append(f"### {line}")
            section_entry_count += 1
        elif _is_bullet(line):
            result.append("- " + re.sub(r"^\s*(?:[-*+•●▪◦]|\d+[.)、])\s*", "", line))
        else:
            result.append(line)
    return clean_text("\n".join(result))


def _level_from_line(line: str) -> SkillLevel:
    """从技能描述中的程度副词推断技能等级。"""
    lowered = line.lower()
    if any(marker in lowered for marker in ("精通", "专家")):
        return SkillLevel.EXPERT
    if any(marker in lowered for marker in ("熟练掌握", "掌握", "熟练使用")):
        return SkillLevel.PROFICIENT
    if any(marker in lowered for marker in ("熟悉", "能够使用", "能使用")):
        return SkillLevel.FAMILIAR
    if any(marker in lowered for marker in ("了解", "基础", "正在学习", "学习中")):
        return SkillLevel.BEGINNER
    return SkillLevel.UNKNOWN


def _normalize_date(date_text: str) -> str | None:
    """把中文或分隔符日期统一成 YYYY 或 YYYY-MM。"""
    compact = re.sub(r"\s+", "", date_text)
    if "XX" in compact.upper():
        return None
    match = re.match(r"(?P<year>\d{4})(?:年|[./-])?(?P<month>\d{1,2})?", compact)
    if not match:
        return None
    year = match.group("year")
    month = match.group("month")
    return f"{year}-{int(month):02d}" if month else year


def _parse_date_range(value: str) -> DateRange:
    """从文本中提取起止时间并识别“至今”等当前状态。"""
    match = DATE_RANGE_RE.search(value)
    if not match:
        return DateRange(raw_text="")
    end_raw = match.group("end")
    current = end_raw.lower() in {"至今", "现在", "目前", "present"}
    return DateRange(
        start_date=_normalize_date(match.group("start")),
        end_date=None if current else _normalize_date(end_raw),
        is_current=current,
        raw_text=match.group(0),
    )


def _top_lines(text: str) -> list[LocatedLine]:
    """返回第一个二级章节前的简历顶部信息行。"""
    result: list[LocatedLine] = []
    for line in iter_located_lines(text):
        if line.text.startswith("## "):
            break
        if content_text(line):
            result.append(line)
    return result


def _parse_basic_information(text: str) -> BasicInformation:
    """解析姓名、联系方式、所在地和毕业状态等基础信息。"""
    heading = first_h1(text)
    top = _top_lines(text)
    name = content_text(heading) if heading else None
    if name:
        name = re.sub(r"[（(](?:虚构人物|示例|测试)[^）)]*[）)]", "", name).strip()
    combined = "\n".join(content_text(line) for line in top)
    email_match = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", combined, re.IGNORECASE)
    phone_match = re.search(
        r"(?<!\d)(?:\+?86[\s-]?)?(?:1[3-9Xx]{1}[\dXx])[\dXx\s-]{7,12}(?!\d)",
        combined,
    )
    city_match = re.search(
        r"(?:地址|城市|所在地|city|location)\s*[：:]\s*([^\n|]{2,30})",
        combined,
        re.IGNORECASE,
    )
    github_match = re.search(r"(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9_.-]+", combined, re.IGNORECASE)
    links = re.findall(r"https?://[^\s|，,]+", combined)
    website = next((link for link in links if "github.com" not in link.lower()), None)
    status_match = re.search(r"(?:20\d{2}\s*届)?应届生|在校生|已工作", combined)
    graduation_match = re.search(r"(20\d{2})\s*届", combined)
    references = []
    if heading:
        references.append(heading.reference(content_text(heading)))
    for line in top:
        value = content_text(line)
        if re.search(r"@|电话|手机|邮箱|地址|城市|github|http|应届生|在校生", value, re.IGNORECASE):
            references.append(line.reference(value))
    return BasicInformation(
        name=name or None,
        contact=ContactInformation(
            phone=phone_match.group(0).strip() if phone_match else None,
            email=email_match.group(0) if email_match else None,
            city=city_match.group(1).strip() if city_match else None,
            github=github_match.group(0) if github_match else None,
            personal_website=website,
            other_links=unique(links),
        ),
        current_status=status_match.group(0) if status_match else None,
        expected_graduation_date=(f"{graduation_match.group(1)}-06" if graduation_match else None),
        source=references,
    )


def _parse_job_preference(text: str) -> JobPreference | None:
    """解析目标岗位、目标城市和到岗条件等求职意向。"""
    lines = content_lines(section_lines(text, TARGET_SECTIONS))
    inline_lines = [
        line
        for line in iter_located_lines(text)
        if re.search(r"^(?:求职意向|目标岗位|求职目标|期望岗位)\s*[：:]", content_text(line))
    ]
    lines = list({line.start_char: line for line in [*lines, *inline_lines]}.values())
    if not lines:
        top = _top_lines(text)
        candidates = [line for line in top if ROLE_RE.search(content_text(line)) and "@" not in line.text]
        if candidates:
            def role_line_score(item: LocatedLine) -> tuple[int, int]:
                """为顶部候选岗位行打分，优先选择明确职位表达。"""
                value = content_text(item)
                explicit_role = bool(
                    re.search(
                        r"(?:工程师|实习生|研究员|分析师|开发|负责人|"
                        r"engineer|developer|researcher|analyst|intern)\s*(?:·|$)",
                        value,
                        re.IGNORECASE,
                    )
                )
                banner_penalty = value.count("/") + int(value.isupper())
                return (int(explicit_role) * 3 - banner_penalty, -len(value))

            lines = [max(candidates, key=role_line_score)]
    if not lines:
        return None
    target_roles: list[str] = []
    for line in lines:
        value = content_text(line)
        value = re.sub(
            r"^(?:目标岗位|求职目标|求职意向|期望岗位|职业目标|求职方向|"
            r"career objective|job objective|objective)\s*[：:]\s*",
            "",
            value,
            flags=re.IGNORECASE,
        )
        value = re.split(r"(?:邮箱|电话|手机)\s*[：:]", value)[0]
        value = re.sub(r"\bTemplate Demo\b", "", value, flags=re.IGNORECASE).strip(" ·|")
        value = re.sub(r"\b\d{1,2}\s*years?\s*old\b", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\b20\d{2}\s*(?:届|graduate|graduation)?\b", "", value, flags=re.IGNORECASE)
        segments = [segment.strip(" ·|") for segment in re.split(r"\s*·\s*", value) if segment.strip(" ·|")]
        role_segments = [segment for segment in segments if ROLE_RE.search(segment)]
        for segment in role_segments or segments:
            target_roles.extend(split_list(segment))
    return JobPreference(
        target_roles=unique(target_roles),
        source=[line.reference(content_text(line)) for line in lines],
    )


def _degree_from_text(value: str) -> EducationDegree:
    """从教育描述中识别博士、硕士、本科等学历等级。"""
    for label, enum_value in (
        ("博士", EducationDegree.DOCTOR),
        ("硕士", EducationDegree.MASTER),
        ("本科", EducationDegree.BACHELOR),
        ("学士", EducationDegree.BACHELOR),
        ("大专", EducationDegree.ASSOCIATE),
        ("高中", EducationDegree.HIGH_SCHOOL),
    ):
        if label in value:
            return enum_value
    return EducationDegree.UNKNOWN


def _education_blocks(text: str) -> list[list[LocatedLine]]:
    """把教育章节按学校或日期边界切分为多个经历块。"""
    lines = content_lines(section_lines(text, EDUCATION_SECTIONS))
    if not lines:
        lines = [
            line
            for line in _top_lines(text)
            if re.search(r"大学|学院|学校|GPA|专业排名", content_text(line), re.IGNORECASE)
        ]
    blocks: list[list[LocatedLine]] = []
    current: list[LocatedLine] = []
    for line in lines:
        value = content_text(line)
        if not value or _is_table_separator(value) or re.fullmatch(r"[|\s-]+", value):
            continue
        is_school = bool(
            re.search(r"(?:大学|学院|学校)(?!生)", value)
            or (re.search(r"实验室|研究院", value) and re.search(r"交换|访问", value) and _has_date_range(value))
        )
        if re.match(r"^(?:竞赛|奖项|荣誉)[：:]", value):
            is_school = False
        if is_school and current:
            blocks.append(current)
            current = []
        current.append(line)
    if current:
        blocks.append(current)
    return blocks


def _parse_education(text: str) -> list[EducationExperience]:
    """将教育经历块解析为学校、专业、学历、GPA 和课程。"""
    results: list[EducationExperience] = []
    for block in _education_blocks(text):
        values = [re.sub(r"\s*\|\s*", " | ", content_text(line)).strip(" |") for line in block]
        joined = " ".join(values)
        first_cells = [cell.strip() for cell in re.split(r"\s*\|\s*", values[0])] if values else []
        school = ""
        school_cell = ""
        school_pattern = re.compile(
            r"[^\s,，|]{1,30}?(?:大学|学院|学校)(?!生)|"
            r"[^\s,，|]{1,30}?(?:研究院|实验室)"
        )
        for cell in first_cells:
            matches = list(school_pattern.finditer(cell))
            if matches:
                school_match = matches[-1]
                school = school_match.group(0).strip()
                school_cell = cell
                break
        if not school:
            matches = list(school_pattern.finditer(joined))
            school = matches[-1].group(0).strip(" ,，|-") if matches else ""
        degree = _degree_from_text(joined)
        major_match = re.search(
            r"([\u4e00-\u9fffA-Za-z0-9+·-]{2,35}?)(?:专业)?(?:学士|硕士|博士)",
            joined,
        )
        major = major_match.group(1).strip(" ,，") if major_match else None
        if major and school and school in major:
            major = None
        if school_cell:
            same_cell_major = school_cell.split(school, 1)[-1]
            same_cell_major = re.sub(r"[（(][^）)]*[）)]", "", same_cell_major)
            same_cell_major = re.sub(
                r"[（(]?\s*\|?\s*(?:交换|访问)\s*[）)]?\s*\|?",
                "",
                same_cell_major,
            )
            same_cell_major = re.sub(
                rf"{YEAR_MONTH_PATTERN}.*|本科|硕士|博士|学士|大专|GPA.*",
                "",
                same_cell_major,
                flags=re.IGNORECASE,
            ).strip(" ,，（）()|-")
            if same_cell_major and re.search(r"计算机|软件|人工智能|数据|电子|信息|网络|工程|实验班|专业", same_cell_major):
                major = same_cell_major
        if first_cells:
            school_index = next((i for i, cell in enumerate(first_cells) if school and school in cell), -1)
            if school_index >= 0:
                for cell in first_cells[school_index + 1 :]:
                    if not cell or _has_date_range(cell) or _degree_from_text(cell) != EducationDegree.UNKNOWN:
                        continue
                    if re.search(r"排名|GPA|课程|研究方向", cell, re.IGNORECASE):
                        continue
                    if re.search(r"计算机|软件|人工智能|数据|电子|信息|网络|工程|专业", cell):
                        major = cell
                        break
        if not major:
            for value in values:
                candidate = DATE_RANGE_RE.sub("", value)
                candidate = candidate.replace(school, "") if school else candidate
                candidate = re.sub(r"本科|硕士|博士|学士|大专|GPA.*", "", candidate).strip(" ,，（）()|")
                if candidate and len(candidate) <= 35 and re.search(r"计算机|软件|人工智能|数据|电子|信息|工程|专业", candidate):
                    major = candidate
                    break
        gpa_match = re.search(r"GPA\s*[：:]?\s*([0-9Xx.]+\s*/\s*[0-9Xx.]+)", joined, re.IGNORECASE)
        ranking_match = re.search(
            r"(?:专业)?排名\s*[：:]?\s*(.+?)(?=\s*[|；;,，]|核心课程|主修课程|相关课程|研究方向|荣誉|竞赛|学生工作|$)",
            joined,
            re.IGNORECASE,
        )
        courses: list[str] = []
        for value in values:
            if re.search(r"(?:主修|核心|相关)课程", value):
                course_value = re.sub(r"^.*?课程\s*[：:]\s*", "", value)
                course_value = re.split(r"\s*[|；;]\s*", course_value)[0]
                courses.extend(split_list(course_value))
        results.append(
            EducationExperience(
                school=school,
                major=major,
                degree=degree,
                date_range=_parse_date_range(joined),
                gpa=gpa_match.group(1).replace(" ", "") if gpa_match else None,
                ranking=ranking_match.group(1).strip() if ranking_match else None,
                relevant_courses=unique(courses),
                descriptions=values[1:] if len(values) > 1 else ([] if school else values),
                source=[line.reference(content_text(line)) for line in block],
            )
        )
    filtered = [item for item in results if item.school or item.major or item.descriptions]
    for index, item in enumerate(filtered[1:], start=1):
        item_evidence = " ".join(source.text for source in item.source)
        if re.search(r"交换|访问", item_evidence) and index > 0:
            previous = filtered[index - 1]
            if item.gpa and not previous.gpa:
                previous.gpa, item.gpa = item.gpa, None
            if item.ranking and not previous.ranking:
                previous.ranking, item.ranking = item.ranking, None
    return filtered


def _parse_skills(text: str) -> tuple[list[SkillGroup], list[SkillItem]]:
    """解析技能分组，并生成去重后的标准技能列表。"""
    explicit_lines = content_lines(section_lines(text, SKILL_SECTIONS))
    all_lines = [line for line in content_lines(iter_located_lines(text)) if not line.text.startswith("### ")]
    skill_by_name: dict[str, SkillItem] = {}
    for line in all_lines:
        value = content_text(line)
        level = _level_from_line(value)
        for definition in extract_skills(value):
            reference = line.reference(value)
            existing = skill_by_name.get(definition.canonical)
            if existing is None:
                skill_by_name[definition.canonical] = SkillItem(
                    name=definition.canonical,
                    normalized_name=definition.canonical,
                    category=SkillCategory(definition.category),
                    level=level,
                    source=[reference],
                )
            else:
                if all(source.start_char != reference.start_char for source in existing.source):
                    existing.source.append(reference)
                if existing.level == SkillLevel.UNKNOWN and level != SkillLevel.UNKNOWN:
                    existing.level = level
    skills = list(skill_by_name.values())
    group = SkillGroup(
        group_name="专业技能",
        skills=skills,
        source=[line.reference(content_text(line)) for line in explicit_lines],
    )
    return ([group] if explicit_lines else []), skills


def _section_blocks(text: str, aliases: set[str]) -> list[tuple[LocatedLine, list[LocatedLine]]]:
    """按三级标题把指定章节切分为“标题行 + 内容行”条目。"""
    lines = section_lines(text, aliases)
    blocks: list[tuple[LocatedLine, list[LocatedLine]]] = []
    heading: LocatedLine | None = None
    body: list[LocatedLine] = []
    for line in lines:
        if line.text.startswith("### "):
            if heading is not None:
                blocks.append((heading, body))
            heading = line
            body = []
        elif heading is not None and content_text(line):
            body.append(line)
    if heading is not None:
        blocks.append((heading, body))
    return blocks


def _is_project_entry(value: str) -> bool:
    """判断混合经历条目的标题是否更像项目而不是工作单位。"""
    return bool(re.search(r"项目|系统|平台|工作台|Agent|MiniCode|MoonInk|ScenePilot|CanvasForge|OmniSearch", value, re.IGNORECASE))


def _experience_type(section: str, position: str, organization: str = "") -> ExperienceType:
    """结合章节、职位和组织名判断工作、实习或校园经历类型。"""
    position_value = position.lower()
    section_value = section.lower()
    value = f"{section_value} {position_value} {organization.lower()}"
    if "校园" in value or "学生会" in value or "校团委" in value:
        return ExperienceType.CAMPUS
    if "实习" in position_value or "intern" in position_value:
        return ExperienceType.INTERNSHIP
    if section_value in {"实习经历", "实践经历", "internship experience"}:
        return ExperienceType.INTERNSHIP
    if "兼职" in value or "part-time" in value:
        return ExperienceType.PART_TIME
    if "志愿" in value:
        return ExperienceType.VOLUNTEER
    if "工作" in section_value or ROLE_RE.search(position):
        return ExperienceType.FULL_TIME
    return ExperienceType.UNKNOWN


def _parse_work_heading(value: str) -> tuple[str, str, DateRange]:
    """从工作条目标题中拆分组织、职位和日期范围。"""
    date_range = _parse_date_range(value)
    without_date = value.replace(date_range.raw_text, "").strip(" |｜,，·")
    without_date = re.sub(
        r"[（(][^）)]*(?:示例|Mock|模拟)[^）)]*[）)]",
        "",
        without_date,
        flags=re.IGNORECASE,
    ).strip()
    parts = [part.strip() for part in re.split(r"[|｜]", without_date) if part.strip()]
    organization = ""
    position = ""
    role_part = next((part for part in parts if ROLE_RE.search(part)), "")
    non_role_parts = [part for part in parts if part != role_part]
    if role_part and non_role_parts:
        position = role_part
        organization = non_role_parts[0]
    else:
        organization_match = ORGANIZATION_RE.search(without_date)
        organization = organization_match.group(0).strip() if organization_match else ""
        if organization:
            before = without_date[: organization_match.start()].strip(" |｜,，·")
            after = without_date[organization_match.end() :].strip(" |｜,，·（）()")
            position = after if ROLE_RE.search(after) else before
        elif len(parts) >= 2:
            organization, position = parts[0], parts[1]
        elif parts:
            organization = parts[0]
    # “公司 · 部门”只把左侧实体作为单位，部门不污染公司名。
    if "·" in organization:
        organization = organization.split("·", 1)[0].strip()
    organization = re.sub(r"[（(](?:仅为示例演示|Mock|模拟经历)[^）)]*[）)]", "", organization, flags=re.IGNORECASE).strip()
    position = re.sub(r"[（(](?:仅为示例演示|Mock|模拟经历)[^）)]*[）)]", "", position, flags=re.IGNORECASE).strip()
    return organization, position, date_range


def _parse_work_experiences(text: str) -> list[WorkExperience]:
    """解析工作、实习和校园经历，并提取职责、成果与技术。"""
    results: list[WorkExperience] = []
    for heading, body in _section_blocks(text, WORK_SECTIONS):
        heading_value = content_text(heading)
        if heading.section == "实习与项目经历" and _is_project_entry(heading_value):
            continue
        organization, position, date_range = _parse_work_heading(heading_value)
        body_values = [content_text(line) for line in body]
        if not position and body_values and ROLE_RE.search(body_values[0]) and not _is_bullet(body[0].text):
            position = body_values.pop(0)
            body = body[1:]
        joined = " ".join([heading_value, *body_values])
        technologies = unique([definition.canonical for definition in extract_skills(joined)])
        achievements = [value for value in body_values if METRIC_RE.search(value)]
        results.append(
            WorkExperience(
                organization=organization,
                position=position,
                experience_type=_experience_type(heading.section, position, organization),
                date_range=date_range,
                responsibilities=body_values,
                achievements=achievements,
                technologies=technologies,
                source=[heading.reference(heading_value)]
                + [line.reference(content_text(line)) for line in body],
            )
        )
    return [item for item in results if item.organization or item.position]


def _parse_projects(text: str) -> list[ProjectExperience]:
    """解析项目名称、角色、描述、技术栈、成果和链接。"""
    projects: list[ProjectExperience] = []
    for heading, lines in _section_blocks(text, PROJECT_SECTIONS):
        heading_value = content_text(heading)
        if heading.section == "实习与项目经历" and not _is_project_entry(heading_value):
            continue
        date_range = _parse_date_range(heading_value)
        title = heading_value.replace(date_range.raw_text, "").strip(" |｜,，·")
        parts = [part.strip() for part in re.split(r"[|｜]", title) if part.strip()]
        name = re.sub(r"\s*项目经历\s*$", "", parts[0] if parts else title).strip()
        role = next((part for part in parts[1:] if ROLE_RE.search(part)), None)
        descriptions = [content_text(line) for line in lines]
        if (
            not role
            and descriptions
            and len(descriptions[0]) <= 30
            and ROLE_RE.search(descriptions[0])
            and not _is_bullet(lines[0].text)
        ):
            role = descriptions.pop(0)
            lines = lines[1:]
        joined = " ".join([heading_value, *descriptions])
        technologies = unique([definition.canonical for definition in extract_skills(joined)])
        achievements = [value for value in descriptions if METRIC_RE.search(value)]
        urls = re.findall(r"https?://[^\s，,]+|github\.com/[A-Za-z0-9_.\-/]+", joined, re.IGNORECASE)
        repository_url = next((url for url in urls if "github.com" in url.lower()), None)
        projects.append(
            ProjectExperience(
                name=name,
                role=role,
                date_range=date_range,
                project_type=("课程项目" if "课程" in name else "项目"),
                description="；".join(descriptions) or None,
                responsibilities=descriptions,
                technologies=technologies,
                achievements=achievements,
                repository_url=repository_url,
                source=[heading.reference(heading_value)]
                + [line.reference(content_text(line)) for line in lines],
            )
        )
    return [project for project in projects if project.name]


def _parse_certificates(text: str) -> list[Certificate]:
    """解析证书章节中的证书名称及来源。"""
    results: list[Certificate] = []
    section_items = content_lines(section_lines(text, CERTIFICATE_SECTIONS))
    for line in iter_located_lines(text):
        value = content_text(line)
        match = re.search(
            r"(?:专业)?证书\s*[：:]\s*(.+)|certifications?\s*[：:]\s*(.+)",
            value,
            re.IGNORECASE,
        )
        if match:
            certificate_text = match.group(1) or match.group(2) or ""
        elif any(item.start_char == line.start_char for item in section_items):
            certificate_text = value
        else:
            continue
        for name in split_list(certificate_text):
            results.append(Certificate(name=name, source=[line.reference(value)]))
    seen: set[str] = set()
    unique_results: list[Certificate] = []
    for item in results:
        key = item.name.casefold()
        if key not in seen:
            seen.add(key)
            unique_results.append(item)
    return unique_results


def _parse_languages(text: str) -> list[LanguageAbility]:
    """解析语言、考试名称、等级和成绩。"""
    results: list[LanguageAbility] = []
    seen: set[str] = set()
    patterns = (
        ("英语", r"CET[- ]?4(?:\s*[（(]?\d+[）)]?)?", LanguageLevel.INTERMEDIATE),
        ("英语", r"CET[- ]?6(?:\s*[（(]?\d+[）)]?)?", LanguageLevel.ADVANCED),
        ("英语", r"IELTS(?:\s*[（(]?\d+(?:\.\d+)?[）)]?)?", LanguageLevel.ADVANCED),
        ("英语", r"TOEFL(?:\s*[（(]?\d+[）)]?)?", LanguageLevel.ADVANCED),
        ("普通话", r"普通话[^，,；;\n]{0,10}", LanguageLevel.NATIVE),
    )
    for line in iter_located_lines(text):
        value = content_text(line)
        for language, pattern, level in patterns:
            for match in re.finditer(pattern, value, re.IGNORECASE):
                examination = match.group(0).strip("。.;；")
                key = examination.casefold()
                if key in seen:
                    continue
                seen.add(key)
                score_match = re.search(r"[（(](\d+(?:\.\d+)?)[）)]", examination)
                results.append(
                    LanguageAbility(
                        language=language,
                        level=level,
                        examination=examination,
                        score=score_match.group(1) if score_match else None,
                        source=[line.reference(value)],
                    )
                )
    return results


def _parse_awards(text: str) -> list[Award]:
    """解析奖项或荣誉名称、等级和说明。"""
    results: list[Award] = []
    section_items = content_lines(section_lines(text, AWARD_SECTIONS))
    for line in iter_located_lines(text):
        value = content_text(line)
        in_award_section = any(item.start_char == line.start_char for item in section_items)
        if len(value) > 240 or (not in_award_section and not re.search(r"奖|荣誉", value)):
            continue
        if value.startswith("##"):
            continue
        level_match = re.search(r"国家级|省级|校级|全国", value)
        results.append(
            Award(
                name=value,
                level=level_match.group(0) if level_match else None,
                source=[line.reference(value)],
            )
        )
    return results


def _parse_publications(text: str) -> list[Publication]:
    """解析论文、专利或其他科研成果。"""
    publications: list[Publication] = []
    for heading, lines in _section_blocks(text, PUBLICATION_SECTIONS):
        title = content_text(heading)
        publication_type = "patent" if "专利" in title else "paper"
        author_text = next((content_text(line) for line in lines if "作者" in content_text(line)), "")
        if author_text:
            author_text = re.sub(r"^(?:作者|authors?)\s*[：:]\s*", "", author_text, flags=re.IGNORECASE)
        publications.append(
            Publication(
                title=re.sub(r"\s*虚拟链接\s*$", "", title).strip(),
                publication_type=publication_type,
                authors=[author_text] if author_text else [],
                source=[heading.reference(title)]
                + [line.reference(content_text(line)) for line in lines],
            )
        )
    return publications


def _parse_professional_summary(text: str) -> str | None:
    """合并个人总结章节内容；没有内容时返回 None。"""
    explicit = [content_text(line) for line in content_lines(section_lines(text, SUMMARY_SECTIONS))]
    if explicit:
        return " ".join(explicit)
    candidates = []
    for line in _top_lines(text):
        value = content_text(line)
        if len(value) >= 30 and not re.search(r"@|电话|手机|邮箱|github|http", value, re.IGNORECASE):
            candidates.append(value)
    return " ".join(candidates) or None


def _parse_other_information(text: str) -> list[str]:
    """收集当前不足、自我评价等未进入主要结构的补充信息。"""
    return [content_text(line) for line in content_lines(section_lines(text, LIMITATION_SECTIONS))]


def _quality_warnings(data: ResumeData) -> list[ParseWarning]:
    """对“有值但明显不可信”的字段给出非致命告警。"""
    warnings: list[ParseWarning] = []
    placeholder_re = re.compile(r"^(?:姓名|无名氏|xxx+|xxxx\s*岗位|示例|sample)$", re.IGNORECASE)
    name = (data.basic_information.name or "").strip()
    if name and placeholder_re.search(name):
        warnings.append(
            ParseWarning(code="PLACEHOLDER_NAME", message=f"姓名疑似占位内容：{name}", section="基本信息")
        )
    contact = data.basic_information.contact
    if contact.phone and re.search(r"x{2,}", contact.phone, re.IGNORECASE):
        warnings.append(
            ParseWarning(code="PLACEHOLDER_PHONE", message="手机号包含占位字符 X", section="基本信息")
        )
    if data.job_preference and any(re.search(r"x{2,}|示例", role, re.IGNORECASE) for role in data.job_preference.target_roles):
        warnings.append(
            ParseWarning(code="PLACEHOLDER_TARGET_ROLE", message="求职岗位疑似模板占位内容", section="求职意向")
        )
    for item in data.education:
        if item.major and ("|" in item.major or re.search(r"排名|课程|GPA", item.major, re.IGNORECASE)):
            warnings.append(
                ParseWarning(code="LOW_CONFIDENCE_MAJOR", message=f"专业字段可能包含相邻字段：{item.major}", section="教育背景")
            )
        if item.ranking and re.search(r"课程|GPA|研究方向", item.ranking, re.IGNORECASE):
            warnings.append(
                ParseWarning(code="LOW_CONFIDENCE_RANKING", message=f"排名字段可能包含相邻字段：{item.ranking}", section="教育背景")
            )
        if "XX" in item.date_range.raw_text.upper():
            warnings.append(
                ParseWarning(code="PLACEHOLDER_EDUCATION_DATE", message="教育日期包含 20XX 占位符", section="教育背景")
            )
    for item in data.work_experiences:
        if re.fullmatch(r"mock|示例|sample", item.organization.strip(), re.IGNORECASE):
            warnings.append(
                ParseWarning(code="LOW_CONFIDENCE_ORGANIZATION", message=f"工作单位疑似模板标记：{item.organization}", section="工作经历")
            )
        if item.organization and not item.position:
            warnings.append(
                ParseWarning(code="MISSING_POSITION", message=f"工作单位“{item.organization}”未识别到职位", section="工作经历")
            )
    # 去重，避免同一类型在重复条目上刷屏。
    unique_warnings: list[ParseWarning] = []
    seen: set[tuple[str, str]] = set()
    for warning in warnings:
        key = (warning.code, warning.message)
        if key not in seen:
            seen.add(key)
            unique_warnings.append(warning)
    return unique_warnings


def parse_resume_text(
    text: str,
    source_name: str | None = None,
    pdf_document: PdfDocumentText | None = None,
) -> ResumeParseResult:
    """将简历原文解析为富结构数据；无法提取的内容通过 warnings 暴露。"""
    started = time.perf_counter()
    suffix = Path(source_name).suffix.lower() if source_name else ""
    prepared = _prepare_pdf_resume_text(text) if suffix == ".pdf" else _prepare_markdown_resume_text(text)
    warnings: list[ParseWarning] = []
    parser_name = "pdf_rule_resume_parser" if suffix == ".pdf" else "markdown_rule_resume_parser"
    extraction_method = "pdfplumber_text_layer" if suffix == ".pdf" else "utf8_text"
    if not prepared:
        return ResumeParseResult(
            success=False,
            errors=["简历内容为空"],
            metadata=ResumeParseMetadata(
                parser_name=parser_name,
                source_file_name=source_name,
                source_file_type=suffix.lstrip(".") or None,
                extraction_method=extraction_method,
            ),
        )

    preference = _parse_job_preference(prepared)
    education = _parse_education(prepared)
    skill_groups, skills = _parse_skills(prepared)
    work_experiences = _parse_work_experiences(prepared)
    projects = _parse_projects(prepared)
    data = ResumeData(
        basic_information=_parse_basic_information(prepared),
        job_preference=preference,
        professional_summary=_parse_professional_summary(prepared),
        education=education,
        skill_groups=skill_groups,
        skills=skills,
        work_experiences=work_experiences,
        projects=projects,
        certificates=_parse_certificates(prepared),
        awards=_parse_awards(prepared),
        languages=_parse_languages(prepared),
        publications=_parse_publications(prepared),
        other_information=_parse_other_information(prepared),
        raw_text=prepared,
    )
    if not education:
        warnings.append(ParseWarning(code="MISSING_EDUCATION", message="未识别到教育经历", section="教育经历"))
    if not skills:
        warnings.append(ParseWarning(code="MISSING_SKILLS", message="未识别到标准技能词", section="技能"))
    if not work_experiences:
        warnings.append(ParseWarning(code="MISSING_WORK_EXPERIENCE", message="未识别到工作或实习经历", section="工作经历"))
    research_resume = bool(data.publications) or any(
        re.search(r"研究员|research", role, re.IGNORECASE)
        for role in (data.job_preference.target_roles if data.job_preference else [])
    )
    if not projects and not research_resume:
        warnings.append(ParseWarning(code="MISSING_PROJECTS", message="未识别到项目经历", section="项目经历"))
    if pdf_document and pdf_document.empty_page_numbers:
        warnings.append(
            ParseWarning(
                code="EMPTY_PDF_PAGES",
                message="以下页面未提取到文本：" + "、".join(map(str, pdf_document.empty_page_numbers)),
            )
        )
    warnings.extend(_quality_warnings(data))
    elapsed_ms = (time.perf_counter() - started) * 1000
    return ResumeParseResult(
        success=True,
        data=data,
        warnings=warnings,
        metadata=ResumeParseMetadata(
            parser_name=parser_name,
            parser_version="3.0",
            source_file_name=source_name,
            source_file_type=suffix.lstrip(".") or None,
            character_count=len(prepared),
            line_count=len(prepared.splitlines()),
            page_count=pdf_document.page_count if pdf_document else None,
            extraction_method=(pdf_document.extraction_method if pdf_document else extraction_method),
            empty_page_numbers=(list(pdf_document.empty_page_numbers) if pdf_document else []),
            ocr_page_numbers=(list(pdf_document.ocr_page_numbers) if pdf_document else []),
            layout_page_numbers=(list(pdf_document.layout_page_numbers) if pdf_document else []),
            processing_time_ms=elapsed_ms,
        ),
    )


def parse_resume_document(path: str | Path) -> ResumeParseResult:
    """按文件类型读取 PDF 或文本简历，并进入统一文本解析流程。"""
    document_path = Path(path)
    if document_path.suffix.lower() == ".pdf":
        pdf_document = read_pdf_document(document_path)
        return parse_resume_text(
            pdf_document.text,
            source_name=str(document_path),
            pdf_document=pdf_document,
        )
    return parse_resume_text(read_document(document_path), source_name=str(document_path))
