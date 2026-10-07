"""BOSS直聘手动导出内容适配器。

本模块刻意不包含网络请求、自动登录、验证码处理或批量爬取逻辑。
它只负责：生成官方搜索链接，以及解析用户主动保存的单页 HTML/文本。
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

from joblens.exceptions import DocumentReadError, DocumentParseError
from joblens.parsers.common_parser import clean_text, unique
from joblens.parsers.job_parser import parse_job_text
from joblens.schemas.job import JobParseResult


ZHIPIN_HOSTS = {"zhipin.com", "www.zhipin.com"}


def _validate_zhipin_url(url: str | None) -> str | None:
    """只接受 HTTPS 的 BOSS直聘域名，拒绝伪造来源地址。"""
    if not url:
        return None
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not (host in ZHIPIN_HOSTS or host.endswith(".zhipin.com")):
        raise ValueError("source_url 必须是 https://*.zhipin.com/ 地址")
    return url


def build_zhipin_search_url(query: str, city_code: str = "100010000") -> str:
    """生成官方搜索链接；调用者应在普通浏览器中自行打开。"""
    keyword = query.strip()
    if not keyword:
        raise ValueError("搜索关键词不能为空")
    if not re.fullmatch(r"\d{9}", city_code):
        raise ValueError("city_code 应为 9 位 BOSS直聘城市代码")
    return f"https://www.zhipin.com/c{city_code}/?{urlencode({'query': keyword})}"


@dataclass
class _CapturedHTML:
    """HTMLParser 收集到的文本、标签、类名和 JSON-LD 中间结果。"""

    text_parts: list[str] = field(default_factory=list)
    meta: dict[str, str] = field(default_factory=dict)
    tag_text: dict[str, list[str]] = field(default_factory=dict)
    class_text: dict[str, list[str]] = field(default_factory=dict)
    json_ld: list[dict[str, Any]] = field(default_factory=list)


class _ExportHTMLParser(HTMLParser):
    """只解析用户保存的 HTML，不执行脚本也不发起网络请求。"""

    BLOCK_TAGS = {"article", "br", "div", "h1", "h2", "h3", "h4", "li", "p", "section"}
    CAPTURE_TAGS = {"h1", "h2", "h3", "h4", "li", "p", "span", "title"}

    def __init__(self) -> None:
        """初始化解析结果以及标签文本、JSON-LD 的临时缓冲区。"""
        super().__init__(convert_charrefs=True)
        self.result = _CapturedHTML()
        self._captures: list[tuple[str, set[str], list[str]]] = []
        self._json_buffer: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """处理开始标签，记录 meta、class 和 JSON-LD 脚本。"""
        attributes = {key.lower(): value or "" for key, value in attrs}
        classes = {item for item in attributes.get("class", "").split() if item}
        if tag == "meta":
            key = attributes.get("property") or attributes.get("name")
            if key and attributes.get("content"):
                self.result.meta[key.lower()] = attributes["content"].strip()
        if tag == "script" and attributes.get("type", "").lower() == "application/ld+json":
            self._json_buffer = []
        if tag in self.BLOCK_TAGS:
            self.result.text_parts.append("\n")
        if tag in self.CAPTURE_TAGS or classes:
            self._captures.append((tag, classes, []))

    def handle_endtag(self, tag: str) -> None:
        """处理结束标签，将临时缓冲内容写入结构化采集结果。"""
        if tag == "script" and self._json_buffer is not None:
            payload = "".join(self._json_buffer).strip()
            self._json_buffer = None
            if payload:
                try:
                    decoded = json.loads(payload)
                    items = decoded if isinstance(decoded, list) else [decoded]
                    self.result.json_ld.extend(item for item in items if isinstance(item, dict))
                except json.JSONDecodeError:
                    pass
        for index in range(len(self._captures) - 1, -1, -1):
            capture_tag, classes, buffer = self._captures[index]
            if capture_tag != tag:
                continue
            self._captures.pop(index)
            value = clean_text("".join(buffer))
            if value:
                self.result.tag_text.setdefault(tag, []).append(value)
                for class_name in classes:
                    self.result.class_text.setdefault(class_name, []).append(value)
            break
        if tag in self.BLOCK_TAGS:
            self.result.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        """收集普通文本或当前 JSON-LD 脚本内容。"""
        if self._json_buffer is not None:
            self._json_buffer.append(data)
            return
        self.result.text_parts.append(data)
        for _, _, buffer in self._captures:
            buffer.append(data)


def _first_class_text(captured: _CapturedHTML, patterns: tuple[str, ...]) -> str | None:
    """按 class 名片段寻找第一个非空文本。"""
    for class_name, values in captured.class_text.items():
        lowered = class_name.lower()
        if any(pattern in lowered for pattern in patterns):
            value = next((item.strip() for item in values if item.strip()), "")
            if value:
                return value
    return None


def _job_posting_json(captured: _CapturedHTML) -> dict[str, Any] | None:
    """从 JSON-LD 顶层或 @graph 中寻找 JobPosting 对象。"""
    for item in captured.json_ld:
        item_type = item.get("@type")
        if item_type == "JobPosting" or (isinstance(item_type, list) and "JobPosting" in item_type):
            return item
        graph = item.get("@graph")
        if isinstance(graph, list):
            for child in graph:
                if isinstance(child, dict) and child.get("@type") == "JobPosting":
                    return child
    return None


def _address_from_json(value: Any) -> tuple[str | None, str | None]:
    """从 JobPosting.jobLocation 中提取城市和详细地址。"""
    locations = value if isinstance(value, list) else [value]
    for location in locations:
        if not isinstance(location, dict):
            continue
        address = location.get("address", location)
        if not isinstance(address, dict):
            continue
        city = address.get("addressLocality") or address.get("addressRegion")
        detail = "".join(
            str(address.get(key) or "")
            for key in ("addressRegion", "addressLocality", "streetAddress")
        ).strip()
        return (str(city).strip() if city else None, detail or None)
    return None, None


def _salary_from_json(value: Any) -> str | None:
    """将 JobPosting.baseSalary 转成岗位解析器可识别的薪资文本。"""
    if not isinstance(value, dict):
        return None
    currency = value.get("currency", "CNY")
    detail = value.get("value")
    if isinstance(detail, dict):
        minimum = detail.get("minValue")
        maximum = detail.get("maxValue")
        unit = str(detail.get("unitText") or "").upper()
        period = {"DAY": "/天", "MONTH": "/月", "YEAR": "/年", "HOUR": "/小时"}.get(unit, "")
        if minimum is not None and maximum is not None:
            prefix = "¥" if str(currency).upper() == "CNY" else f"{currency} "
            return f"{prefix}{minimum}-{maximum}{period}"
    return None


def _plain_html_fragment(value: str) -> str:
    """把 JSON-LD 中可能包含标签的 description 转成纯文本。"""
    parser = _ExportHTMLParser()
    parser.feed(value)
    return clean_text(html.unescape("".join(parser.result.text_parts)))


def _split_items(value: str) -> list[str]:
    """按换行和中英文序号切分职责或要求条目。"""
    normalized = clean_text(value)
    normalized = re.sub(r"(?<!\d)(?=(?:\d{1,2}[.、)]|[一二三四五六七八九十]+[、.]))", "\n", normalized)
    return [
        re.sub(r"^(?:[-*•·]|\d{1,2}[.、)]|[一二三四五六七八九十]+[、.])\s*", "", line).strip()
        for line in normalized.splitlines()
        if line.strip()
    ]


def _description_sections(description: str) -> dict[str, list[str]]:
    """把连续职位描述归类为职责、要求和福利三个章节。"""
    sections: dict[str, list[str]] = {"岗位职责": [], "任职要求": [], "职位福利": []}
    current: str | None = None
    heading_patterns = (
        ("岗位职责", r"^(?:岗位|工作|职位)?职责|^工作内容"),
        ("任职要求", r"^(?:任职|岗位|职位)?要求|^任职资格|^资格要求"),
        ("职位福利", r"^(?:职位|岗位)?福利|^我们提供|^福利待遇|^岗位亮点"),
    )
    for item in _split_items(description):
        heading = next((name for name, pattern in heading_patterns if re.search(pattern, item, re.IGNORECASE)), None)
        if heading and len(item) <= 25:
            current = heading
            continue
        target = current
        if target is None:
            if re.search(r"要求|本科|硕士|学历|经验|熟悉|掌握|具备|优先", item):
                target = "任职要求"
            else:
                target = "岗位职责"
        sections[target].append(item)
    return sections


def _markdown_from_export(captured: _CapturedHTML, source_url: str | None) -> str:
    """把 HTML 采集结果整理成本项目岗位解析器接受的 Markdown。"""
    posting = _job_posting_json(captured) or {}
    organization = posting.get("hiringOrganization")
    company = organization.get("name") if isinstance(organization, dict) else None
    company = company or _first_class_text(captured, ("company-name", "sider-company-name"))
    title = posting.get("title") or _first_class_text(captured, ("job-title", "name"))
    if not title:
        title = next(iter(captured.tag_text.get("h1", [])), None)
    if not title:
        meta_title = captured.meta.get("og:title") or captured.meta.get("twitter:title") or captured.meta.get("title")
        title = re.split(r"[-_|]BOSS直聘", meta_title or "")[0].strip() or None

    description_value = posting.get("description")
    description = _plain_html_fragment(str(description_value)) if description_value else None
    description = description or _first_class_text(captured, ("job-sec-text", "job-description", "detail-content"))
    if not description:
        description = clean_text("".join(captured.text_parts))

    city, address = _address_from_json(posting.get("jobLocation"))
    location_text = _first_class_text(captured, ("job-location", "location-address", "location"))
    city = city or (re.split(r"[区县·\s]", location_text)[0] if location_text else None)
    address = address or location_text
    salary = _salary_from_json(posting.get("baseSalary"))
    salary = salary or _first_class_text(captured, ("salary",))
    published_date = posting.get("datePosted")
    industry = posting.get("industry")

    company_text = _first_class_text(captured, ("company-info", "company-tag-list")) or ""
    size_match = re.search(r"(?:0-20|20-99|100-499|500-999|1000-9999|10000人以上)人?", company_text)
    financing_match = re.search(r"未融资|不需要融资|天使轮|[A-F]轮|已上市", company_text)

    if company:
        lines = [f"# {company} — {title or '岗位未识别'}"]
    else:
        lines = [f"# {title or '岗位未识别'}"]
    for label, value in (
        ("薪资", salary),
        ("城市", city),
        ("地址", address),
        ("行业", industry),
        ("公司规模", size_match.group(0) if size_match else None),
        ("融资阶段", financing_match.group(0) if financing_match else None),
        ("发布日期", published_date),
        ("来源链接", source_url),
    ):
        if value:
            lines.append(f"{label}：{value}")

    sections = _description_sections(description)
    for heading, items in sections.items():
        if not items:
            continue
        lines.extend(["", f"## {heading}", ""])
        lines.extend(f"- {item}" for item in unique(items))
    return clean_text("\n".join(lines))


def parse_zhipin_html(
    html_text: str,
    *,
    source_url: str | None = None,
    source_name: str | None = None,
) -> JobParseResult:
    """解析用户手动保存的单个 BOSS直聘 HTML 页面。"""
    validated_url = _validate_zhipin_url(source_url)
    parser = _ExportHTMLParser()
    parser.feed(html_text)
    markdown = _markdown_from_export(parser.result, validated_url)
    result = parse_job_text(markdown, source_name=source_name, source_url=validated_url)
    if (
        result.data is None
        or not result.data.basic_information.title
        or result.data.basic_information.title == "岗位未识别"
    ):
        raise DocumentParseError("导出的页面中未识别到岗位标题；请确认保存的是岗位详情页")
    result.metadata.parser_name = "zhipin_export_job_parser"
    result.metadata.parser_version = "1.0"
    return result


def parse_zhipin_export(path: str | Path, *, source_url: str | None = None) -> JobParseResult:
    """读取手动保存的 HTML、Markdown 或 TXT 岗位页面。"""
    export_path = Path(path).expanduser()
    if not export_path.exists() or not export_path.is_file():
        raise DocumentReadError(f"导出文件不存在：{export_path}")
    if export_path.suffix.lower() not in {".html", ".htm", ".md", ".markdown", ".txt"}:
        raise DocumentReadError("BOSS直聘导入仅支持 HTML、Markdown 和 TXT")
    try:
        content = export_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentReadError(f"导出文件不是有效 UTF-8：{export_path}") from exc
    if not content.strip():
        raise DocumentReadError(f"导出文件为空：{export_path}")
    if export_path.suffix.lower() in {".html", ".htm"}:
        return parse_zhipin_html(content, source_url=source_url, source_name=str(export_path))
    return parse_job_text(content, source_name=str(export_path), source_url=_validate_zhipin_url(source_url))
