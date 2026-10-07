"""可追溯的本地简历和岗位解析器。"""

from .job_parser import parse_job_document, parse_job_text
from .resume_parser import parse_resume_document, parse_resume_text
from .zhipin_job_parser import (
    build_zhipin_search_url,
    parse_zhipin_export,
    parse_zhipin_html,
)

__all__ = [
    "parse_job_document",
    "parse_job_text",
    "parse_resume_document",
    "parse_resume_text",
    "build_zhipin_search_url",
    "parse_zhipin_export",
    "parse_zhipin_html",
]
