"""阶段 1 的固定分析流程。"""

from __future__ import annotations

import re
from pathlib import Path

from joblens.extractors import (
    extract_job_local,
    extract_job_openai,
    extract_resume_local,
    extract_resume_openai,
)
from joblens.io import read_document
from joblens.matcher import match_resume_to_job
from joblens.models import AnalysisReport, JobDescription, ResumeProfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_resume(path: str | Path, provider: str = "local") -> ResumeProfile:
    """读取简历文件，并根据 provider 选择本地或外部模型解析器。"""
    document_path = Path(path).expanduser().resolve()
    text = read_document(document_path)
    source = str(document_path)
    if provider == "local":
        return extract_resume_local(text, source)
    if provider == "openai":
        return extract_resume_openai(text, source)
    raise ValueError(f"不支持的解析提供商：{provider}")


def parse_job_description(path: str | Path, provider: str = "local") -> JobDescription:
    """读取岗位文件，并根据 provider 选择本地或外部模型解析器。"""
    document_path = Path(path).expanduser().resolve()
    text = read_document(document_path)
    source = str(document_path)
    if provider == "local":
        return extract_job_local(text, source)
    if provider == "openai":
        return extract_job_openai(text, source)
    raise ValueError(f"不支持的解析提供商：{provider}")


def match_requirements(resume: ResumeProfile, job: JobDescription) -> AnalysisReport:
    """调用确定性匹配器，将轻量简历与岗位模型生成分析报告。"""
    return match_resume_to_job(resume, job)


def _default_report_path(job: JobDescription) -> Path:
    """根据公司和岗位名生成文件系统安全的默认报告路径。"""
    stem = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", f"{job.company}_{job.title}").strip("_")
    return PROJECT_ROOT / "reports" / f"{stem or 'analysis'}_report.json"


def generate_report(
    resume_path: str | Path,
    job_path: str | Path,
    provider: str = "local",
    output_path: str | Path | None = None,
) -> tuple[AnalysisReport, Path]:
    """依次完成简历解析、岗位解析、匹配和 JSON 报告落盘。"""
    resume = parse_resume(resume_path, provider=provider)
    job = parse_job_description(job_path, provider=provider)
    report = match_requirements(resume, job)
    destination = Path(output_path).expanduser().resolve() if output_path else _default_report_path(job)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report, destination
