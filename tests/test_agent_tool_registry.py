"""模型工具统一契约、白名单和安全边界测试。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from joblens.agent import ToolErrorCode, build_default_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESUME = PROJECT_ROOT / "data" / "resumes" / "01_cn_standard_single_page.pdf"
JOB = PROJECT_ROOT / "data" / "jobs" / "04_baidu_llm_application_engineer.md"


def test_registry_exposes_exactly_four_allowlisted_tools() -> None:
    """默认注册中心只能暴露四个经过审核的业务工具。"""
    registry = build_default_registry(PROJECT_ROOT)
    assert registry.names() == (
        "match_resume_job_evidence",
        "parse_job_description",
        "parse_resume_pdf",
        "recommend_jobs_with_rag",
    )


def test_openai_responses_schemas_are_strict() -> None:
    """Responses API 工具定义必须使用严格且封闭的对象 Schema。"""
    registry = build_default_registry(PROJECT_ROOT)
    definitions = registry.openai_tools(api="responses")
    assert len(definitions) == 4
    for definition in definitions:
        assert definition["type"] == "function"
        assert definition["strict"] is True
        parameters = definition["parameters"]
        assert parameters["type"] == "object"
        assert parameters["additionalProperties"] is False
        assert set(parameters["required"]) == set(parameters["properties"])


def test_openai_chat_completions_shape_is_supported() -> None:
    """注册中心也能生成 Chat Completions 所需的嵌套格式。"""
    registry = build_default_registry(PROJECT_ROOT)
    definition = registry.openai_tools(api="chat_completions")[0]
    assert definition["type"] == "function"
    assert definition["function"]["strict"] is True


def test_unknown_tool_and_extra_arguments_are_rejected() -> None:
    """未知工具名和输入模型未声明的参数都必须被拒绝。"""
    registry = build_default_registry(PROJECT_ROOT)
    unknown = registry.execute("run_python", {})
    assert not unknown.ok
    assert unknown.error is not None
    assert unknown.error.code == ToolErrorCode.UNKNOWN_TOOL

    invalid = registry.execute(
        "parse_job_description",
        {"file_path": str(JOB), "command": "whoami"},
    )
    assert not invalid.ok
    assert invalid.error is not None
    assert invalid.error.code == ToolErrorCode.INVALID_ARGUMENTS

    invalid_top_k = registry.execute(
        "recommend_jobs_with_rag",
        {"resume_path": str(RESUME), "top_k": 99},
    )
    assert not invalid_top_k.ok
    assert invalid_top_k.error is not None
    assert invalid_top_k.error.code == ToolErrorCode.INVALID_ARGUMENTS


def test_file_outside_allowed_roots_is_rejected() -> None:
    """模型不能读取项目允许目录之外的文件。"""
    registry = build_default_registry(PROJECT_ROOT)
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "outside.pdf"
        path.write_bytes(b"not a real pdf")
        result = registry.execute("parse_resume_pdf", {"file_path": str(path)})
    assert not result.ok
    assert result.error is not None
    assert result.error.code == ToolErrorCode.SAFETY_VIOLATION


def test_resume_tool_returns_compact_non_contact_summary() -> None:
    """简历工具返回精简结果，且不泄露联系方式和完整原文。"""
    registry = build_default_registry(PROJECT_ROOT)
    result = registry.execute(
        "parse_resume_pdf",
        json.dumps({"file_path": str(RESUME)}, ensure_ascii=False),
    )
    assert result.ok, result.error
    assert result.data is not None
    assert result.data["page_count"] in {1, 2}
    assert "raw_text" not in result.data
    assert "contact" not in result.data


def test_job_tool_and_local_evidence_tool_execute() -> None:
    """岗位解析和本地证据匹配工具能够处理真实项目数据。"""
    registry = build_default_registry(PROJECT_ROOT)
    job = registry.execute("parse_job_description", {"file_path": str(JOB)})
    assert job.ok, job.error
    assert job.data is not None
    assert job.data["requirements"]

    match = registry.execute(
        "match_resume_job_evidence",
        {"resume_path": str(RESUME), "job_path": str(JOB)},
    )
    assert match.ok, match.error
    assert match.data is not None
    assert 0 <= match.data["weighted_score"] <= 100
    assert match.data["requirement_matches"]


def test_safety_manifest_declares_no_network_or_external_transfer() -> None:
    """默认四个工具必须声明为只读、无网络且不外传数据。"""
    registry = build_default_registry(PROJECT_ROOT)
    manifest = registry.safety_manifest()
    assert all(item["safety"]["read_only"] for item in manifest)
    assert all(not item["safety"]["network_access"] for item in manifest)
    assert all(not item["safety"]["external_data_transfer"] for item in manifest)
