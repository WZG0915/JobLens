"""本地对话网页的轻量接口测试。"""

from __future__ import annotations

from pathlib import Path

import pytest


fastapi = pytest.importorskip("fastapi")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient

from joblens.agent import AgentModelResponse
from joblens.web import create_app


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_home_and_health_are_available() -> None:
    client = TestClient(create_app(PROJECT_ROOT))
    home = client.get("/")
    health = client.get("/api/health")

    assert home.status_code == 200
    assert "JobLens" in home.text
    assert health.status_code == 200
    assert health.json()["jobs_ready"] is True


def test_chat_requires_pdf_and_rejects_fake_pdf() -> None:
    client = TestClient(create_app(PROJECT_ROOT))
    missing = client.post("/api/chat", data={"message": "推荐岗位", "top_k": "5"})
    assert missing.status_code == 400

    fake = client.post(
        "/api/chat",
        data={"message": "解析简历", "top_k": "5"},
        files={"resume_file": ("resume.pdf", b"not-pdf", "application/pdf")},
    )
    assert fake.status_code == 400
    assert "有效的PDF" in fake.json()["detail"]


class DirectAnswerClient:
    """测试专用模型客户端：不联网，直接返回一条可识别答复。"""

    def create_response(self, *, input_items, tools, instructions) -> AgentModelResponse:
        assert input_items
        assert tools
        assert "JobLens" in instructions
        return AgentModelResponse(output_text="这是 AI 模型生成的测试答复。")


def test_chat_uses_injected_ai_client() -> None:
    """网页不应再依赖关键词路由才能产生自然语言答复。"""
    client = TestClient(
        create_app(PROJECT_ROOT, model_client_factory=DirectAnswerClient)
    )
    response = client.post(
        "/api/chat",
        data={"message": "请告诉我下一步该怎么准备", "top_k": "3"},
        files={"resume_file": ("resume.pdf", b"%PDF-1.4\n%%EOF", "application/pdf")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["message"] == "这是 AI 模型生成的测试答复。"
    assert payload["answered_by"] == "ai"
    assert payload["tool_calls"] == []

    health = client.get("/api/health").json()
    assert health["ai_ready"] is True
    assert health["dialogue_mode"] == "ai"
