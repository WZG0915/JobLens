"""大模型提供商配置测试。"""

from __future__ import annotations

from joblens.config import load_llm_settings


def test_deepseek_provider_uses_official_base_url(monkeypatch) -> None:
    """只填写 deepseek 提供商时，应自动选择其 OpenAI 兼容地址。"""
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("LLM_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)

    settings = load_llm_settings()

    assert settings.provider == "deepseek"
    assert settings.base_url == "https://api.deepseek.com"


def test_explicit_base_url_overrides_provider_default(monkeypatch) -> None:
    """私有网关或代理地址必须优先于内置提供商地址。"""
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("LLM_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://gateway.example/v1")

    settings = load_llm_settings()

    assert settings.base_url == "https://gateway.example/v1"
