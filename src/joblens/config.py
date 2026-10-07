"""外部 LLM 配置，只从环境变量读取敏感信息。"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from joblens.exceptions import ProviderConfigurationError


# 已验证支持 OpenAI Responses API 兼容格式的常用提供商地址。
# 用户显式填写 LLM_BASE_URL 时始终优先使用用户配置。
PROVIDER_BASE_URLS: dict[str, str | None] = {
    "openai": None,
    "deepseek": "https://api.deepseek.com",
}


@dataclass(frozen=True)
class LLMSettings:
    """外部大模型连接配置；冻结后可避免运行期间被意外修改。"""

    model: str
    api_key: str
    base_url: str | None = None
    provider: str = "openai"


def load_llm_settings() -> LLMSettings:
    """加载 OpenAI 兼容配置；缺少必需项时尽早给出可读错误。"""
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower() or "openai"
    model = os.getenv("LLM_MODEL", "").strip()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    configured_base_url = os.getenv("LLM_BASE_URL", "").strip() or None
    missing = [name for name, value in (("LLM_MODEL", model), ("LLM_API_KEY", api_key)) if not value]
    if missing:
        raise ProviderConfigurationError(
            "OpenAI 模式缺少环境变量：" + "、".join(missing)
        )
    if configured_base_url:
        base_url = configured_base_url
    elif provider in PROVIDER_BASE_URLS:
        base_url = PROVIDER_BASE_URLS[provider]
    else:
        raise ProviderConfigurationError(
            f"未知 LLM_PROVIDER：{provider}；请同时设置 LLM_BASE_URL"
        )
    return LLMSettings(
        model=model,
        api_key=api_key,
        base_url=base_url,
        provider=provider,
    )
