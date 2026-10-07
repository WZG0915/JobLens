"""Agent 使用的大模型客户端抽象与 OpenAI Responses API 实现。

运行器依赖 :class:`AgentModelClient` 协议，而不是直接依赖某个 SDK。这样单元测试可用
脚本化假客户端，不消耗 API，也能精确验证工具调用循环。
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from joblens.config import LLMSettings, load_llm_settings
from joblens.exceptions import ProviderConfigurationError, ProviderResponseError


LOGGER = logging.getLogger(__name__)


class AgentModelData(BaseModel):
    """模型客户端数据对象的公共配置。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class AgentToolCall(AgentModelData):
    """从一次模型响应中提取出的函数调用。"""

    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    # Responses API 返回 JSON 字符串；保持原样交给注册中心统一解析和校验。
    arguments: str


class AgentModelResponse(AgentModelData):
    """运行器真正关心的一次模型响应。"""

    response_id: str | None = None
    output_text: str = ""
    tool_calls: list[AgentToolCall] = Field(default_factory=list)
    # 包含 reasoning/function_call/message 等项目，用于无状态地延续下一轮请求。
    output_items: list[dict[str, Any]] = Field(default_factory=list)


@runtime_checkable
class AgentModelClient(Protocol):
    """所有 Agent 模型客户端必须实现的最小接口。"""

    def create_response(
        self,
        *,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        instructions: str,
    ) -> AgentModelResponse:
        """根据当前对话历史和工具定义生成下一次模型响应。"""


class OpenAIResponsesClient:
    """使用 OpenAI Responses API 的生产客户端。

    默认 ``store=False``，并由运行器把上一轮 output items 显式放回下一轮输入。
    这既适配包含 reasoning items 的模型，也不依赖服务端会话存储。
    """

    def __init__(
        self,
        settings: LLMSettings | None = None,
        *,
        timeout_seconds: float = 60.0,
    ) -> None:
        self._settings = settings or load_llm_settings()
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderConfigurationError(
                '缺少 openai 依赖，请运行 pip install -e ".[llm]"'
            ) from exc

        options: dict[str, Any] = {
            "api_key": self._settings.api_key,
            "timeout": timeout_seconds,
        }
        if self._settings.base_url:
            options["base_url"] = self._settings.base_url
        self._client = OpenAI(**options)

    def create_response(
        self,
        *,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        instructions: str,
    ) -> AgentModelResponse:
        """调用 Responses API，并把 SDK 对象转换为稳定的项目内数据结构。"""

        try:
            response = self._client.responses.create(
                model=self._settings.model,
                instructions=instructions,
                input=input_items,
                tools=tools,
                tool_choice="auto",
                # 当前工具涉及较重的 PDF 解析；串行执行更容易控制资源与轨迹。
                parallel_tool_calls=False,
                store=False,
            )
        except Exception as exc:
            # 这里只记录异常类型，不打印 SDK 响应体，避免日志中出现 Key 片段或请求细节。
            LOGGER.warning(
                "OpenAI-compatible Responses API request failed: %s",
                type(exc).__name__,
            )
            raise ProviderResponseError(
                "大模型请求失败；请检查网络、模型名称、API Key 和服务端配置"
            ) from exc

        output_items = [_to_plain_dict(item) for item in (response.output or [])]
        tool_calls = [
            AgentToolCall(
                call_id=str(item.get("call_id", "")),
                name=str(item.get("name", "")),
                arguments=str(item.get("arguments", "{}")),
            )
            for item in output_items
            if item.get("type") == "function_call"
        ]
        return AgentModelResponse(
            response_id=getattr(response, "id", None),
            output_text=getattr(response, "output_text", "") or "",
            tool_calls=tool_calls,
            output_items=output_items,
        )


def _to_plain_dict(item: Any) -> dict[str, Any]:
    """把 OpenAI SDK 输出项转成下一轮可复用的普通 JSON 字典。"""

    if isinstance(item, dict):
        return dict(item)
    model_dump = getattr(item, "model_dump", None)
    if callable(model_dump):
        value = model_dump(mode="json", exclude_none=True)
        if isinstance(value, dict):
            return value
    raise ProviderResponseError("大模型返回了无法识别的输出项")
