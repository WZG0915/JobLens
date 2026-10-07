"""模型工具白名单注册中心与统一执行入口。

注册中心相当于 Agent 与业务代码之间的“安全总机”。大模型只能给出工具名和 JSON
参数，不能直接拿到 Python 函数。注册中心会依次完成：查找白名单、校验参数、执行、
校验输出、包装错误。后续实现 Agent Runner 时，只需要依赖这里的 ``openai_tools``
和 ``execute`` 两个主要方法。
"""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from joblens.exceptions import JobLensError

from .builtin_tools import builtin_tools
from .contracts import (
    ModelTool,
    ToolContext,
    ToolError,
    ToolErrorCode,
    ToolExecutionResult,
    ToolSafetyViolation,
    parse_tool_arguments,
    strict_json_schema,
)


LOGGER = logging.getLogger(__name__)
# OpenAI 函数工具名只使用字母、数字、下划线和连字符，并限制为 64 个字符。
TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# 防止工具返回过大的 JSON，挤占模型上下文或导致接口请求过大。
MAX_RESULT_BYTES = 256 * 1024


class ToolRegistry:
    """显式白名单注册、Schema 导出、参数验证与安全执行。

    一个常见使用流程是：

    1. 用 :func:`build_default_registry` 创建注册中心；
    2. 用 :meth:`openai_tools` 把已注册工具描述交给模型；
    3. 收到模型的工具调用后，用 :meth:`execute` 执行；
    4. 把 ``result.to_model_message()`` 作为工具结果传回模型。
    """

    def __init__(self, context: ToolContext) -> None:
        """使用可信安全上下文创建一个初始为空的注册中心。"""
        # context 保存可信的本地路径边界，不接收模型传入的覆盖值。
        self._context = context
        # 字典的键是模型调用时使用的工具名，值是完整 ModelTool 定义。
        self._tools: dict[str, ModelTool] = {}
        # RLock 让未来的并发 Agent 调用可以安全读取或注册工具。
        self._lock = threading.RLock()

    @property
    def context(self) -> ToolContext:
        """只读暴露注册中心使用的安全上下文。"""
        return self._context

    def register(self, tool: ModelTool) -> None:
        """注册一个工具；非法名称和重复名称都会立即报错。"""

        if not TOOL_NAME_RE.fullmatch(tool.name):
            raise ValueError(f"非法工具名：{tool.name!r}")
        if not tool.description.strip():
            raise ValueError("工具描述不能为空")
        with self._lock:
            if tool.name in self._tools:
                raise ValueError(f"工具已注册：{tool.name}")
            self._tools[tool.name] = tool

    def register_many(self, tools: list[ModelTool]) -> None:
        """批量注册工具，内部仍逐个执行完整校验。"""
        for tool in tools:
            self.register(tool)

    def names(self) -> tuple[str, ...]:
        """返回排序后的工具名快照，tuple 可避免调用方意外修改。"""
        with self._lock:
            return tuple(sorted(self._tools))

    def get(self, name: str) -> ModelTool | None:
        """按名字查找工具；不存在时返回 None。"""
        with self._lock:
            return self._tools.get(name)

    def openai_tools(
        self,
        *,
        api: Literal["responses", "chat_completions"] = "responses",
    ) -> list[dict[str, Any]]:
        """导出 OpenAI Responses 或 Chat Completions 的 function tools。

        两类 API 的工具内容相同，区别主要是 Chat Completions 需要额外的
        ``{"type": "function", "function": {...}}`` 包装层。
        """
        definitions: list[dict[str, Any]] = []
        # 加锁期间只复制快照，后面的 Schema 生成不长期占用锁。
        with self._lock:
            tools = [self._tools[name] for name in sorted(self._tools)]
        for tool in tools:
            # description 帮助模型选择工具；parameters 约束模型应该生成哪些参数。
            function = {
                "name": tool.name,
                "description": tool.description,
                "parameters": strict_json_schema(tool.input_model),
                "strict": True,
            }
            if api == "responses":
                # Responses API：name/description/parameters 与 type 位于同一层。
                definitions.append({"type": "function", **function})
            else:
                # Chat Completions API：函数定义放在 function 字段中。
                definitions.append({"type": "function", "function": function})
        return definitions

    def safety_manifest(self) -> list[dict[str, Any]]:
        """导出输入、输出和安全策略，供调试、文档和审计使用。

        安全清单不会直接执行工具，并且其中的策略来自服务端代码，模型无法通过
        工具参数把 ``network_access`` 或 ``allowed_extensions`` 改掉。
        """
        with self._lock:
            tools = [self._tools[name] for name in sorted(self._tools)]
        return [
            {
                "name": tool.name,
                "input_schema": strict_json_schema(tool.input_model),
                "output_schema": tool.output_model.model_json_schema(),
                "safety": tool.safety.model_dump(mode="json"),
            }
            for tool in tools
        ]

    def execute(
        self,
        name: str,
        arguments: str | dict[str, Any],
    ) -> ToolExecutionResult:
        """执行模型请求的工具，是所有工具调用的唯一安全入口。

        ``arguments`` 可以是 SDK 返回的 JSON 字符串，也可以是已经解析好的字典。
        本方法永远返回 ``ToolExecutionResult``，可预期错误不会继续向 Agent 外层抛出。
        """
        started = time.perf_counter()

        # 第一步：只允许调用显式注册过的工具。
        tool = self.get(name)
        if tool is None:
            return self._error(
                name or "<empty>",
                started,
                ToolErrorCode.UNKNOWN_TOOL,
                "工具未注册或不在允许调用的白名单中",
            )
        # 第二步：先解析 JSON，再使用该工具自己的 Pydantic 输入模型校验。
        try:
            payload = parse_tool_arguments(arguments)
            validated = tool.input_model.model_validate(payload)
        except (ValueError, ValidationError) as exc:
            details = (
                exc.errors(include_input=False, include_url=False)
                if isinstance(exc, ValidationError)
                else []
            )
            return self._error(
                name,
                started,
                ToolErrorCode.INVALID_ARGUMENTS,
                "工具参数校验失败",
                details=details,
            )

        # 第三步：执行处理函数，并再次校验输出。输入合法不代表实现一定返回合法结果。
        try:
            raw_output = tool.handler(validated, self._context)
            output = tool.output_model.model_validate(raw_output)
            # mode="json" 会把 Enum、Path 等值转换成普通 JSON 可序列化类型。
            data = output.model_dump(mode="json")
            result = ToolExecutionResult(
                ok=True,
                tool_name=name,
                data=data,
                elapsed_ms=_elapsed_ms(started),
            )
            # 在把结果交给模型前检查序列化后的真实字节数。
            if len(result.to_model_message().encode("utf-8")) > MAX_RESULT_BYTES:
                return self._error(
                    name,
                    started,
                    ToolErrorCode.INVALID_RESULT,
                    "工具结果超过 256 KB 安全限制",
                )
            return result
        # 以下异常被分成不同错误码，便于后续 Agent 做重试或向用户解释。
        except ToolSafetyViolation as exc:
            return self._error(
                name,
                started,
                ToolErrorCode.SAFETY_VIOLATION,
                str(exc),
            )
        except JobLensError as exc:
            return self._error(
                name,
                started,
                ToolErrorCode.EXECUTION_FAILED,
                str(exc),
            )
        except ValidationError as exc:
            # 这是工具实现错误，记录完整日志，但只向模型返回安全摘要。
            LOGGER.exception("Tool %s returned an invalid result", name)
            return self._error(
                name,
                started,
                ToolErrorCode.INVALID_RESULT,
                "工具实现返回了不符合输出契约的数据",
                details=exc.errors(include_input=False, include_url=False),
            )
        except Exception:
            # 不把 traceback、环境变量或底层依赖消息暴露给模型。
            LOGGER.exception("Unexpected failure while executing tool %s", name)
            return self._error(
                name,
                started,
                ToolErrorCode.EXECUTION_FAILED,
                "工具执行失败；请检查本地日志",
            )

    @staticmethod
    def _error(
        name: str,
        started: float,
        code: ToolErrorCode,
        message: str,
        *,
        retryable: bool = False,
        details: list[dict[str, Any]] | None = None,
    ) -> ToolExecutionResult:
        """构造统一失败结果，确保每个失败分支的格式完全一致。"""
        return ToolExecutionResult(
            ok=False,
            tool_name=name,
            error=ToolError(
                code=code,
                message=message,
                retryable=retryable,
                details=details or [],
            ),
            elapsed_ms=_elapsed_ms(started),
        )


def build_default_registry(
    project_root: str | Path | None = None,
    *,
    additional_allowed_roots: tuple[str | Path, ...] = (),
) -> ToolRegistry:
    """构建只包含显式审核业务工具的默认注册中心。

    默认项目根目录根据当前文件位置推导。测试或部署时也可以显式传入
    ``project_root``；只有确有需要时才应使用 ``additional_allowed_roots`` 扩大读取范围。
    """

    # registry.py 位于 project/src/joblens/agent，向上三级得到 project 根目录。
    root = (
        Path(project_root).resolve()
        if project_root
        else Path(__file__).resolve().parents[3]
    )
    # 额外目录由应用开发者配置，不会作为模型工具参数暴露出去。
    extra_roots = tuple(Path(path).expanduser().resolve() for path in additional_allowed_roots)
    context = ToolContext(
        project_root=root,
        allowed_roots=(root, *extra_roots),
    )
    registry = ToolRegistry(context)
    # 只有 builtin_tools() 明确返回的工具会进入默认白名单。
    registry.register_many(builtin_tools())
    return registry


def _elapsed_ms(started: float) -> float:
    """把 perf_counter 的时间差转换为毫秒，保留三位小数。"""
    return round((time.perf_counter() - started) * 1000, 3)
