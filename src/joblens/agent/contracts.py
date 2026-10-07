"""模型可调用工具的公共契约与安全上下文。

可以把本文件理解为 Agent 工具层的“统一接口说明书”：

1. 输入和输出必须长什么样，由 Pydantic 模型定义；
2. 一个工具有哪些基本信息，由 :class:`ModelTool` 定义；
3. 工具能访问哪些本地文件，由 :class:`ToolContext` 控制；
4. 无论成功还是失败，都统一返回 :class:`ToolExecutionResult`。

这样做的目的，是不让大模型直接调用任意 Python 函数，而是只能按照事先声明好的
参数格式调用注册中心中的白名单工具。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field


class ToolContractModel(BaseModel):
    """所有工具输入、输出模型的公共父类。

    ``extra="forbid"`` 表示如果模型多生成了一个未声明字段，校验会直接失败；
    ``str_strip_whitespace=True`` 会自动去掉字符串首尾的空白字符。
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ToolErrorCode(str, Enum):
    """工具失败原因的稳定编号，便于 Agent 判断下一步如何处理。"""

    # 模型请求了注册中心里不存在的工具。
    UNKNOWN_TOOL = "unknown_tool"
    # 工具存在，但模型生成的参数不符合输入模型。
    INVALID_ARGUMENTS = "invalid_arguments"
    # 参数格式正确，但违反了路径、扩展名或文件大小等安全规则。
    SAFETY_VIOLATION = "safety_violation"
    # 业务函数在解析或匹配过程中失败。
    EXECUTION_FAILED = "execution_failed"
    # 工具函数返回了不符合输出模型的数据。
    INVALID_RESULT = "invalid_result"


class ToolError(ToolContractModel):
    """统一错误对象；不要把原始 traceback 直接返回给大模型。"""

    code: ToolErrorCode
    message: str = Field(min_length=1)
    # 后续 Agent 可以根据这个字段决定是否自动重试。
    retryable: bool = False
    # 参数校验失败时可放入字段位置等结构化信息。
    details: list[dict[str, Any]] = Field(default_factory=list)


class ToolExecutionResult(ToolContractModel):
    """注册中心的统一执行结果，可直接作为 ``function_call_output``。

    成功时 ``data`` 有值、``error`` 为空；失败时则相反。使用统一外壳后，
    Agent Runner 不需要分别处理简历解析、岗位解析和匹配工具的异常格式。
    """

    ok: bool
    tool_name: str = Field(min_length=1)
    data: dict[str, Any] | None = None
    error: ToolError | None = None
    elapsed_ms: float = Field(ge=0)

    def to_model_message(self) -> str:
        """输出紧凑 JSON，避免把 Python 对象或异常直接交给模型。"""
        return self.model_dump_json(exclude_none=True)


class ToolSafetyPolicy(ToolContractModel):
    """描述工具能力边界的安全元数据。

    注意：当前策略主要用于声明、审计和后续 Agent 决策。真正的文件检查发生在
    :meth:`ToolContext.resolve_input_file` 中，不能只依赖模型“自觉遵守”。
    """

    read_only: bool = True
    network_access: bool = False
    external_data_transfer: bool = False
    handles_personal_data: bool = False
    allowed_extensions: list[str] = Field(default_factory=list)
    max_file_bytes: int = Field(default=25 * 1024 * 1024, ge=1)


class ToolSafetyViolation(ValueError):
    """模型参数违反文件边界或工具策略时抛出的专用异常。"""


@dataclass(frozen=True)
class ToolContext:
    """每次工具执行共享的、不可由模型修改的本地安全边界。

    ``project_root`` 用来解释相对路径，例如 ``data/resumes/a.pdf``；
    ``allowed_roots`` 是真正允许读取的目录白名单。
    """

    project_root: Path
    allowed_roots: tuple[Path, ...]

    def resolve_input_file(
        self,
        value: str,
        *,
        allowed_extensions: set[str],
        max_file_bytes: int,
    ) -> Path:
        """把模型给出的路径转换为经过安全校验的绝对文件路径。"""

        # 第一道检查：空路径以及 NUL 字符都不应进入文件系统 API。
        if not value or "\x00" in value:
            raise ToolSafetyViolation("文件路径为空或包含非法字符")
        candidate = Path(value).expanduser()
        # 模型常返回项目相对路径；统一相对于 project_root 进行解释。
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        try:
            # strict=True 要求文件真实存在；resolve 也会消解“..”和符号链接。
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ToolSafetyViolation("输入文件不存在或无法解析") from exc
        if not resolved.is_file():
            raise ToolSafetyViolation("输入路径不是普通文件")
        # 一定要检查解析后的真实路径，避免通过“../”或符号链接逃出允许目录。
        if not any(_is_relative_to(resolved, root) for root in self.allowed_roots):
            raise ToolSafetyViolation("输入文件不在允许访问的项目目录中")
        # 不同工具拥有不同扩展名白名单，例如简历工具目前只接受 PDF。
        if resolved.suffix.lower() not in allowed_extensions:
            supported = "、".join(sorted(allowed_extensions))
            raise ToolSafetyViolation(f"文件类型不受此工具支持；允许：{supported}")
        try:
            size = resolved.stat().st_size
        except OSError as exc:
            raise ToolSafetyViolation("无法读取输入文件信息") from exc
        if size <= 0:
            raise ToolSafetyViolation("输入文件为空")
        # 文件大小限制可防止模型误选超大文件，导致内存和处理时间失控。
        if size > max_file_bytes:
            raise ToolSafetyViolation(
                f"输入文件超过 {max_file_bytes // (1024 * 1024)} MB 安全限制"
            )
        return resolved


# 所有处理函数都遵循同一签名：已校验参数 + 安全上下文 -> 结构化输出。
ToolHandler = Callable[[BaseModel, ToolContext], BaseModel]


@dataclass(frozen=True)
class ModelTool:
    """把“模型看到的说明”和“实际执行函数”绑定成一个完整工具。

    例如 ``parse_resume_pdf`` 会绑定输入模型、输出模型、处理函数以及安全策略。
    ``frozen=True`` 防止注册完成后意外替换这些关键引用。
    """

    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: ToolHandler
    safety: ToolSafetyPolicy


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """把 Pydantic 模型转换为模型函数调用使用的严格 JSON Schema。

    Pydantic 会先生成普通 JSON Schema。随后递归访问其中的每一个对象：
    禁止额外字段，并把已声明字段全部加入 ``required``。
    """
    schema = model.model_json_schema()

    def visit(node: Any) -> None:
        """递归修改当前 Schema 节点及其所有子节点。"""
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                # 大模型只能生成 properties 中明确声明的键。
                node["additionalProperties"] = False
                node["required"] = list(properties)
            # schema 可能包含嵌套对象、数组和 $defs，因此需要递归处理。
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(schema)
    return schema


def parse_tool_arguments(arguments: str | dict[str, Any]) -> dict[str, Any]:
    """把模型参数统一解析为字典，只接受 JSON object。

    不同 SDK 可能传入 JSON 字符串，也可能已经转换成 Python ``dict``，
    所以这里同时支持两种形式，但拒绝数组、数字等不符合工具参数语义的顶层值。
    """
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        raise ValueError("工具参数必须是 JSON 对象或其字符串形式")
    if len(arguments.encode("utf-8")) > 64 * 1024:
        raise ValueError("工具参数超过 64 KB 限制")
    try:
        payload = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise ValueError("工具参数不是有效 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("工具参数的顶层必须是 JSON 对象")
    return payload


def _is_relative_to(path: Path, root: Path) -> bool:
    """判断 path 是否位于 root 内；不通过字符串前缀做不可靠比较。"""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
