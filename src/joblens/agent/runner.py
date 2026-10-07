"""JobLens 阶段二的 Agent 决策循环。

运行器负责协调模型和工具注册中心：模型只能提出函数调用，真正执行始终经过注册中心。
每次运行都有步数、调用数和重复调用上限，避免模型陷入无限循环。
"""

from __future__ import annotations

import json
from collections import Counter
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from joblens.exceptions import JobLensError

from .contracts import parse_tool_arguments
from .model_client import AgentModelClient, AgentModelResponse, AgentToolCall
from .prompts import DEFAULT_AGENT_INSTRUCTIONS
from .registry import ToolRegistry


class AgentRunData(BaseModel):
    """Agent 运行结果模型的公共配置。"""

    # 结果字段 ``model_turns`` 是业务含义，不是 Pydantic 的受保护方法名。
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        protected_namespaces=(),
    )


class AgentStopReason(str, Enum):
    """Agent 结束运行的稳定原因，便于 CLI、测试和未来前端判断。"""

    COMPLETED = "completed"
    MAX_STEPS = "max_steps"
    MAX_TOOL_CALLS = "max_tool_calls"
    REPEATED_TOOL_CALL = "repeated_tool_call"
    MODEL_ERROR = "model_error"
    EMPTY_RESPONSE = "empty_response"


class ToolCallTrace(AgentRunData):
    """一次工具调用的可审计记录。"""

    call_id: str
    tool_name: str
    arguments: dict[str, Any] | None = None
    arguments_valid_json: bool
    ok: bool
    error_code: str | None = None
    error_message: str | None = None
    elapsed_ms: float = Field(ge=0)


class AgentStepTrace(AgentRunData):
    """一次模型决策以及该轮触发的工具调用。"""

    step_number: int = Field(ge=1)
    response_id: str | None = None
    output_text: str = ""
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)


class AgentRunResult(AgentRunData):
    """一次完整 Agent 运行的结构化结果。"""

    success: bool
    answer: str
    stop_reason: AgentStopReason
    model_turns: int = Field(ge=0)
    tool_call_count: int = Field(ge=0)
    steps: list[AgentStepTrace] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class AgentRunner:
    """带安全上限、错误边界和轨迹记录的模型—工具循环。"""

    def __init__(
        self,
        model_client: AgentModelClient,
        registry: ToolRegistry,
        *,
        max_steps: int = 6,
        max_tool_calls: int = 6,
        max_identical_calls: int = 2,
        instructions: str = DEFAULT_AGENT_INSTRUCTIONS,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps 必须大于等于 1")
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls 必须大于等于 1")
        if max_identical_calls < 1:
            raise ValueError("max_identical_calls 必须大于等于 1")
        if not instructions.strip():
            raise ValueError("Agent 系统指令不能为空")
        self._model_client = model_client
        self._registry = registry
        self._max_steps = max_steps
        self._max_tool_calls = max_tool_calls
        self._max_identical_calls = max_identical_calls
        self._instructions = instructions

    def run(self, user_message: str) -> AgentRunResult:
        """运行一次完整任务，直到模型回答或命中安全停止条件。"""

        message = user_message.strip()
        if not message:
            raise ValueError("用户问题不能为空")

        # 显式保存全部输入历史；工具输出只包含注册中心生成的精简结构化摘要。
        input_items: list[dict[str, Any]] = [{"role": "user", "content": message}]
        tool_definitions = self._registry.openai_tools(api="responses")
        steps: list[AgentStepTrace] = []
        call_signatures: Counter[str] = Counter()
        tool_call_count = 0

        for step_number in range(1, self._max_steps + 1):
            try:
                response = self._model_client.create_response(
                    input_items=input_items,
                    tools=tool_definitions,
                    instructions=self._instructions,
                )
            except JobLensError as exc:
                return self._stopped(
                    AgentStopReason.MODEL_ERROR,
                    "Agent 无法完成请求：大模型调用失败。",
                    steps,
                    tool_call_count,
                    [str(exc)],
                )
            except Exception:
                # 自定义客户端也不能用未处理异常破坏应用边界。
                return self._stopped(
                    AgentStopReason.MODEL_ERROR,
                    "Agent 无法完成请求：模型客户端发生未预期错误。",
                    steps,
                    tool_call_count,
                    ["模型客户端发生未预期错误"],
                )

            self._append_model_output(input_items, response)
            step_trace = AgentStepTrace(
                step_number=step_number,
                response_id=response.response_id,
                output_text=response.output_text,
            )

            if not response.tool_calls:
                steps.append(step_trace)
                if response.output_text.strip():
                    return AgentRunResult(
                        success=True,
                        answer=response.output_text.strip(),
                        stop_reason=AgentStopReason.COMPLETED,
                        model_turns=len(steps),
                        tool_call_count=tool_call_count,
                        steps=steps,
                    )
                return self._stopped(
                    AgentStopReason.EMPTY_RESPONSE,
                    "Agent 未生成可用答复，请换一种问法后重试。",
                    steps,
                    tool_call_count,
                    ["模型既未返回文本，也未请求工具"],
                )

            for call in response.tool_calls:
                if tool_call_count >= self._max_tool_calls:
                    steps.append(step_trace)
                    return self._stopped(
                        AgentStopReason.MAX_TOOL_CALLS,
                        "Agent 已达到工具调用上限，任务已安全停止。",
                        steps,
                        tool_call_count,
                        [f"工具调用次数达到上限 {self._max_tool_calls}"],
                    )

                signature = _tool_call_signature(call)
                call_signatures[signature] += 1
                if call_signatures[signature] > self._max_identical_calls:
                    steps.append(step_trace)
                    return self._stopped(
                        AgentStopReason.REPEATED_TOOL_CALL,
                        "Agent 检测到重复工具调用，任务已安全停止。",
                        steps,
                        tool_call_count,
                        [f"相同工具与参数重复超过 {self._max_identical_calls} 次"],
                    )

                result = self._registry.execute(call.name, call.arguments)
                tool_call_count += 1
                arguments, valid_json = _trace_arguments(call.arguments)
                step_trace.tool_calls.append(
                    ToolCallTrace(
                        call_id=call.call_id,
                        tool_name=call.name,
                        arguments=arguments,
                        arguments_valid_json=valid_json,
                        ok=result.ok,
                        error_code=result.error.code.value if result.error else None,
                        error_message=result.error.message if result.error else None,
                        elapsed_ms=result.elapsed_ms,
                    )
                )
                # call_id 将本地执行结果与模型提出的函数调用精确对应。
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": result.to_model_message(),
                    }
                )
            steps.append(step_trace)

        return self._stopped(
            AgentStopReason.MAX_STEPS,
            "Agent 已达到最大决策步数，任务已安全停止。",
            steps,
            tool_call_count,
            [f"模型决策轮数达到上限 {self._max_steps}"],
        )

    @staticmethod
    def _append_model_output(
        input_items: list[dict[str, Any]],
        response: AgentModelResponse,
    ) -> None:
        """把模型原始输出加入历史；测试客户端可只提供 tool_calls。"""

        if response.output_items:
            input_items.extend(response.output_items)
            return
        # 兼容最小化的自定义客户端，生产 OpenAI 客户端会提供完整 output_items。
        for call in response.tool_calls:
            input_items.append(
                {
                    "type": "function_call",
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments": call.arguments,
                }
            )

    @staticmethod
    def _stopped(
        reason: AgentStopReason,
        answer: str,
        steps: list[AgentStepTrace],
        tool_call_count: int,
        errors: list[str],
    ) -> AgentRunResult:
        """构造所有非成功停止分支的统一结果。"""

        return AgentRunResult(
            success=False,
            answer=answer,
            stop_reason=reason,
            model_turns=len(steps),
            tool_call_count=tool_call_count,
            steps=steps,
            errors=errors,
        )


def _tool_call_signature(call: AgentToolCall) -> str:
    """生成稳定签名，用来识别模型是否反复调用相同工具和参数。"""

    try:
        payload = parse_tool_arguments(call.arguments)
        normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except ValueError:
        normalized = call.arguments.strip()
    return f"{call.name}:{normalized}"


def _trace_arguments(arguments: str) -> tuple[dict[str, Any] | None, bool]:
    """只在参数是合法 JSON object 时写入结构化执行轨迹。"""

    try:
        return parse_tool_arguments(arguments), True
    except ValueError:
        return None, False
