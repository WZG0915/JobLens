"""阶段二 Agent 模型—工具循环测试；全部使用假模型，不访问网络。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from joblens.agent import (
    AgentModelResponse,
    AgentRunner,
    AgentStopReason,
    AgentToolCall,
    build_default_registry,
)
from joblens.exceptions import ProviderResponseError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JOB = "data/jobs/04_baidu_llm_application_engineer.md"


class ScriptedClient:
    """按顺序返回预设响应，并保存每轮收到的输入，供断言使用。"""

    def __init__(self, responses: list[AgentModelResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[list[dict[str, Any]]] = []

    def create_response(self, **kwargs: Any) -> AgentModelResponse:
        self.requests.append(list(kwargs["input_items"]))
        return self.responses.pop(0)


def _call(call_id: str = "call_1") -> AgentModelResponse:
    arguments = json.dumps({"file_path": JOB}, ensure_ascii=False)
    return AgentModelResponse(
        response_id="resp_tool",
        tool_calls=[
            AgentToolCall(
                call_id=call_id,
                name="parse_job_description",
                arguments=arguments,
            )
        ],
        output_items=[
            {
                "type": "function_call",
                "call_id": call_id,
                "name": "parse_job_description",
                "arguments": arguments,
            }
        ],
    )


def test_agent_completes_without_tool_when_model_answers_directly() -> None:
    """一般咨询可以直接回答，不应为了调用而调用工具。"""

    client = ScriptedClient([AgentModelResponse(output_text="请提供简历和岗位文件路径。")])
    runner = AgentRunner(client, build_default_registry(PROJECT_ROOT))
    result = runner.run("我需要准备什么？")
    assert result.success
    assert result.stop_reason == AgentStopReason.COMPLETED
    assert result.tool_call_count == 0
    assert result.model_turns == 1


def test_agent_runner_receives_rag_recommendation_tool() -> None:
    """默认注册中心导出的工具会直接进入AgentRunner的模型决策上下文。"""
    class InspectingClient:
        def __init__(self) -> None:
            self.tool_names: list[str] = []

        def create_response(self, **kwargs: Any) -> AgentModelResponse:
            self.tool_names = [tool["name"] for tool in kwargs["tools"]]
            return AgentModelResponse(output_text="已看到推荐工具。")

    client = InspectingClient()
    result = AgentRunner(client, build_default_registry(PROJECT_ROOT)).run("推荐岗位")
    assert result.success
    assert "recommend_jobs_with_rag" in client.tool_names


def test_agent_executes_tool_and_returns_output_to_next_model_turn() -> None:
    """模型提出函数调用后，运行器应执行并用 call_id 回传结果。"""

    client = ScriptedClient(
        [_call(), AgentModelResponse(response_id="resp_final", output_text="岗位解析完成。")]
    )
    runner = AgentRunner(client, build_default_registry(PROJECT_ROOT))
    result = runner.run(f"请解析 {JOB}")
    assert result.success
    assert result.tool_call_count == 1
    assert result.steps[0].tool_calls[0].ok
    second_input = client.requests[1]
    output = next(item for item in second_input if item.get("type") == "function_call_output")
    assert output["call_id"] == "call_1"
    payload = json.loads(output["output"])
    assert payload["ok"] is True
    assert payload["tool_name"] == "parse_job_description"


def test_agent_returns_tool_error_to_model_for_recovery() -> None:
    """未知工具不直接崩溃，结构化错误仍交给下一轮模型解释。"""

    bad_call = AgentModelResponse(
        tool_calls=[
            AgentToolCall(call_id="bad_1", name="run_python", arguments="{}")
        ]
    )
    client = ScriptedClient(
        [bad_call, AgentModelResponse(output_text="该操作不在允许工具范围内。")]
    )
    result = AgentRunner(client, build_default_registry(PROJECT_ROOT)).run("运行代码")
    assert result.success
    trace = result.steps[0].tool_calls[0]
    assert not trace.ok
    assert trace.error_code == "unknown_tool"


def test_repeated_identical_calls_are_stopped() -> None:
    """同一调用反复出现时，应在第三次执行前停止循环。"""

    client = ScriptedClient([_call("c1"), _call("c2"), _call("c3")])
    runner = AgentRunner(
        client,
        build_default_registry(PROJECT_ROOT),
        max_steps=5,
        max_identical_calls=2,
    )
    result = runner.run("反复解析")
    assert not result.success
    assert result.stop_reason == AgentStopReason.REPEATED_TOOL_CALL
    assert result.tool_call_count == 2


def test_max_steps_and_empty_response_stop_safely() -> None:
    """步数耗尽以及空响应都应得到明确停止原因。"""

    step_client = ScriptedClient([_call()])
    step_result = AgentRunner(
        step_client,
        build_default_registry(PROJECT_ROOT),
        max_steps=1,
    ).run("解析岗位")
    assert step_result.stop_reason == AgentStopReason.MAX_STEPS

    empty_client = ScriptedClient([AgentModelResponse()])
    empty_result = AgentRunner(
        empty_client,
        build_default_registry(PROJECT_ROOT),
    ).run("你好")
    assert empty_result.stop_reason == AgentStopReason.EMPTY_RESPONSE


def test_tool_call_budget_is_enforced_before_extra_execution() -> None:
    """工具预算耗尽后，下一次调用不得被执行。"""

    client = ScriptedClient([_call("budget_1"), _call("budget_2")])
    result = AgentRunner(
        client,
        build_default_registry(PROJECT_ROOT),
        max_steps=3,
        max_tool_calls=1,
    ).run("解析岗位")
    assert not result.success
    assert result.stop_reason == AgentStopReason.MAX_TOOL_CALLS
    assert result.tool_call_count == 1


def test_model_failures_are_converted_to_structured_result() -> None:
    """模型服务异常不能向命令行泄漏原始异常。"""

    class FailingClient:
        def create_response(self, **kwargs: Any) -> AgentModelResponse:
            raise ProviderResponseError("测试错误")

    result = AgentRunner(
        FailingClient(),
        build_default_registry(PROJECT_ROOT),
    ).run("测试")
    assert not result.success
    assert result.stop_reason == AgentStopReason.MODEL_ERROR
    assert result.errors == ["测试错误"]
