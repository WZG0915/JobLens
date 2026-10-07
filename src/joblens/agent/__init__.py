"""JobLens Agent 的模型工具契约、注册中心和阶段二运行器。

这个 ``__init__.py`` 是包的公开入口。外部代码不需要了解各类具体放在哪个文件，
可以直接使用 ``from joblens.agent import build_default_registry``。
"""

from .builtin_tools import (
    MatchEvidenceInput,
    MatchEvidenceOutput,
    ParseJobInput,
    ParseJobOutput,
    ParseResumeInput,
    ParseResumeOutput,
    RecommendJobsInput,
    RecommendJobsOutput,
    RecommendedJobSummary,
    builtin_tools,
)
from .contracts import (
    ModelTool,
    ToolContext,
    ToolError,
    ToolErrorCode,
    ToolExecutionResult,
    ToolSafetyPolicy,
)
from .registry import ToolRegistry, build_default_registry
from .model_client import (
    AgentModelClient,
    AgentModelResponse,
    AgentToolCall,
    OpenAIResponsesClient,
)
from .prompts import DEFAULT_AGENT_INSTRUCTIONS
from .runner import (
    AgentRunResult,
    AgentRunner,
    AgentStepTrace,
    AgentStopReason,
    ToolCallTrace,
)

# __all__ 明确声明这个包希望对外提供的名称；以下划线开头的内部处理函数不会暴露。
__all__ = [
    "AgentModelClient",
    "AgentModelResponse",
    "AgentRunResult",
    "AgentRunner",
    "AgentStepTrace",
    "AgentStopReason",
    "AgentToolCall",
    "DEFAULT_AGENT_INSTRUCTIONS",
    "MatchEvidenceInput",
    "MatchEvidenceOutput",
    "ModelTool",
    "ParseJobInput",
    "ParseJobOutput",
    "ParseResumeInput",
    "ParseResumeOutput",
    "RecommendJobsInput",
    "RecommendJobsOutput",
    "RecommendedJobSummary",
    "ToolContext",
    "ToolError",
    "ToolErrorCode",
    "ToolExecutionResult",
    "ToolRegistry",
    "ToolSafetyPolicy",
    "ToolCallTrace",
    "OpenAIResponsesClient",
    "build_default_registry",
    "builtin_tools",
]
