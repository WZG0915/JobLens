"""JobLens 的领域异常。

命令行入口只需要捕获 :class:`JobLensError`，而业务代码仍可通过具体
子类区分文件、解析、配置和结构化输出问题。
"""


class JobLensError(Exception):
    """所有可预期 JobLens 错误的基类。"""


class DocumentReadError(JobLensError):
    """输入文档不存在、类型不受支持或无法读取。"""


class DocumentParseError(JobLensError):
    """文档存在，但不能解析为最低限度的业务结构。"""


class StructuredOutputError(JobLensError):
    """本地解析器或大模型返回的数据不符合 Pydantic 模型。"""


class ProviderConfigurationError(JobLensError):
    """外部模型提供商缺少依赖或环境变量。"""


class ProviderResponseError(JobLensError):
    """外部模型调用失败或没有返回可用的结构化结果。"""


class RagDependencyError(JobLensError):
    """本地 RAG 所需的可选依赖尚未安装。"""


class RagIndexError(JobLensError):
    """向量索引配置不一致、数据损坏或读写失败。"""
