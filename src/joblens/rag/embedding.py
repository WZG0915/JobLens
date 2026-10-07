"""本地 BGE Embedding 的轻量封装。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

from joblens.exceptions import RagDependencyError, RagIndexError


DEFAULT_BGE_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


@runtime_checkable
class EmbeddingProvider(Protocol):
    """向量数据库所依赖的最小 Embedding 接口。"""

    @property
    def model_name(self) -> str:
        """返回模型名称，用于阻止不同模型的向量混写。"""

    @property
    def dimension(self) -> int:
        """返回向量维度。"""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """批量编码岗位 Chunk。"""

    def embed_query(self, text: str) -> list[float]:
        """编码一条检索查询。"""


@dataclass(frozen=True)
class BgeEmbeddingSettings:
    """BGE 本地运行参数；默认模型兼顾中文效果和实验成本。"""

    model_name: str = DEFAULT_BGE_MODEL
    device: str = "auto"
    batch_size: int = 32
    normalize_embeddings: bool = True
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    local_files_only: bool = False

    @classmethod
    def from_environment(cls) -> "BgeEmbeddingSettings":
        """允许通过环境变量替换模型、设备和批大小。"""
        batch_text = os.getenv("JOBLENS_BGE_BATCH_SIZE", "32").strip()
        try:
            batch_size = int(batch_text)
        except ValueError as exc:
            raise ValueError("JOBLENS_BGE_BATCH_SIZE 必须是正整数") from exc
        if batch_size < 1:
            raise ValueError("JOBLENS_BGE_BATCH_SIZE 必须是正整数")
        return cls(
            model_name=os.getenv("JOBLENS_BGE_MODEL", DEFAULT_BGE_MODEL).strip()
            or DEFAULT_BGE_MODEL,
            device=os.getenv("JOBLENS_BGE_DEVICE", "auto").strip() or "auto",
            batch_size=batch_size,
            local_files_only=os.getenv("JOBLENS_BGE_LOCAL_ONLY", "0").strip().lower()
            in {"1", "true", "yes"},
        )


class BgeLocalEmbedding:
    """通过 sentence-transformers 在本机运行 BGE。

    模型在对象初始化时加载一次，后续批量编码会复用同一个实例。测试可以
    注入兼容 ``encode`` 的假模型，从而不下载权重。
    """

    def __init__(
        self,
        settings: BgeEmbeddingSettings | None = None,
        *,
        model: Any | None = None,
    ) -> None:
        self.settings = settings or BgeEmbeddingSettings.from_environment()
        if not self.settings.model_name:
            raise ValueError("BGE 模型名称不能为空")
        if self.settings.batch_size < 1:
            raise ValueError("BGE batch_size 必须大于 0")
        self._model = model or self._load_model()
        # sentence-transformers 5.7 将方法改名；兼容旧版，避免用户被版本锁死。
        dimension_getter = getattr(self._model, "get_embedding_dimension", None)
        if not callable(dimension_getter):
            dimension_getter = getattr(self._model, "get_sentence_embedding_dimension", None)
        if not callable(dimension_getter):
            raise RagIndexError("BGE 模型没有提供向量维度读取方法")
        dimension = dimension_getter()
        if not dimension or int(dimension) < 1:
            raise RagIndexError("BGE 模型没有返回有效的向量维度")
        self._dimension = int(dimension)

    def _load_model(self) -> Any:
        """延迟导入重依赖，让非 RAG 命令仍可轻量运行。"""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RagDependencyError(
                '缺少本地 RAG 依赖，请运行 pip install -e ".[rag]"'
            ) from exc

        options: dict[str, Any] = {}
        if self.settings.device != "auto":
            options["device"] = self.settings.device
        if self.settings.local_files_only:
            options["local_files_only"] = True
        try:
            return SentenceTransformer(self.settings.model_name, **options)
        except Exception as exc:
            raise RagIndexError(
                f"无法加载 BGE 模型 {self.settings.model_name}；"
                "首次使用需要联网下载，之后可设置 JOBLENS_BGE_LOCAL_ONLY=1"
            ) from exc

    @property
    def model_name(self) -> str:
        """返回当前模型名称。"""
        return self.settings.model_name

    @property
    def dimension(self) -> int:
        """返回模型输出维度。"""
        return self._dimension

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        """执行统一的批量编码和返回值校验。"""
        values = [text.strip() for text in texts]
        if not values or any(not value for value in values):
            raise ValueError("待编码文本不能为空")
        try:
            encoded = self._model.encode(
                values,
                batch_size=self.settings.batch_size,
                normalize_embeddings=self.settings.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise RagIndexError("BGE 文本编码失败") from exc
        rows = encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)
        vectors = [[float(number) for number in row] for row in rows]
        if len(vectors) != len(values) or any(
            len(vector) != self.dimension for vector in vectors
        ):
            raise RagIndexError("BGE 返回的向量数量或维度不正确")
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """编码岗位文本；文档端不添加查询指令。"""
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        """编码查询，并使用 BGE 中文检索指令增强查询侧表达。"""
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("查询文本不能为空")
        query_text = f"{self.settings.query_instruction}{cleaned}"
        return self._encode([query_text])[0]
