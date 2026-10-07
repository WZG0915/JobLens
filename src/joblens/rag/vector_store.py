"""Chroma 岗位向量数据库适配层。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from joblens.exceptions import RagDependencyError, RagIndexError

from .embedding import EmbeddingProvider
from .schemas import JobChunk, JobChunkHit


DEFAULT_COLLECTION_NAME = "joblens_job_chunks"


class ChromaJobVectorStore:
    """只存取 JobChunk 的 Chroma 封装。

    代码主动传入向量，不使用 Chroma 的默认在线 Embedding，确保所有向量
    都来自项目配置的本地 BGE 模型。
    """

    def __init__(
        self,
        persist_directory: str | Path,
        embedding: EmbeddingProvider,
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        client: Any | None = None,
    ) -> None:
        if len(collection_name) < 3:
            raise ValueError("Chroma collection_name 至少需要 3 个字符")
        self.persist_directory = Path(persist_directory).expanduser().resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embedding = embedding
        self.collection_name = collection_name
        self._client = client or self._create_client()
        self._collection = self._get_or_create_collection()
        self._validate_collection_compatibility()

    def _create_client(self) -> Any:
        """延迟导入 Chroma，避免影响只使用解析器的用户。"""
        try:
            import chromadb
        except ImportError as exc:
            raise RagDependencyError(
                '缺少 Chroma，请运行 pip install -e ".[rag]"'
            ) from exc
        try:
            return chromadb.PersistentClient(path=str(self.persist_directory))
        except Exception as exc:
            raise RagIndexError(f"无法打开 Chroma 数据库：{self.persist_directory}") from exc

    def _collection_metadata(self) -> dict[str, str | int]:
        """记录向量模型身份，防止不同维度或模型混用。"""
        return {
            "hnsw:space": "cosine",
            "joblens_schema": "job_chunk_v1",
            "embedding_model": self.embedding.model_name,
            "embedding_dimension": self.embedding.dimension,
        }

    def _get_or_create_collection(self) -> Any:
        """取得固定集合；不存在时创建余弦距离集合。"""
        try:
            return self._client.get_or_create_collection(
                name=self.collection_name,
                metadata=self._collection_metadata(),
            )
        except Exception as exc:
            raise RagIndexError(f"无法创建或读取 Chroma 集合：{self.collection_name}") from exc

    def _validate_collection_compatibility(self) -> None:
        """有数据的集合必须继续使用相同模型和维度。"""
        metadata = self._collection.metadata or {}
        if self._collection.count() == 0:
            return
        stored_model = str(metadata.get("embedding_model", ""))
        stored_dimension = int(metadata.get("embedding_dimension", 0) or 0)
        if stored_model != self.embedding.model_name or stored_dimension != self.embedding.dimension:
            raise RagIndexError(
                "现有 Chroma 索引与当前 BGE 模型不兼容；"
                f"索引={stored_model}/{stored_dimension}，"
                f"当前={self.embedding.model_name}/{self.embedding.dimension}"
            )

    @staticmethod
    def _metadata(chunk: JobChunk) -> dict[str, str | int | float | bool]:
        """Chroma 元数据仅支持标量，因此完整 Chunk 以 JSON 保存。"""
        metadata: dict[str, str | int | float | bool] = {
            "chunk_json": chunk.model_dump_json(),
            "job_id": chunk.job_id,
            "section": chunk.section.value,
            "source_path": chunk.source_path,
            "source_hash": chunk.source_hash,
            "title": chunk.title,
        }
        if chunk.company:
            metadata["company"] = chunk.company
        return metadata

    def rebuild(self, chunks: Sequence[JobChunk], *, batch_size: int = 64) -> int:
        """完整重建集合；先生成全部向量，成功后才替换旧集合。"""
        if batch_size < 1:
            raise ValueError("Chroma batch_size 必须大于 0")
        if not chunks:
            raise ValueError("不能用空 Chunk 集合构建岗位索引")
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("待写入 Chroma 的 chunk_id 存在重复")

        # 模型失败时保留旧集合，减少半成品索引覆盖可用索引的风险。
        embedding_texts = [chunk.to_embedding_text() for chunk in chunks]
        embeddings = self.embedding.embed_documents(embedding_texts)
        if len(embeddings) != len(chunks):
            raise RagIndexError("Embedding 数量与 Chunk 数量不一致")

        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            # 首次创建或空测试客户端可能不存在集合；随后统一重新创建。
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata=self._collection_metadata(),
        )

        for start in range(0, len(chunks), batch_size):
            chunk_batch = chunks[start : start + batch_size]
            self._collection.upsert(
                ids=[chunk.chunk_id for chunk in chunk_batch],
                embeddings=embeddings[start : start + batch_size],
                documents=[chunk.text for chunk in chunk_batch],
                metadatas=[self._metadata(chunk) for chunk in chunk_batch],
            )
        count = self.count()
        if count != len(chunks):
            raise RagIndexError(f"Chroma 写入不完整：期望 {len(chunks)}，实际 {count}")
        return count

    def count(self) -> int:
        """返回当前集合中的块数量。"""
        return int(self._collection.count())

    def indexed_source_hashes(self) -> dict[str, str]:
        """读取索引内每个岗位的原文哈希，用于发现岗位库已变化。"""
        try:
            result = self._collection.get(include=["metadatas"])
        except Exception as exc:
            raise RagIndexError("无法读取 Chroma 索引元数据") from exc
        hashes: dict[str, str] = {}
        for metadata in result.get("metadatas") or []:
            if not metadata:
                continue
            job_id = str(metadata.get("job_id", ""))
            source_hash = str(metadata.get("source_hash", ""))
            if job_id and source_hash:
                previous = hashes.get(job_id)
                if previous is not None and previous != source_hash:
                    raise RagIndexError(f"岗位 {job_id} 在索引中存在多个原文哈希")
                hashes[job_id] = source_hash
        return hashes

    def query(
        self,
        query_text: str,
        *,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[JobChunkHit]:
        """执行一次余弦向量检索，并恢复项目自己的 JobChunk 模型。"""
        if n_results < 1:
            raise ValueError("n_results 必须大于 0")
        collection_count = self.count()
        if collection_count == 0:
            return []
        try:
            result = self._collection.query(
                query_embeddings=[self.embedding.embed_query(query_text)],
                n_results=min(n_results, collection_count),
                where=where,
                include=["metadatas", "distances"],
            )
        except Exception as exc:
            raise RagIndexError("Chroma 岗位检索失败") from exc

        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        hits: list[JobChunkHit] = []
        for rank, (metadata, distance) in enumerate(zip(metadatas, distances), start=1):
            try:
                chunk_data = json.loads(str(metadata["chunk_json"]))
                chunk = JobChunk.model_validate(chunk_data)
            except (KeyError, TypeError, ValueError) as exc:
                raise RagIndexError("Chroma 中的岗位 Chunk 元数据已损坏") from exc
            score = max(0.0, min(1.0, 1.0 - float(distance)))
            hits.append(
                JobChunkHit(
                    rank=rank,
                    chunk=chunk,
                    final_score=score,
                    vector_score=score,
                )
            )
        return hits
