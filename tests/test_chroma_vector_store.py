"""Chroma 适配层的可选集成测试。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from joblens.rag import ChromaJobVectorStore, JobChunk, JobChunkSection


chromadb = pytest.importorskip("chromadb")


class FakeEmbedding:
    """用文本哈希生成可复现向量，测试时不依赖真实 BGE 权重。"""

    model_name = "fake-embedding-v1"
    dimension = 4

    @staticmethod
    def _one(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [float(digest[index]) / 255 for index in range(4)]
        length = sum(value * value for value in values) ** 0.5 or 1.0
        return [value / length for value in values]

    def embed_documents(self, texts):
        return [self._one(text) for text in texts]

    def embed_query(self, text):
        return self._one(text)


def _chunk(index: int) -> JobChunk:
    return JobChunk(
        chunk_id=f"job_001:basic:{index}",
        job_id="job_001",
        source_path=str(Path("job.md").resolve()),
        source_hash="a" * 64,
        section=JobChunkSection.BASIC,
        text=f"Python Agent 岗位 {index}",
        title="AI Agent工程师",
        company="测试公司",
    )


def test_chroma_rebuild_and_query_round_trip(tmp_path: Path) -> None:
    store = ChromaJobVectorStore(tmp_path / "chroma", FakeEmbedding())
    chunks = [_chunk(1), _chunk(2)]

    assert store.rebuild(chunks) == 2
    hits = store.query("Python Agent", n_results=2)

    assert len(hits) == 2
    assert {hit.chunk.chunk_id for hit in hits} == {chunk.chunk_id for chunk in chunks}
    assert all(0 <= hit.final_score <= 1 for hit in hits)
