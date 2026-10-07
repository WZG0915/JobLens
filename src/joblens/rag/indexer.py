"""岗位目录到 Chroma 向量索引的构建流程。"""

from __future__ import annotations

from pathlib import Path

from .embedding import BgeLocalEmbedding, EmbeddingProvider
from .job_chunker import chunk_job_corpus
from .schemas import JobIndexBuildResult
from .vector_store import ChromaJobVectorStore, DEFAULT_COLLECTION_NAME


def build_job_index(
    jobs_directory: str | Path,
    persist_directory: str | Path,
    *,
    embedding: EmbeddingProvider | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    chroma_batch_size: int = 64,
) -> JobIndexBuildResult:
    """解析岗位、结构化分块、生成BGE向量并完整重建Chroma集合。"""
    corpus = chunk_job_corpus(jobs_directory)
    chunks = [chunk for job in corpus.jobs for chunk in job.chunks]
    provider = embedding or BgeLocalEmbedding()
    store = ChromaJobVectorStore(
        persist_directory,
        provider,
        collection_name=collection_name,
    )
    stored_count = store.rebuild(chunks, batch_size=chroma_batch_size)
    return JobIndexBuildResult(
        collection_name=collection_name,
        persist_directory=str(Path(persist_directory).expanduser().resolve()),
        embedding_model=provider.model_name,
        embedding_dimension=provider.dimension,
        job_count=len(corpus.jobs),
        chunk_count=len(chunks),
        stored_chunk_count=stored_count,
        source_hashes={job.job_id: job.source_hash for job in corpus.jobs},
    )
