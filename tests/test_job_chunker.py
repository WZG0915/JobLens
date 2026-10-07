"""岗位 RAG 数据契约与结构化分块测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from joblens.rag import (
    JobChunk,
    JobChunkHit,
    JobChunkSection,
    JobRagRetrievalResult,
    JobRetrievalCandidate,
    RagRetrievalMode,
    chunk_job_corpus,
    chunk_job_document,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JOBS_DIRECTORY = PROJECT_ROOT / "data" / "jobs"
SAMPLE_JOB = JOBS_DIRECTORY / "11_synthetic_agent_application_intern.md"


def test_one_job_is_split_by_business_section() -> None:
    """每条职责和每条要求应成为独立、可追溯的块。"""
    result = chunk_job_document(SAMPLE_JOB)
    counts = result.section_counts()

    assert result.job_id == SAMPLE_JOB.stem
    assert counts[JobChunkSection.BASIC] == 1
    assert counts[JobChunkSection.RESPONSIBILITY] >= 4
    assert counts[JobChunkSection.REQUIREMENT] >= 5
    assert counts[JobChunkSection.AVAILABILITY] == 1
    assert len({chunk.chunk_id for chunk in result.chunks}) == len(result.chunks)
    assert all(chunk.text.strip() for chunk in result.chunks)

    requirement = next(
        chunk for chunk in result.chunks if chunk.section == JobChunkSection.REQUIREMENT
    )
    assert requirement.requirement_id
    assert requirement.requirement_category is not None
    assert requirement.requirement_importance is not None
    assert requirement.source_references
    embedding_text = requirement.to_embedding_text()
    assert requirement.text in embedding_text
    assert "岗位：" in embedding_text


def test_chunking_is_deterministic() -> None:
    """同一文件重复构建时，ID、哈希和正文都必须稳定。"""
    first = chunk_job_document(SAMPLE_JOB)
    second = chunk_job_document(SAMPLE_JOB)

    assert first.source_hash == second.source_hash
    assert [chunk.chunk_id for chunk in first.chunks] == [
        chunk.chunk_id for chunk in second.chunks
    ]
    assert [chunk.text for chunk in first.chunks] == [chunk.text for chunk in second.chunks]


def test_all_local_jobs_can_be_chunked() -> None:
    """当前三十份本地岗位都应成功进入结构化岗位语料库。"""
    corpus = chunk_job_corpus(JOBS_DIRECTORY)

    assert len(corpus.jobs) == 30
    assert corpus.chunk_count > 30
    assert len({job.job_id for job in corpus.jobs}) == 30
    assert all(job.chunks for job in corpus.jobs)
    assert all(chunk.source_references for job in corpus.jobs for chunk in job.chunks if chunk.section in {JobChunkSection.RESPONSIBILITY, JobChunkSection.REQUIREMENT})


def test_chunk_model_rejects_inconsistent_section_metadata() -> None:
    """要求 ID 不能被错误地放到职责块中。"""
    result = chunk_job_document(SAMPLE_JOB)
    responsibility = next(
        chunk for chunk in result.chunks if chunk.section == JobChunkSection.RESPONSIBILITY
    )
    invalid_data = responsibility.model_dump()
    invalid_data["requirement_id"] = "req_999"

    with pytest.raises(ValidationError):
        JobChunk.model_validate(invalid_data)


def test_rag_result_contract_checks_candidate_ranking() -> None:
    """RAG 结果的岗位排名必须连续，命中块也必须属于该岗位。"""
    chunk = chunk_job_document(SAMPLE_JOB).chunks[0]
    hit = JobChunkHit(rank=1, chunk=chunk, final_score=0.91, matched_terms=["Agent"])
    candidate = JobRetrievalCandidate(
        rank=1,
        job_id=chunk.job_id,
        source_path=chunk.source_path,
        title=chunk.title,
        company=chunk.company,
        retrieval_score=0.91,
        matched_terms=["Agent"],
        chunk_hits=[hit],
    )
    result = JobRagRetrievalResult(
        query="Python Agent RAG",
        mode=RagRetrievalMode.HYBRID,
        top_k_jobs=5,
        total_jobs_scanned=30,
        total_chunks_scanned=200,
        candidates=[candidate],
    )
    assert result.candidates[0].chunk_hits[0].chunk.chunk_id == chunk.chunk_id

    invalid_data = result.model_dump()
    invalid_data["candidates"][0]["rank"] = 2
    with pytest.raises(ValidationError):
        JobRagRetrievalResult.model_validate(invalid_data)
