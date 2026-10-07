"""岗位混合召回、证据匹配与最终排序的端到端流程。"""

from __future__ import annotations

import time
from pathlib import Path

from joblens.exceptions import DocumentParseError, RagIndexError
from joblens.parsers import parse_job_document, parse_resume_document
from joblens.schemas.match import EvidenceMatchResult, EvidenceMatchStatus
from joblens.tools import match_resume_job_data

from .embedding import BgeEmbeddingSettings, BgeLocalEmbedding, EmbeddingProvider
from .job_chunker import chunk_job_corpus
from .resume_query_generator import generate_resume_queries
from .retriever import HybridJobRetriever
from .schemas import (
    JobRecommendationResult,
    JobRetrievalCandidate,
    RecommendedJob,
    ResumeMultiQueryResult,
)
from .vector_store import ChromaJobVectorStore, DEFAULT_COLLECTION_NAME


def _strengths(result: EvidenceMatchResult, limit: int = 4) -> list[str]:
    """提取已经有明确简历证据支持的岗位要求。"""
    return [
        item.requirement
        for item in result.matches
        if item.status == EvidenceMatchStatus.MATCHED
    ][:limit]


def _gaps(result: EvidenceMatchResult, limit: int = 4) -> list[str]:
    """提取未满足或证据不足的岗位要求。"""
    priorities = {
        EvidenceMatchStatus.NOT_MATCHED: 0,
        EvidenceMatchStatus.INSUFFICIENT_EVIDENCE: 1,
        EvidenceMatchStatus.PARTIALLY_MATCHED: 2,
        EvidenceMatchStatus.MATCHED: 3,
    }
    values = [item for item in result.matches if item.status != EvidenceMatchStatus.MATCHED]
    values.sort(key=lambda item: (not item.is_hard_constraint, priorities[item.status]))
    return [item.requirement for item in values[:limit]]


def rank_job_recommendations(
    candidates: list[JobRetrievalCandidate],
    evidence_results: dict[str, EvidenceMatchResult],
    *,
    top_k: int = 5,
) -> list[RecommendedJob]:
    """以证据匹配为主、RAG召回为辅，并对硬约束失败进行惩罚。"""
    if top_k < 1:
        raise ValueError("top_k 必须大于 0")
    ranked: list[RecommendedJob] = []
    for candidate in candidates:
        evidence = evidence_results.get(candidate.job_id)
        if evidence is None or not evidence.success or evidence.summary is None:
            continue
        summary = evidence.summary
        base_score = 0.35 * candidate.retrieval_score + 0.65 * (summary.weighted_score / 100)
        # 每失败一个明确硬约束，总分乘以 0.7；最终排序仍将无硬伤岗位放在前面。
        final_score = round(100 * base_score * (0.7 ** summary.hard_constraints_failed), 2)
        ranked.append(
            RecommendedJob(
                rank=1,
                job_id=candidate.job_id,
                job_source=candidate.source_path,
                company=evidence.job_company or candidate.company,
                title=evidence.job_title or candidate.title,
                final_score=final_score,
                retrieval_score=candidate.retrieval_score,
                evidence_score=summary.weighted_score,
                evidence_coverage=summary.evidence_coverage,
                hard_constraints_failed=summary.hard_constraints_failed,
                hard_constraints_unknown=summary.hard_constraints_unknown,
                review_required=summary.review_required,
                matched_skills=candidate.matched_terms,
                strengths=_strengths(evidence),
                gaps=_gaps(evidence),
                recommendations=evidence.recommendations[:5],
                evidence_match=evidence,
            )
        )
    ranked.sort(
        key=lambda item: (
            item.hard_constraints_failed > 0,
            item.hard_constraints_failed,
            -item.final_score,
            -item.evidence_coverage,
            item.job_id,
        )
    )
    return [
        item.model_copy(update={"rank": index})
        for index, item in enumerate(ranked[:top_k], start=1)
    ]


def recommend_jobs_with_rag(
    resume_path: str | Path,
    jobs_directory: str | Path,
    persist_directory: str | Path,
    *,
    embedding: EmbeddingProvider | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    top_k_retrieval: int = 10,
    top_k_final: int = 5,
    top_k_per_query: int = 30,
) -> JobRecommendationResult:
    """执行简历多查询、混合召回、证据匹配和最终岗位排序。"""
    started = time.perf_counter()
    resume_file = Path(resume_path).expanduser().resolve()
    jobs_dir = Path(jobs_directory).expanduser().resolve()
    database_dir = Path(persist_directory).expanduser().resolve()

    resume_result = parse_resume_document(resume_file)
    if not resume_result.success or resume_result.data is None:
        raise DocumentParseError("简历解析失败：" + "；".join(resume_result.errors))
    resume = resume_result.data
    multi_query = generate_resume_queries(resume)

    corpus = chunk_job_corpus(jobs_dir)
    chunks = [chunk for job in corpus.jobs for chunk in job.chunks]
    provider = embedding or BgeLocalEmbedding(
        BgeEmbeddingSettings.from_environment()
    )
    store = ChromaJobVectorStore(
        database_dir,
        provider,
        collection_name=collection_name,
    )
    expected_hashes = {job.job_id: job.source_hash for job in corpus.jobs}
    indexed_hashes = store.indexed_source_hashes()
    if store.count() != len(chunks) or indexed_hashes != expected_hashes:
        raise RagIndexError(
            "岗位源文件与Chroma索引不一致，请先执行 python app.py build-job-index"
        )

    retrieval = HybridJobRetriever(store, chunks).retrieve(
        multi_query,
        top_k_jobs=top_k_retrieval,
        top_k_per_query=top_k_per_query,
    )
    evidence_results: dict[str, EvidenceMatchResult] = {}
    warnings = [warning.message for warning in resume_result.warnings]
    for candidate in retrieval.candidates:
        job_result = parse_job_document(candidate.source_path)
        if not job_result.success or job_result.data is None:
            warnings.append(f"{candidate.job_id} 岗位解析失败，已跳过")
            continue
        match = match_resume_job_data(
            resume,
            job_result.data,
            resume_source=str(resume_file),
            job_source=candidate.source_path,
            mode="local",
        )
        evidence_results[candidate.job_id] = match
        warnings.extend(match.warnings)

    recommendations = rank_job_recommendations(
        retrieval.candidates,
        evidence_results,
        top_k=top_k_final,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    if not recommendations:
        return JobRecommendationResult(
            success=False,
            resume_source=str(resume_file),
            multi_query=multi_query,
            retrieval=retrieval,
            warnings=list(dict.fromkeys(warnings)),
            errors=["RAG候选岗位均未能完成证据匹配"],
            processing_time_ms=elapsed_ms,
        )
    return JobRecommendationResult(
        success=True,
        resume_source=str(resume_file),
        multi_query=multi_query,
        retrieval=retrieval,
        recommendations=recommendations,
        warnings=list(dict.fromkeys(warnings)),
        processing_time_ms=elapsed_ms,
    )
