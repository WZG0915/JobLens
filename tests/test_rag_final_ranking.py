"""证据匹配接入与最终岗位排序测试。"""

from __future__ import annotations

from pathlib import Path

from joblens.rag import JobChunk, JobChunkHit, JobChunkSection, JobRetrievalCandidate
from joblens.rag.recommender import rank_job_recommendations
from joblens.schemas.match import EvidenceMatchResult, EvidenceMatchSummary


def _candidate(job_id: str, retrieval_score: float) -> JobRetrievalCandidate:
    chunk = JobChunk(
        chunk_id=f"{job_id}:basic",
        job_id=job_id,
        source_path=str(Path(f"{job_id}.md").resolve()),
        source_hash=("a" if job_id == "safe" else "b") * 64,
        section=JobChunkSection.BASIC,
        text="Python Agent岗位",
        title=job_id,
    )
    return JobRetrievalCandidate(
        rank=1,
        job_id=job_id,
        source_path=chunk.source_path,
        title=job_id,
        retrieval_score=retrieval_score,
        chunk_hits=[JobChunkHit(rank=1, chunk=chunk, final_score=retrieval_score)],
    )


def _evidence(job_id: str, score: float, hard_failed: int) -> EvidenceMatchResult:
    return EvidenceMatchResult(
        success=True,
        job_source=str(Path(f"{job_id}.md").resolve()),
        job_company="测试公司",
        job_title=job_id,
        summary=EvidenceMatchSummary(
            total_requirements=1,
            matched=1,
            partially_matched=0,
            insufficient_evidence=0,
            not_matched=0,
            hard_constraints_total=hard_failed,
            hard_constraints_passed=0,
            hard_constraints_failed=hard_failed,
            hard_constraints_unknown=0,
            evidence_coverage=1,
            weighted_score=score,
            review_required=0,
        ),
    )


def test_final_ranking_prioritizes_no_failed_hard_constraints() -> None:
    candidates = [_candidate("risky", 0.99), _candidate("safe", 0.70)]
    evidence = {
        "risky": _evidence("risky", 95, 1),
        "safe": _evidence("safe", 75, 0),
    }
    ranked = rank_job_recommendations(candidates, evidence, top_k=2)
    assert ranked[0].job_id == "safe"
    assert ranked[0].rank == 1
    assert ranked[1].hard_constraints_failed == 1
