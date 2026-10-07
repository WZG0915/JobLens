"""BGE向量、BM25和技能精确匹配的混合岗位召回。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from .bm25 import Bm25Index
from .embedding import EmbeddingProvider
from .schemas import (
    JobChunk,
    JobChunkHit,
    JobRagRetrievalResult,
    JobRetrievalCandidate,
    RagRetrievalMode,
    ResumeMultiQueryResult,
    ResumeQueryType,
)
from .vector_store import ChromaJobVectorStore


def _term_key(value: str) -> str:
    """技能精确匹配使用的大小写与符号无关键。"""
    return re.sub(r"[^a-z0-9+#\u4e00-\u9fff]", "", value.casefold())


def _term_present(term: str, chunk: JobChunk) -> bool:
    """优先匹配结构化关键词，再检查块正文中的完整英文词或中文片段。"""
    key = _term_key(term)
    if not key:
        return False
    if key in {_term_key(value) for value in chunk.keywords}:
        return True
    text = chunk.text.casefold()
    if re.fullmatch(r"[a-z0-9+#.\- ]+", term.casefold()):
        return bool(
            re.search(
                rf"(?<![a-z0-9+#]){re.escape(term.casefold())}(?![a-z0-9+#])",
                text,
            )
        )
    return term.casefold() in text


def _skill_match(chunk: JobChunk, skills: Sequence[str]) -> tuple[float, list[str]]:
    """计算简历技能对当前块的精确覆盖，最多三项即可获得满分。"""
    matched = [skill for skill in skills if _term_present(skill, chunk)]
    if not matched:
        return 0.0, []
    chunk_skill_count = len({_term_key(value) for value in chunk.keywords if _term_key(value)})
    denominator = max(1, min(3, chunk_skill_count or len(matched)))
    return min(1.0, len(matched) / denominator), matched


@dataclass
class _AccumulatedHit:
    """同一Chunk在多条查询中的最佳分量与查询覆盖。"""

    chunk: JobChunk
    final_score: float = 0.0
    vector_score: float = 0.0
    bm25_score: float = 0.0
    skill_score: float = 0.0
    query_ids: set[str] = field(default_factory=set)
    matched_terms: set[str] = field(default_factory=set)


class HybridJobRetriever:
    """在同一个候选集合中融合向量、BM25与技能精确匹配。"""

    def __init__(
        self,
        vector_store: ChromaJobVectorStore,
        chunks: Sequence[JobChunk],
        *,
        vector_weight: float = 0.55,
        bm25_weight: float = 0.30,
        skill_weight: float = 0.15,
    ) -> None:
        if not chunks:
            raise ValueError("混合检索器需要至少一个岗位 Chunk")
        if abs(vector_weight + bm25_weight + skill_weight - 1.0) > 1e-9:
            raise ValueError("混合检索权重之和必须等于 1")
        self.vector_store = vector_store
        self.chunks = list(chunks)
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.skill_weight = skill_weight
        self._chunks_by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        if len(self._chunks_by_id) != len(self.chunks):
            raise ValueError("岗位语料中存在重复 chunk_id")
        self._bm25 = Bm25Index([chunk.to_embedding_text() for chunk in self.chunks])

    def retrieve(
        self,
        multi_query: ResumeMultiQueryResult,
        *,
        top_k_jobs: int = 10,
        top_k_per_query: int = 30,
        top_chunks_per_job: int = 5,
    ) -> JobRagRetrievalResult:
        """逐查询召回并按岗位聚合，返回证据匹配前的候选岗位。"""
        if top_k_jobs < 1 or top_k_per_query < 1 or top_chunks_per_job < 1:
            raise ValueError("检索数量参数必须大于 0")
        skill_query = next(
            (query for query in multi_query.queries if query.query_type == ResumeQueryType.SKILLS),
            None,
        )
        resume_skills = skill_query.keywords if skill_query else []
        accumulated: dict[str, _AccumulatedHit] = {}

        for query in multi_query.queries:
            vector_hits = self.vector_store.query(query.text, n_results=top_k_per_query)
            vector_scores = {hit.chunk.chunk_id: hit.final_score for hit in vector_hits}
            bm25_hits = self._bm25.search(query.text, n_results=top_k_per_query)
            bm25_scores = {self.chunks[hit.index].chunk_id: hit.score for hit in bm25_hits}
            candidate_ids = set(vector_scores) | set(bm25_scores)
            for chunk_id in candidate_ids:
                chunk = self._chunks_by_id.get(chunk_id)
                if chunk is None:
                    # 索引中残留旧Chunk时忽略，调用方还会通过哈希检查提示重建。
                    continue
                vector_score = vector_scores.get(chunk_id, 0.0)
                bm25_score = bm25_scores.get(chunk_id, 0.0)
                skill_score, matched_skills = _skill_match(chunk, resume_skills)
                combined = query.weight * (
                    self.vector_weight * vector_score
                    + self.bm25_weight * bm25_score
                    + self.skill_weight * skill_score
                )
                current = accumulated.setdefault(chunk_id, _AccumulatedHit(chunk=chunk))
                current.final_score = max(current.final_score, combined)
                current.vector_score = max(current.vector_score, vector_score)
                current.bm25_score = max(current.bm25_score, bm25_score)
                current.skill_score = max(current.skill_score, skill_score)
                current.query_ids.add(query.query_id)
                current.matched_terms.update(matched_skills)
                current.matched_terms.update(
                    keyword for keyword in query.keywords if _term_present(keyword, chunk)
                )

        jobs: dict[str, list[_AccumulatedHit]] = {}
        for hit in accumulated.values():
            jobs.setdefault(hit.chunk.job_id, []).append(hit)

        candidates: list[JobRetrievalCandidate] = []
        for job_id, hits in jobs.items():
            ordered = sorted(hits, key=lambda item: (-item.final_score, item.chunk.chunk_id))
            selected = ordered[:top_chunks_per_job]
            best = selected[0].final_score
            average = sum(item.final_score for item in selected[:3]) / min(3, len(selected))
            query_ids = set().union(*(item.query_ids for item in hits))
            coverage = len(query_ids) / len(multi_query.queries)
            retrieval_score = min(1.0, 0.55 * best + 0.35 * average + 0.10 * coverage)
            chunk_hits = [
                JobChunkHit(
                    rank=index,
                    chunk=item.chunk,
                    final_score=item.final_score,
                    vector_score=item.vector_score,
                    keyword_score=min(1.0, 0.67 * item.bm25_score + 0.33 * item.skill_score),
                    bm25_score=item.bm25_score,
                    skill_score=item.skill_score,
                    query_ids=sorted(item.query_ids),
                    matched_terms=sorted(item.matched_terms),
                )
                for index, item in enumerate(selected, start=1)
            ]
            first = selected[0].chunk
            candidates.append(
                JobRetrievalCandidate(
                    rank=1,
                    job_id=job_id,
                    source_path=first.source_path,
                    title=first.title,
                    company=first.company,
                    retrieval_score=retrieval_score,
                    vector_score=max(item.vector_score for item in selected),
                    bm25_score=max(item.bm25_score for item in selected),
                    skill_score=max(item.skill_score for item in selected),
                    query_coverage=coverage,
                    matched_terms=sorted(set().union(*(item.matched_terms for item in hits))),
                    chunk_hits=chunk_hits,
                )
            )

        candidates.sort(
            key=lambda item: (-item.retrieval_score, -item.skill_score, item.job_id)
        )
        candidates = [
            candidate.model_copy(update={"rank": index})
            for index, candidate in enumerate(candidates[:top_k_jobs], start=1)
        ]
        return JobRagRetrievalResult(
            query="\n".join(query.text for query in multi_query.queries),
            mode=RagRetrievalMode.HYBRID,
            top_k_jobs=top_k_jobs,
            total_jobs_scanned=len({chunk.job_id for chunk in self.chunks}),
            total_chunks_scanned=len(self.chunks),
            candidates=candidates,
        )
