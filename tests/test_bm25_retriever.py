"""BM25、技能精确匹配和岗位聚合测试。"""

from __future__ import annotations

from pathlib import Path

from joblens.rag import (
    Bm25Index,
    HybridJobRetriever,
    JobChunk,
    JobChunkHit,
    JobChunkSection,
    ResumeMultiQueryResult,
    ResumeQueryType,
    ResumeSearchQuery,
)
from joblens.schemas.job import RequirementCategory, RequirementImportance


def _requirement_chunk(job_id: str, text: str, keywords: list[str]) -> JobChunk:
    return JobChunk(
        chunk_id=f"{job_id}:requirement:req_001",
        job_id=job_id,
        source_path=str(Path(f"{job_id}.md").resolve()),
        source_hash=("a" if job_id == "job_python" else "b") * 64,
        section=JobChunkSection.REQUIREMENT,
        text=text,
        title="AI Agent工程师" if job_id == "job_python" else "Java后端工程师",
        company="测试公司",
        keywords=keywords,
        requirement_id="req_001",
        requirement_category=RequirementCategory.AI_ML,
        requirement_importance=RequirementImportance.REQUIRED,
    )


class FakeVectorStore:
    """根据查询词返回稳定的向量命中。"""

    def __init__(self, chunks: list[JobChunk]) -> None:
        self.chunks = chunks

    def query(self, text: str, *, n_results: int):
        ordered = self.chunks if "Agent" in text or "Python" in text else list(reversed(self.chunks))
        return [
            JobChunkHit(rank=index, chunk=chunk, final_score=0.9 - index * 0.1, vector_score=0.9 - index * 0.1)
            for index, chunk in enumerate(ordered[:n_results], start=1)
        ]


def test_bm25_prefers_exact_python_agent_document() -> None:
    index = Bm25Index(["Python AI Agent RAG 工具调用", "Java Spring 微服务开发"])
    hits = index.search("Python Agent 工具调用", n_results=2)
    assert hits[0].index == 0
    assert hits[0].score == 1.0


def test_hybrid_retrieval_aggregates_queries_by_job() -> None:
    chunks = [
        _requirement_chunk("job_python", "熟悉 Python、Agent 和 RAG 工具调用", ["Python", "Agent", "RAG"]),
        _requirement_chunk("job_java", "熟悉 Java、Spring Boot 和微服务", ["Java", "Spring Boot"]),
    ]
    multi_query = ResumeMultiQueryResult(
        resume_hash="c" * 64,
        queries=[
            ResumeSearchQuery(
                query_id="query_001",
                query_type=ResumeQueryType.SKILLS,
                text="核心技能：Python、Agent、RAG",
                keywords=["Python", "Agent", "RAG"],
                weight=1,
            ),
            ResumeSearchQuery(
                query_id="query_002",
                query_type=ResumeQueryType.PROJECT,
                text="使用Python开发Agent工具调用项目",
                keywords=["Python", "Agent"],
                weight=0.95,
            ),
        ],
    )
    result = HybridJobRetriever(FakeVectorStore(chunks), chunks).retrieve(
        multi_query,
        top_k_jobs=2,
        top_k_per_query=2,
    )
    assert result.candidates[0].job_id == "job_python"
    assert result.candidates[0].bm25_score > 0
    assert result.candidates[0].skill_score == 1
    assert result.candidates[0].query_coverage == 1
    assert {"Python", "Agent", "RAG"}.issubset(result.candidates[0].matched_terms)
