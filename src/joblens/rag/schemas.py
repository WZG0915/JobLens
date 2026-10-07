"""岗位 RAG 阶段使用的数据契约。

这一层只描述“分块后是什么样”和“检索结果是什么样”，不依赖具体的
向量数据库。后续无论接入 Chroma、FAISS 还是其他数据库，都应返回这里
定义的模型，避免上层 Agent 被某一种数据库的返回格式绑死。
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from joblens.schemas.common import SourceReference, StrictBaseModel
from joblens.schemas.job import RequirementCategory, RequirementImportance
from joblens.schemas.match import EvidenceMatchResult


class JobChunkSection(str, Enum):
    """岗位块在结构化 JD 中所属的章节。"""

    BASIC = "basic"
    SUMMARY = "summary"
    RESPONSIBILITY = "responsibility"
    REQUIREMENT = "requirement"
    AVAILABILITY = "availability"
    BENEFIT = "benefit"
    OTHER = "other"


class JobChunk(StrictBaseModel):
    """可写入向量数据库的最小岗位检索单元。

    ``text`` 保留该条职责或要求的原始语义；标题、公司和结构化标签单独
    保存，既便于过滤，也避免后续只能从拼接文本中反向猜测元数据。
    """

    chunk_id: str = Field(min_length=1, description="全库唯一、可重复生成的块 ID")
    job_id: str = Field(min_length=1, description="所属岗位的稳定 ID")
    source_path: str = Field(min_length=1, description="原始 JD 文件路径")
    source_hash: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="标准化 JD 原文的 SHA-256，用于判断数据是否变化",
    )
    section: JobChunkSection
    text: str = Field(min_length=1, description="用于检索和展示的块正文")
    title: str = ""
    company: str | None = None
    keywords: list[str] = Field(default_factory=list)
    source_references: list[SourceReference] = Field(default_factory=list)

    # 职责与要求的原始 ID 不能混用；它们是后续证据匹配的稳定连接键。
    responsibility_id: str | None = None
    requirement_id: str | None = None
    requirement_category: RequirementCategory | None = None
    requirement_importance: RequirementImportance | None = None
    is_hard_constraint: bool = False

    @model_validator(mode="after")
    def validate_section_metadata(self) -> "JobChunk":
        """保证章节类型与专属元数据一致，尽早发现构造错误。"""
        if self.section == JobChunkSection.RESPONSIBILITY:
            if not self.responsibility_id:
                raise ValueError("职责块必须提供 responsibility_id")
            if self.requirement_id is not None:
                raise ValueError("职责块不能提供 requirement_id")
        elif self.responsibility_id is not None:
            raise ValueError("只有职责块可以提供 responsibility_id")

        if self.section == JobChunkSection.REQUIREMENT:
            if not self.requirement_id:
                raise ValueError("要求块必须提供 requirement_id")
            if self.requirement_category is None or self.requirement_importance is None:
                raise ValueError("要求块必须提供类别和重要程度")
        elif any(
            value is not None
            for value in (
                self.requirement_id,
                self.requirement_category,
                self.requirement_importance,
            )
        ):
            raise ValueError("只有要求块可以提供要求专属元数据")
        elif self.is_hard_constraint:
            raise ValueError("只有要求块可以标记为硬约束")
        return self

    def to_embedding_text(self) -> str:
        """生成送给 Embedding 模型的自描述文本。

        数据库中仍保存原始 ``text``；这里只在向量化时补充岗位上下文，
        避免“负责模型评测”一类短句脱离岗位标题后语义过弱。
        """
        parts = [f"岗位：{self.title or '未命名岗位'}"]
        if self.company:
            parts.append(f"公司：{self.company}")
        parts.append(f"章节：{self.section.value}")
        if self.keywords:
            parts.append(f"关键词：{'、'.join(self.keywords)}")
        parts.append(f"内容：{self.text}")
        return "\n".join(parts)


class JobChunkingResult(StrictBaseModel):
    """一份岗位 JD 的结构化分块结果。"""

    job_id: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunks: list[JobChunk] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_chunks(self) -> "JobChunkingResult":
        """一份结果中的块必须同源且 ID 不重复。"""
        ids = [chunk.chunk_id for chunk in self.chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("同一岗位中出现重复 chunk_id")
        for chunk in self.chunks:
            if chunk.job_id != self.job_id:
                raise ValueError("chunk.job_id 与分块结果 job_id 不一致")
            if chunk.source_path != self.source_path:
                raise ValueError("chunk.source_path 与分块结果来源不一致")
            if chunk.source_hash != self.source_hash:
                raise ValueError("chunk.source_hash 与分块结果哈希不一致")
        return self

    def section_counts(self) -> dict[JobChunkSection, int]:
        """按章节统计块数量，便于测试和构建索引日志。"""
        counts: dict[JobChunkSection, int] = {}
        for chunk in self.chunks:
            counts[chunk.section] = counts.get(chunk.section, 0) + 1
        return counts


class JobCorpusChunkingResult(StrictBaseModel):
    """整个本地岗位目录的分块结果。"""

    source_directory: str = Field(min_length=1)
    jobs: list[JobChunkingResult] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_jobs(self) -> "JobCorpusChunkingResult":
        """岗位 ID 在一个语料库中必须唯一。"""
        job_ids = [job.job_id for job in self.jobs]
        if len(job_ids) != len(set(job_ids)):
            raise ValueError("岗位语料库中出现重复 job_id")
        return self

    @property
    def chunk_count(self) -> int:
        """返回整个岗位库的块总数。"""
        return sum(len(job.chunks) for job in self.jobs)


class RagRetrievalMode(str, Enum):
    """RAG 检索模式，为后续替换检索器预留统一标识。"""

    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class JobChunkHit(StrictBaseModel):
    """一次查询命中的岗位块及其评分。"""

    rank: int = Field(ge=1)
    chunk: JobChunk
    final_score: float = Field(ge=0, le=1)
    vector_score: float | None = Field(default=None, ge=0, le=1)
    keyword_score: float | None = Field(default=None, ge=0, le=1)
    bm25_score: float | None = Field(default=None, ge=0, le=1)
    skill_score: float | None = Field(default=None, ge=0, le=1)
    query_ids: list[str] = Field(default_factory=list)
    matched_terms: list[str] = Field(default_factory=list)


class JobRetrievalCandidate(StrictBaseModel):
    """把同一岗位的若干命中块聚合后的候选岗位。"""

    rank: int = Field(ge=1)
    job_id: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    title: str = ""
    company: str | None = None
    retrieval_score: float = Field(ge=0, le=1)
    vector_score: float = Field(default=0, ge=0, le=1)
    bm25_score: float = Field(default=0, ge=0, le=1)
    skill_score: float = Field(default=0, ge=0, le=1)
    query_coverage: float = Field(default=0, ge=0, le=1)
    matched_terms: list[str] = Field(default_factory=list)
    chunk_hits: list[JobChunkHit] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_chunk_hits(self) -> "JobRetrievalCandidate":
        """候选岗位只能聚合自己的块，并要求块排名连续。"""
        expected_ranks = list(range(1, len(self.chunk_hits) + 1))
        actual_ranks = [hit.rank for hit in self.chunk_hits]
        if actual_ranks != expected_ranks:
            raise ValueError("候选岗位的 chunk_hits 排名必须从 1 连续递增")
        if any(hit.chunk.job_id != self.job_id for hit in self.chunk_hits):
            raise ValueError("候选岗位中包含其他岗位的 chunk")
        return self


class JobRagRetrievalResult(StrictBaseModel):
    """岗位 RAG 检索器向 Agent 返回的统一结果。"""

    query: str = Field(min_length=1)
    mode: RagRetrievalMode
    top_k_jobs: int = Field(default=5, ge=1)
    total_jobs_scanned: int = Field(default=0, ge=0)
    total_chunks_scanned: int = Field(default=0, ge=0)
    candidates: list[JobRetrievalCandidate] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_candidates(self) -> "JobRagRetrievalResult":
        """候选岗位必须去重、连续排名，并且不超过 top_k。"""
        if len(self.candidates) > self.top_k_jobs:
            raise ValueError("候选岗位数量不能超过 top_k_jobs")
        ranks = [candidate.rank for candidate in self.candidates]
        if ranks != list(range(1, len(self.candidates) + 1)):
            raise ValueError("候选岗位排名必须从 1 连续递增")
        job_ids = [candidate.job_id for candidate in self.candidates]
        if len(job_ids) != len(set(job_ids)):
            raise ValueError("RAG 结果中出现重复岗位")
        return self


class JobIndexBuildResult(StrictBaseModel):
    """本地岗位向量索引的一次构建结果。"""

    collection_name: str = Field(min_length=3)
    persist_directory: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    embedding_dimension: int = Field(ge=1)
    job_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    stored_chunk_count: int = Field(ge=0)
    source_hashes: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_counts(self) -> "JobIndexBuildResult":
        """完整重建后，写入数量必须与待索引块数量一致。"""
        if self.stored_chunk_count != self.chunk_count:
            raise ValueError("Chroma 中的块数量与本次构建数量不一致")
        if len(self.source_hashes) != self.job_count:
            raise ValueError("source_hashes 数量与岗位数量不一致")
        return self


class ResumeQueryType(str, Enum):
    """简历多查询的语义视角。"""

    TARGET_ROLE = "target_role"
    SUMMARY = "summary"
    SKILLS = "skills"
    PROJECT = "project"
    WORK_EXPERIENCE = "work_experience"
    EDUCATION = "education"
    AVAILABILITY = "availability"
    FALLBACK = "fallback"


class ResumeSearchQuery(StrictBaseModel):
    """由简历某个证据视角产生的一条岗位检索查询。"""

    query_id: str = Field(min_length=1)
    query_type: ResumeQueryType
    text: str = Field(min_length=1)
    keywords: list[str] = Field(default_factory=list)
    weight: float = Field(default=1.0, gt=0, le=1)
    source_references: list[SourceReference] = Field(default_factory=list)


class ResumeMultiQueryResult(StrictBaseModel):
    """一份简历生成的、可解释且不含联系方式的多查询集合。"""

    resume_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    queries: list[ResumeSearchQuery] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_queries(self) -> "ResumeMultiQueryResult":
        """查询 ID 与查询正文都应去重。"""
        ids = [query.query_id for query in self.queries]
        texts = [query.text.casefold() for query in self.queries]
        if len(ids) != len(set(ids)):
            raise ValueError("简历多查询中出现重复 query_id")
        if len(texts) != len(set(texts)):
            raise ValueError("简历多查询中出现重复正文")
        return self


class RecommendedJob(StrictBaseModel):
    """经过混合召回和逐要求证据匹配后的最终岗位。"""

    rank: int = Field(ge=1)
    job_id: str = Field(min_length=1)
    job_source: str = Field(min_length=1)
    company: str | None = None
    title: str = ""
    final_score: float = Field(ge=0, le=100)
    retrieval_score: float = Field(ge=0, le=1)
    evidence_score: float = Field(ge=0, le=100)
    evidence_coverage: float = Field(ge=0, le=1)
    hard_constraints_failed: int = Field(ge=0)
    hard_constraints_unknown: int = Field(ge=0)
    review_required: int = Field(ge=0)
    matched_skills: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    evidence_match: EvidenceMatchResult


class JobRecommendationResult(StrictBaseModel):
    """RAG召回、证据匹配与最终排序的完整可追溯结果。"""

    success: bool
    resume_source: str = Field(min_length=1)
    multi_query: ResumeMultiQueryResult
    retrieval: JobRagRetrievalResult
    recommendations: list[RecommendedJob] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    processing_time_ms: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_result(self) -> "JobRecommendationResult":
        """成功结果应有推荐，失败结果应说明错误，岗位排名必须连续。"""
        if self.success and not self.recommendations:
            raise ValueError("推荐成功时 recommendations 不能为空")
        if not self.success and not self.errors:
            raise ValueError("推荐失败时 errors 不能为空")
        ranks = [item.rank for item in self.recommendations]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("最终岗位排名必须从 1 连续递增")
        return self
