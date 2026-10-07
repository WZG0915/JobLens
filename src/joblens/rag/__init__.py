"""JobLens 的岗位 RAG 数据模型与结构化分块入口。"""

from .embedding import (
    DEFAULT_BGE_MODEL,
    BgeEmbeddingSettings,
    BgeLocalEmbedding,
    EmbeddingProvider,
)
from .indexer import build_job_index
from .job_chunker import chunk_job, chunk_job_corpus, chunk_job_document
from .resume_query_generator import (
    generate_resume_queries,
    generate_resume_queries_from_document,
)
from .bm25 import Bm25Hit, Bm25Index, tokenize_for_bm25
from .recommender import rank_job_recommendations, recommend_jobs_with_rag
from .retriever import HybridJobRetriever
from .schemas import (
    JobChunk,
    JobChunkHit,
    JobChunkingResult,
    JobChunkSection,
    JobCorpusChunkingResult,
    JobIndexBuildResult,
    JobRecommendationResult,
    JobRagRetrievalResult,
    JobRetrievalCandidate,
    RagRetrievalMode,
    ResumeMultiQueryResult,
    ResumeQueryType,
    ResumeSearchQuery,
    RecommendedJob,
)
from .vector_store import ChromaJobVectorStore, DEFAULT_COLLECTION_NAME

__all__ = [
    "BgeEmbeddingSettings",
    "BgeLocalEmbedding",
    "ChromaJobVectorStore",
    "DEFAULT_BGE_MODEL",
    "DEFAULT_COLLECTION_NAME",
    "EmbeddingProvider",
    "Bm25Hit",
    "Bm25Index",
    "HybridJobRetriever",
    "JobChunk",
    "JobChunkHit",
    "JobChunkingResult",
    "JobChunkSection",
    "JobCorpusChunkingResult",
    "JobIndexBuildResult",
    "JobRecommendationResult",
    "JobRagRetrievalResult",
    "JobRetrievalCandidate",
    "RagRetrievalMode",
    "ResumeMultiQueryResult",
    "ResumeQueryType",
    "ResumeSearchQuery",
    "RecommendedJob",
    "build_job_index",
    "chunk_job",
    "chunk_job_corpus",
    "chunk_job_document",
    "generate_resume_queries",
    "generate_resume_queries_from_document",
    "rank_job_recommendations",
    "recommend_jobs_with_rag",
    "tokenize_for_bm25",
]
