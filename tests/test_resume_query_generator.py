"""简历多查询生成测试。"""

from __future__ import annotations

from pathlib import Path

from joblens.rag import ResumeQueryType, generate_resume_queries_from_document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RESUME = PROJECT_ROOT / "data" / "resumes" / "sample_resume.md"


def test_resume_multi_query_is_traceable_and_excludes_contact_information() -> None:
    result = generate_resume_queries_from_document(SAMPLE_RESUME)

    assert len(result.queries) >= 4
    assert len({query.query_id for query in result.queries}) == len(result.queries)
    assert ResumeQueryType.SKILLS in {query.query_type for query in result.queries}
    assert ResumeQueryType.PROJECT in {query.query_type for query in result.queries}
    combined = "\n".join(query.text for query in result.queries)
    assert "Python" in combined
    assert "@" not in combined
    assert "138" not in combined
    skill_query = next(query for query in result.queries if query.query_type == ResumeQueryType.SKILLS)
    assert "RAG" not in skill_query.keywords
    assert "Docker" not in skill_query.keywords
    assert ">" not in combined


def test_resume_multi_query_is_deterministic() -> None:
    first = generate_resume_queries_from_document(SAMPLE_RESUME)
    second = generate_resume_queries_from_document(SAMPLE_RESUME)
    assert first == second
