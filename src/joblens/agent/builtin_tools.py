"""JobLens 只读、模型可调用的内置工具。

本文件是“适配层”，负责把已有业务函数包装成统一工具：

``模型参数 -> 路径安全检查 -> 调用原业务函数 -> 精简输出 -> 返回模型``

这里不重复实现简历解析、岗位解析或证据匹配算法。真正的算法仍位于
``joblens.parsers`` 和 ``joblens.tools``，这样命令行与 Agent 可以复用同一套逻辑。
"""

from __future__ import annotations

from dataclasses import replace

from pydantic import Field

from joblens.exceptions import DocumentParseError
from joblens.parsers import parse_job_document, parse_resume_document
from joblens.rag import (
    BgeEmbeddingSettings,
    BgeLocalEmbedding,
    recommend_jobs_with_rag,
)
from joblens.tools import match_resume_job_files

from .contracts import (
    ModelTool,
    ToolContext,
    ToolContractModel,
    ToolSafetyPolicy,
)


# 文件类型白名单。扩展名统一使用小写，并包含开头的点。
PDF_EXTENSIONS = {".pdf"}
JOB_EXTENSIONS = {".md", ".markdown", ".txt"}
# 所有文件型工具都把单个输入文件限制在 25 MB 以内。
MAX_INPUT_BYTES = 25 * 1024 * 1024


class ParseResumeInput(ToolContractModel):
    """简历解析工具的输入；模型只需要提供一个文件路径。"""

    file_path: str = Field(
        min_length=1,
        description="项目允许目录内的 PDF 简历路径；可使用项目相对路径或绝对路径。",
    )


class ParseResumeOutput(ToolContractModel):
    """适合返回给模型的简历摘要。

    原始解析结果非常详细，包含联系方式、来源位置和完整原文。Agent 通常不需要
    一次读取全部数据，因此这里保留求职分析所需字段，并主动省略电话、邮箱和 raw_text。
    """

    source_path: str
    candidate_name: str | None = None
    target_roles: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    work_experiences: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    warning_codes: list[str] = Field(default_factory=list)
    extraction_method: str | None = None
    page_count: int | None = None
    processing_time_ms: float | None = None


class ParseJobInput(ToolContractModel):
    """岗位解析工具的输入。"""

    file_path: str = Field(
        min_length=1,
        description="项目允许目录内的 Markdown 或 TXT 岗位文件路径。",
    )


class RequirementSummary(ToolContractModel):
    """一条岗位要求的模型友好摘要。"""

    requirement_id: str
    description: str
    category: str
    importance: str
    is_hard_constraint: bool


class ParseJobOutput(ToolContractModel):
    """岗位解析工具返回给模型的精简结果。"""

    source_path: str
    company: str | None = None
    title: str
    cities: list[str] = Field(default_factory=list)
    employment_type: str
    experience_level: str
    responsibilities: list[str] = Field(default_factory=list)
    requirements: list[RequirementSummary] = Field(default_factory=list)
    warning_codes: list[str] = Field(default_factory=list)
    source_url: str | None = None
    processing_time_ms: float | None = None


class MatchEvidenceInput(ToolContractModel):
    """证据匹配工具需要同时指定简历和岗位文件。"""

    resume_path: str = Field(
        min_length=1,
        description="项目允许目录内的 PDF 简历路径。",
    )
    job_path: str = Field(
        min_length=1,
        description="项目允许目录内的 Markdown 或 TXT 岗位文件路径。",
    )


class RequirementMatchSummary(ToolContractModel):
    """一条岗位要求与简历证据之间的匹配结论。"""

    requirement_id: str
    requirement: str
    status: str
    is_hard_constraint: bool
    confidence: float
    evidence: list[str] = Field(default_factory=list)
    explanation: str
    needs_human_review: bool


class MatchEvidenceOutput(ToolContractModel):
    """整份简历和岗位的证据匹配摘要。"""

    resume_source: str
    job_source: str
    job_company: str | None = None
    job_title: str | None = None
    weighted_score: float
    evidence_coverage: float
    matched: int
    partially_matched: int
    insufficient_evidence: int
    not_matched: int
    hard_constraints_failed: int
    review_required: int
    requirement_matches: list[RequirementMatchSummary] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    processing_time_ms: float | None = None


class RecommendJobsInput(ToolContractModel):
    """RAG岗位推荐工具输入；岗位库和索引目录由服务端固定。"""

    resume_path: str = Field(
        min_length=1,
        description="项目允许目录内的PDF简历路径。",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="最终返回岗位数量，范围1到10。",
    )


class RecommendedJobSummary(ToolContractModel):
    """提供给Agent的单个岗位推荐摘要，不返回整份简历原文。"""

    rank: int
    job_path: str
    company: str | None = None
    title: str
    final_score: float
    retrieval_score: float
    evidence_score: float
    evidence_coverage: float
    hard_constraints_failed: int
    hard_constraints_unknown: int
    matched_skills: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class RecommendJobsOutput(ToolContractModel):
    """RAG岗位推荐工具的模型友好输出。"""

    resume_source: str
    queries: list[str] = Field(default_factory=list)
    jobs_scanned: int
    chunks_scanned: int
    recommendations: list[RecommendedJobSummary] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    processing_time_ms: float | None = None


def _short(value: str | None, limit: int = 500) -> str:
    """限制返回文本长度，避免一次工具调用挤占过多模型上下文。"""
    text = (value or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _date_text(item: object) -> str:
    """从教育、工作或项目条目中读取原始日期文本。"""
    date_range = getattr(item, "date_range", None)
    if date_range is None:
        return ""
    return getattr(date_range, "raw_text", "") or ""


def _parse_resume(arguments: ParseResumeInput, context: ToolContext) -> ParseResumeOutput:
    """执行 PDF 简历解析，并把富结构结果压缩成模型需要的摘要。"""

    # 先做路径、目录、扩展名和大小检查，再调用真正的 PDF 解析器。
    path = context.resolve_input_file(
        arguments.file_path,
        allowed_extensions=PDF_EXTENSIONS,
        max_file_bytes=MAX_INPUT_BYTES,
    )
    result = parse_resume_document(path)
    # 底层解析器使用 success/errors 表示可预期失败；这里转成领域异常，
    # 最终由注册中心包装为统一的 ToolExecutionResult。
    if not result.success or result.data is None:
        raise DocumentParseError("简历解析失败：" + "；".join(result.errors))
    data = result.data
    preference = data.job_preference
    # 对教育、工作、项目生成简短的一行摘要，避免把完整嵌套结构全部交给模型。
    education = [
        _short(
            " | ".join(
                filter(None, [item.school, item.major, item.degree.value, _date_text(item)])
            )
        )
        for item in data.education
    ]
    work = [
        _short(
            " | ".join(filter(None, [item.organization, item.position, _date_text(item)]))
        )
        for item in data.work_experiences
    ]
    projects = [
        _short(" | ".join(filter(None, [item.name, item.role, _date_text(item)])))
        for item in data.projects
    ]
    return ParseResumeOutput(
        source_path=str(path),
        candidate_name=data.basic_information.name,
        target_roles=list(preference.target_roles) if preference else [],
        education=education,
        # 列表设置上限，避免异常文档产生过大的工具结果。
        skills=[item.name for item in data.skills[:100]],
        work_experiences=work[:30],
        projects=projects[:30],
        warning_codes=[warning.code for warning in result.warnings],
        extraction_method=result.metadata.extraction_method,
        page_count=result.metadata.page_count,
        processing_time_ms=result.metadata.processing_time_ms,
    )


def _parse_job(arguments: ParseJobInput, context: ToolContext) -> ParseJobOutput:
    """执行岗位解析，返回职责及可独立匹配的原子化要求。"""

    path = context.resolve_input_file(
        arguments.file_path,
        allowed_extensions=JOB_EXTENSIONS,
        max_file_bytes=MAX_INPUT_BYTES,
    )
    result = parse_job_document(path)
    if not result.success or result.data is None:
        raise DocumentParseError("岗位解析失败：" + "；".join(result.errors))
    data = result.data
    return ParseJobOutput(
        source_path=str(path),
        company=data.company.name,
        title=data.basic_information.title,
        cities=data.basic_information.cities,
        employment_type=data.basic_information.employment_type.value,
        experience_level=data.basic_information.experience_level.value,
        responsibilities=[_short(item.description) for item in data.responsibilities[:50]],
        requirements=[
            # 枚举不能直接放入普通 JSON，因此取 .value 转成稳定字符串。
            RequirementSummary(
                requirement_id=item.id,
                description=_short(item.description),
                category=item.category.value,
                importance=item.importance.value,
                is_hard_constraint=item.is_hard_constraint,
            )
            for item in data.requirements[:100]
        ],
        warning_codes=[warning.code for warning in result.warnings],
        source_url=data.basic_information.source_url,
        processing_time_ms=result.metadata.processing_time_ms,
    )


def _match_evidence(arguments: MatchEvidenceInput, context: ToolContext) -> MatchEvidenceOutput:
    """解析两个文件并在本地完成逐条岗位要求的证据匹配。"""

    # 两个输入分别使用各自的扩展名白名单，不能用岗位文本冒充 PDF 简历。
    resume_path = context.resolve_input_file(
        arguments.resume_path,
        allowed_extensions=PDF_EXTENSIONS,
        max_file_bytes=MAX_INPUT_BYTES,
    )
    job_path = context.resolve_input_file(
        arguments.job_path,
        allowed_extensions=JOB_EXTENSIONS,
        max_file_bytes=MAX_INPUT_BYTES,
    )
    # Agent 工具固定使用 local：不会将简历个人信息发送给外部模型。
    result = match_resume_job_files(resume_path, job_path, mode="local")
    if not result.success or result.summary is None:
        raise DocumentParseError("证据匹配失败：" + "；".join(result.errors))
    summary = result.summary
    # 返回总分的同时保留逐条证据，避免模型只能看到一个无法解释的分数。
    return MatchEvidenceOutput(
        resume_source=str(resume_path),
        job_source=str(job_path),
        job_company=result.job_company,
        job_title=result.job_title,
        weighted_score=summary.weighted_score,
        evidence_coverage=summary.evidence_coverage,
        matched=summary.matched,
        partially_matched=summary.partially_matched,
        insufficient_evidence=summary.insufficient_evidence,
        not_matched=summary.not_matched,
        hard_constraints_failed=summary.hard_constraints_failed,
        review_required=summary.review_required,
        requirement_matches=[
            RequirementMatchSummary(
                requirement_id=item.requirement_id,
                requirement=_short(item.requirement),
                status=item.status.value,
                is_hard_constraint=item.is_hard_constraint,
                confidence=item.confidence,
                # 每条要求最多给模型三条简历证据，每条最多 300 字。
                evidence=[_short(evidence.text, 300) for evidence in item.evidence[:3]],
                explanation=_short(item.explanation),
                needs_human_review=item.needs_human_review,
            )
            for item in result.matches
        ],
        recommendations=[_short(item) for item in result.recommendations],
        warnings=[_short(item) for item in result.warnings],
        processing_time_ms=result.processing_time_ms,
    )


def _recommend_jobs(
    arguments: RecommendJobsInput,
    context: ToolContext,
) -> RecommendJobsOutput:
    """在固定本地岗位库中执行RAG召回、证据匹配和最终排序。"""
    resume_path = context.resolve_input_file(
        arguments.resume_path,
        allowed_extensions=PDF_EXTENSIONS,
        max_file_bytes=MAX_INPUT_BYTES,
    )
    settings = replace(
        BgeEmbeddingSettings.from_environment(),
        # 工具安全清单声明不访问网络，所以工具执行时只允许读取已缓存模型。
        local_files_only=True,
    )
    result = recommend_jobs_with_rag(
        resume_path,
        context.project_root / "data" / "jobs",
        context.project_root / "data" / "vector_store" / "chroma",
        embedding=BgeLocalEmbedding(settings),
        top_k_retrieval=max(arguments.top_k * 2, 10),
        top_k_final=arguments.top_k,
    )
    if not result.success:
        raise DocumentParseError("岗位推荐失败：" + "；".join(result.errors))
    return RecommendJobsOutput(
        resume_source=str(resume_path),
        queries=[query.text for query in result.multi_query.queries],
        jobs_scanned=result.retrieval.total_jobs_scanned,
        chunks_scanned=result.retrieval.total_chunks_scanned,
        recommendations=[
            RecommendedJobSummary(
                rank=item.rank,
                job_path=item.job_source,
                company=item.company,
                title=item.title,
                final_score=item.final_score,
                retrieval_score=item.retrieval_score,
                evidence_score=item.evidence_score,
                evidence_coverage=item.evidence_coverage,
                hard_constraints_failed=item.hard_constraints_failed,
                hard_constraints_unknown=item.hard_constraints_unknown,
                matched_skills=item.matched_skills[:20],
                strengths=[_short(value, 300) for value in item.strengths],
                gaps=[_short(value, 300) for value in item.gaps],
                recommendations=[_short(value, 300) for value in item.recommendations],
            )
            for item in result.recommendations
        ],
        warnings=[_short(value) for value in result.warnings],
        processing_time_ms=result.processing_time_ms,
    )


def builtin_tools() -> list[ModelTool]:
    """创建默认工具，供 :class:`ToolRegistry` 显式注册。

    仅仅在项目里编写或导入一个 Python 函数，并不会自动把它暴露给模型。
    只有加入这个列表并经过注册中心注册的工具才可以被调用。
    """

    # 简历工具处理个人信息，因此在策略中明确标记 handles_personal_data=True。
    local_resume_policy = ToolSafetyPolicy(
        read_only=True,
        network_access=False,
        external_data_transfer=False,
        handles_personal_data=True,
        allowed_extensions=sorted(PDF_EXTENSIONS),
        max_file_bytes=MAX_INPUT_BYTES,
    )
    return [
        # 工具一：PDF 简历 -> 简历摘要。
        ModelTool(
            name="parse_resume_pdf",
            description=(
                "解析项目目录内的一份 PDF 简历，返回姓名、求职方向、教育、技能、经历、"
                "项目及解析质量摘要。仅当需要读取简历内容时调用。"
            ),
            input_model=ParseResumeInput,
            output_model=ParseResumeOutput,
            handler=_parse_resume,
            safety=local_resume_policy,
        ),
        # 工具二：Markdown/TXT 岗位 -> 岗位结构化摘要。
        ModelTool(
            name="parse_job_description",
            description=(
                "解析项目目录内的一份 Markdown/TXT 岗位描述，返回公司、岗位、职责和原子化要求。"
            ),
            input_model=ParseJobInput,
            output_model=ParseJobOutput,
            handler=_parse_job,
            safety=ToolSafetyPolicy(
                read_only=True,
                network_access=False,
                external_data_transfer=False,
                handles_personal_data=False,
                allowed_extensions=sorted(JOB_EXTENSIONS),
                max_file_bytes=MAX_INPUT_BYTES,
            ),
        ),
        # 工具三：PDF 简历 + 岗位文件 -> 可追溯的本地匹配结论。
        ModelTool(
            name="match_resume_job_evidence",
            description=(
                "将项目目录内的 PDF 简历与一份 Markdown/TXT 岗位逐项做本地证据匹配，"
                "返回得分、硬约束、证据原文和改进建议。不会调用外部模型。"
            ),
            input_model=MatchEvidenceInput,
            output_model=MatchEvidenceOutput,
            handler=_match_evidence,
            # model_copy 创建一份新策略，避免修改上面的简历解析策略对象。
            safety=local_resume_policy.model_copy(
                update={"allowed_extensions": sorted(PDF_EXTENSIONS | JOB_EXTENSIONS)}
            ),
        ),
        # 工具四：PDF简历 -> 本地岗位库混合召回、证据重排与推荐。
        ModelTool(
            name="recommend_jobs_with_rag",
            description=(
                "根据项目目录内的一份PDF简历，从本地30份岗位库中生成多查询，"
                "融合BGE向量、BM25和技能精确匹配召回岗位，再执行逐要求证据匹配并"
                "返回最终排序。用户要求推荐、筛选或排名多个岗位时优先调用。"
            ),
            input_model=RecommendJobsInput,
            output_model=RecommendJobsOutput,
            handler=_recommend_jobs,
            safety=local_resume_policy,
        ),
    ]
