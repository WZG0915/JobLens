"""FastAPI 对话网页后端。

网页负责接收 PDF、维护临时会话，并把自然语言请求交给阶段二 ``AgentRunner``。
模型只能调用注册中心中的白名单工具；PDF 读取、RAG 检索和证据匹配仍在本机完成。
当外部模型暂时不可用时，服务会明确告知用户，并回退到原有本地流程。
"""

from __future__ import annotations

import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from joblens.agent import (
    AgentModelClient,
    AgentRunner,
    OpenAIResponsesClient,
    ToolExecutionResult,
    ToolRegistry,
    build_default_registry,
)
from joblens.config import load_llm_settings
from joblens.exceptions import JobLensError
from joblens.parsers import parse_resume_document
from joblens.rag import (
    BgeEmbeddingSettings,
    BgeLocalEmbedding,
    JobRecommendationResult,
    recommend_jobs_with_rag,
)


MAX_RESUME_BYTES = 25 * 1024 * 1024
MAX_HISTORY_TURNS = 4
MAX_HISTORY_MESSAGE_CHARS = 1200
RECOMMEND_PATTERN = re.compile(r"推荐|岗位|职位|匹配|适合|排行|排名|工作")
RESUME_PATTERN = re.compile(r"简历|解析|技能|项目|教育|经历|能力")


class WebModel(BaseModel):
    """网页 API 使用的严格数据模型。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RecommendationView(WebModel):
    """浏览器展示的一张岗位推荐卡片。"""

    rank: int
    company: str | None = None
    title: str
    job_source: str
    final_score: float
    retrieval_score: float
    evidence_score: float
    evidence_coverage: float
    hard_constraints_failed: int
    matched_skills: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class ChatResponse(WebModel):
    """一次聊天回复；结构化岗位卡与自然语言提示分开返回。"""

    session_id: str
    message: str
    resume_name: str | None = None
    recommendations: list[RecommendationView] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    processing_time_ms: float = Field(ge=0)
    answered_by: str = "ai"
    tool_calls: list[str] = Field(default_factory=list)


class HealthResponse(WebModel):
    """前端用于显示模型、本地服务和岗位索引状态。"""

    status: str
    index_ready: bool
    jobs_ready: bool
    model_name: str
    ai_ready: bool
    ai_model: str | None = None
    dialogue_mode: str


@dataclass
class ResumeSession:
    """浏览器会话、随机上传文件和最近对话之间的服务端映射。"""

    session_id: str
    path: Path
    original_name: str
    # 只保存最近的用户问题与助手答复，不保存完整工具结果和 PDF 原文。
    history: list[tuple[str, str]] = field(default_factory=list)


class SessionStore:
    """线程安全的进程内简历会话；删除会话时同步删除上传文件。"""

    def __init__(self, upload_directory: Path) -> None:
        self.upload_directory = upload_directory.resolve()
        self.upload_directory.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, ResumeSession] = {}
        self._lock = threading.RLock()

    def save(self, content: bytes, original_name: str, session_id: str | None = None) -> ResumeSession:
        """使用随机文件名保存已通过 PDF 校验的上传内容。"""
        active_id = session_id if session_id and session_id in self._sessions else uuid.uuid4().hex
        path = self.upload_directory / f"{active_id}.pdf"
        with self._lock:
            previous = self._sessions.get(active_id)
            if previous and previous.path != path:
                previous.path.unlink(missing_ok=True)
            path.write_bytes(content)
            # 新上传可能是另一份简历，因此重置旧对话，避免两个候选人的上下文混淆。
            session = ResumeSession(active_id, path, original_name)
            self._sessions[active_id] = session
        return session

    def get(self, session_id: str | None) -> ResumeSession | None:
        """取得当前会话；无效或过期 ID 返回 None。"""
        if not session_id:
            return None
        with self._lock:
            return self._sessions.get(session_id)

    def append_history(self, session_id: str, user_message: str, answer: str) -> None:
        """保存有限长度的对话摘要，供下一轮模型理解指代和追问。"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            session.history.append((user_message, answer))
            del session.history[:-MAX_HISTORY_TURNS]

    def history_snapshot(self, session_id: str) -> list[tuple[str, str]]:
        """返回历史副本，避免在模型请求期间长期持有线程锁。"""
        with self._lock:
            session = self._sessions.get(session_id)
            return list(session.history) if session else []

    def delete(self, session_id: str) -> bool:
        """删除会话和对应的随机上传文件。"""
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session:
            session.path.unlink(missing_ok=True)
            return True
        return False


class CapturingRegistry:
    """装饰工具注册中心，并保留本轮结构化结果供网页绘制岗位卡片。"""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        self.results: list[ToolExecutionResult] = []

    def openai_tools(self, *, api: str = "responses") -> list[dict[str, Any]]:
        """把原注册中心的工具定义原样交给模型。"""
        return self._registry.openai_tools(api=api)  # type: ignore[arg-type]

    def execute(self, name: str, arguments: str | dict[str, Any]) -> ToolExecutionResult:
        """仍由安全注册中心执行工具，并额外记录经过校验的返回值。"""
        result = self._registry.execute(name, arguments)
        self.results.append(result)
        return result


def _validate_pdf(filename: str | None, content: bytes) -> None:
    """限制上传类型、大小并检查 PDF 文件头。"""
    if not filename or Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="只支持上传PDF简历")
    if not content:
        raise HTTPException(status_code=400, detail="上传的简历为空")
    if len(content) > MAX_RESUME_BYTES:
        raise HTTPException(status_code=413, detail="PDF超过25 MB限制")
    if not content.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="文件内容不是有效的PDF")


@lru_cache(maxsize=1)
def _local_embedding() -> BgeLocalEmbedding:
    """每个服务进程只加载一次 BGE，避免每条回退消息重复等待模型加载。"""
    environment = BgeEmbeddingSettings.from_environment()
    settings = BgeEmbeddingSettings(
        model_name=environment.model_name,
        device=environment.device,
        batch_size=environment.batch_size,
        normalize_embeddings=environment.normalize_embeddings,
        query_instruction=environment.query_instruction,
        local_files_only=True,
    )
    return BgeLocalEmbedding(settings)


def _resume_summary(path: Path) -> tuple[str, list[str]]:
    """生成不含联系方式和原文的本地简历摘要。"""
    result = parse_resume_document(path)
    if not result.success or result.data is None:
        raise ValueError("简历解析失败：" + "；".join(result.errors))
    data = result.data
    preference = data.job_preference
    target_roles = "、".join(preference.target_roles) if preference else "未填写"
    skills = "、".join(skill.normalized_name for skill in data.skills[:12]) or "未识别"
    message = (
        "简历已经在本地解析完成。\n"
        f"求职方向：{target_roles}\n"
        f"识别到教育经历 {len(data.education)} 项、工作/实习经历 {len(data.work_experiences)} 项、"
        f"项目 {len(data.projects)} 项。\n"
        f"主要技能：{skills}\n\n"
        "你可以继续输入“推荐5个岗位”，我会运行本地RAG检索和证据匹配。"
    )
    return message, [warning.message for warning in result.warnings]


def _recommend(project_root: Path, resume_path: Path, top_k: int) -> JobRecommendationResult:
    """运行原有确定性推荐，供 AI 不可用时回退。"""
    return recommend_jobs_with_rag(
        resume_path,
        project_root / "data" / "jobs",
        project_root / "data" / "vector_store" / "chroma",
        embedding=_local_embedding(),
        top_k_retrieval=max(10, top_k * 2),
        top_k_final=top_k,
    )


def _recommendation_message(result: JobRecommendationResult) -> str:
    """生成本地回退流程的简短结论。"""
    best = result.recommendations[0]
    return (
        f"已从 {result.retrieval.total_jobs_scanned} 份岗位、"
        f"{result.retrieval.total_chunks_scanned} 个结构化片段中完成检索，并对候选岗位逐项核对简历证据。\n\n"
        f"当前最推荐：{best.company or '未知公司'} — {best.title}，"
        f"综合得分 {best.final_score:.2f}。下方可以查看完整排名、匹配优势和能力差距。"
    )


def _to_views(result: JobRecommendationResult) -> list[RecommendationView]:
    """从本地完整审计结果提取浏览器需要的安全摘要。"""
    return [
        RecommendationView(
            rank=item.rank,
            company=item.company,
            title=item.title,
            job_source=Path(item.job_source).name,
            final_score=item.final_score,
            retrieval_score=item.retrieval_score,
            evidence_score=item.evidence_score,
            evidence_coverage=item.evidence_coverage,
            hard_constraints_failed=item.hard_constraints_failed,
            matched_skills=item.matched_skills[:12],
            strengths=item.strengths[:4],
            gaps=item.gaps[:4],
            suggestions=item.recommendations[:4],
        )
        for item in result.recommendations
    ]


def _views_from_tool_results(
    results: list[ToolExecutionResult],
) -> tuple[list[RecommendationView], list[str], list[str]]:
    """提取 AI 调用 RAG 工具后已经通过输出契约校验的数据。"""
    for result in reversed(results):
        if not result.ok or result.tool_name != "recommend_jobs_with_rag" or not result.data:
            continue
        data = result.data
        views = [
            RecommendationView(
                rank=item["rank"],
                company=item.get("company"),
                title=item["title"],
                job_source=Path(item["job_path"]).name,
                final_score=item["final_score"],
                retrieval_score=item["retrieval_score"],
                evidence_score=item["evidence_score"],
                evidence_coverage=item["evidence_coverage"],
                hard_constraints_failed=item["hard_constraints_failed"],
                matched_skills=item.get("matched_skills", [])[:12],
                strengths=item.get("strengths", [])[:4],
                gaps=item.get("gaps", [])[:4],
                suggestions=item.get("recommendations", [])[:4],
            )
            for item in data.get("recommendations", [])
        ]
        return views, list(data.get("queries", [])), list(data.get("warnings", []))
    return [], [], []


def _agent_prompt(
    root: Path,
    session: ResumeSession,
    history: list[tuple[str, str]],
    user_message: str,
    top_k: int,
) -> str:
    """给模型补充可信文件路径和有限对话上下文，不发送 PDF 二进制。"""
    try:
        resume_path = session.path.relative_to(root).as_posix()
    except ValueError:
        resume_path = str(session.path)
    history_lines: list[str] = []
    for old_user, old_answer in history[-MAX_HISTORY_TURNS:]:
        history_lines.append("用户：" + old_user[:MAX_HISTORY_MESSAGE_CHARS])
        history_lines.append("助手：" + old_answer[:MAX_HISTORY_MESSAGE_CHARS])
    history_text = "\n".join(history_lines) if history_lines else "（这是本会话第一轮）"
    return (
        "这是 JobLens 网页会话。服务端已经保存用户上传的 PDF，当前简历的项目相对路径是："
        f"{resume_path}\n"
        "需要读取简历、匹配岗位或推荐岗位时，直接把这个路径传给对应工具；不要要求用户再次上传。\n"
        f"若用户要求推荐岗位，返回数量使用 top_k={top_k}，并优先调用 recommend_jobs_with_rag。\n"
        "PDF 二进制和联系方式不会直接提供给你；请仅依据本地工具返回的精简结果回答。\n\n"
        f"最近对话：\n{history_text}\n\n"
        f"本轮用户消息：{user_message}"
    )


def _run_agent(
    root: Path,
    model_client_factory: Callable[[], AgentModelClient],
    prompt: str,
) -> tuple[Any, CapturingRegistry]:
    """在线程池中运行同步模型—工具循环。"""
    registry = CapturingRegistry(build_default_registry(root))
    runner = AgentRunner(model_client_factory(), registry)  # type: ignore[arg-type]
    return runner.run(prompt), registry


def create_app(
    project_root: str | Path | None = None,
    *,
    model_client_factory: Callable[[], AgentModelClient] | None = None,
) -> FastAPI:
    """创建本地 Web 应用；测试可注入不会联网的模型客户端。"""
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[3]
    static_directory = Path(__file__).resolve().parent / "static"
    sessions = SessionStore(root / "data" / "private" / "web_uploads")
    recommendation_lock = threading.Lock()
    client_factory = model_client_factory or OpenAIResponsesClient

    application = FastAPI(
        title="JobLens AI 求职助手",
        version="0.2.0",
        docs_url="/api/docs",
        redoc_url=None,
    )
    application.mount("/static", StaticFiles(directory=static_directory), name="static")

    @application.get("/", include_in_schema=False)
    async def home() -> FileResponse:
        """返回单页对话界面。"""
        return FileResponse(static_directory / "index.html")

    @application.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """检查岗位索引和 AI 配置；不会发送模型请求，也不会返回 Key。"""
        jobs_ready = len(list((root / "data" / "jobs").glob("*.md"))) > 0
        index_ready = (root / "data" / "vector_store" / "chroma" / "chroma.sqlite3").exists()
        try:
            if model_client_factory is None:
                settings = load_llm_settings()
                ai_model = settings.model
                # 实例化只校验 SDK 与配置，不会访问网络。
                client_factory()
            else:
                client_factory()
                ai_model = "test-client"
            ai_ready = True
        except Exception:
            ai_ready = False
            ai_model = None
        fully_ready = jobs_ready and index_ready and ai_ready
        return HealthResponse(
            status="ready" if fully_ready else "degraded",
            index_ready=index_ready,
            jobs_ready=jobs_ready,
            model_name=BgeEmbeddingSettings.from_environment().model_name,
            ai_ready=ai_ready,
            ai_model=ai_model,
            dialogue_mode="ai" if ai_ready else "local_fallback",
        )

    @application.post("/api/chat", response_model=ChatResponse)
    async def chat(
        message: str = Form(default=""),
        session_id: str | None = Form(default=None),
        top_k: int = Form(default=5),
        resume_file: UploadFile | None = File(default=None),
    ) -> ChatResponse:
        """接收自然语言和可选 PDF，由 AI 决策是否调用本地工具。"""
        started = time.perf_counter()
        if not 1 <= top_k <= 10:
            raise HTTPException(status_code=422, detail="top_k必须在1到10之间")
        if len(message) > 2000:
            raise HTTPException(status_code=413, detail="消息不能超过2000个字符")

        session = sessions.get(session_id)
        uploaded = False
        if resume_file is not None:
            content = await resume_file.read(MAX_RESUME_BYTES + 1)
            _validate_pdf(resume_file.filename, content)
            session = sessions.save(content, resume_file.filename or "resume.pdf", session_id)
            uploaded = True
        if session is None:
            raise HTTPException(status_code=400, detail="请先上传一份PDF简历")

        cleaned_message = message.strip() or ("解析我的简历" if uploaded else "请分析我的求职情况")
        history = sessions.history_snapshot(session.session_id)
        prompt = _agent_prompt(root, session, history, cleaned_message, top_k)
        recommendations: list[RecommendationView] = []
        queries: list[str] = []
        warnings: list[str] = []
        tool_calls: list[str] = []
        answered_by = "ai"

        try:
            agent_result, captured = await run_in_threadpool(
                _run_agent, root, client_factory, prompt
            )
            tool_calls = [
                trace.tool_name
                for step in agent_result.steps
                for trace in step.tool_calls
            ]
            if not agent_result.success:
                raise JobLensError("；".join(agent_result.errors) or agent_result.answer)
            response_message = agent_result.answer
            recommendations, queries, tool_warnings = _views_from_tool_results(captured.results)
            warnings.extend(tool_warnings)
        except Exception as ai_error:
            # AI 故障不影响本地解析和检索，但必须明确标注答复来源，不能伪装成模型结果。
            answered_by = "local_fallback"
            warnings.append(f"AI连接失败，已使用本地流程：{str(ai_error)}")
            try:
                if RECOMMEND_PATTERN.search(cleaned_message):
                    def run_locked() -> JobRecommendationResult:
                        with recommendation_lock:
                            return _recommend(root, session.path, top_k)

                    result = await run_in_threadpool(run_locked)
                    if not result.success:
                        raise ValueError("；".join(result.errors))
                    response_message = _recommendation_message(result)
                    recommendations = _to_views(result)
                    queries = [query.text for query in result.multi_query.queries]
                    warnings.extend(result.warnings)
                elif uploaded or RESUME_PATTERN.search(cleaned_message):
                    response_message, local_warnings = await run_in_threadpool(
                        _resume_summary, session.path
                    )
                    warnings.extend(local_warnings)
                else:
                    response_message = (
                        "AI 当前不可用。本地模式仍可解析简历并推荐岗位，"
                        "请输入“解析我的简历”或“推荐5个岗位”。"
                    )
            except (JobLensError, OSError, ValueError) as local_error:
                raise HTTPException(status_code=503, detail=str(local_error)) from local_error

        sessions.append_history(session.session_id, cleaned_message, response_message)
        return ChatResponse(
            session_id=session.session_id,
            message=response_message,
            resume_name=session.original_name,
            recommendations=recommendations,
            queries=queries,
            warnings=warnings,
            processing_time_ms=round((time.perf_counter() - started) * 1000, 2),
            answered_by=answered_by,
            tool_calls=tool_calls,
        )

    @application.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str) -> dict[str, bool]:
        """允许用户主动清除本地上传的简历和对话历史。"""
        return {"deleted": sessions.delete(session_id)}

    return application


app = create_app()
