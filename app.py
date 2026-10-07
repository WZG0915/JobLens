"""JobLens 命令行入口。

阶段 1 提供固定分析流程；阶段 2 增加由模型选择白名单工具的 Agent 循环。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
# 当前项目还没有安装成 Python 包，因此先把 src 加入模块搜索路径。
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from joblens.exceptions import JobLensError  # noqa: E402
from joblens.agent import (  # noqa: E402
    AgentRunner,
    OpenAIResponsesClient,
    build_default_registry,
)
from joblens.evaluation import evaluate_resume_corpus  # noqa: E402
from joblens.parsers import (  # noqa: E402
    build_zhipin_search_url,
    parse_job_document,
    parse_resume_document,
    parse_zhipin_export,
)
from joblens.pipeline import generate_report  # noqa: E402
from joblens.rag import (  # noqa: E402
    BgeEmbeddingSettings,
    BgeLocalEmbedding,
    build_job_index,
    generate_resume_queries_from_document,
    recommend_jobs_with_rag,
)
from joblens.tools import (  # noqa: E402
    match_resume_job_files,
    match_resume_to_job_directory,
)


SAMPLE_RESUME = PROJECT_ROOT / "data" / "resumes" / "sample_resume.md"
DEFAULT_PDF_RESUME = PROJECT_ROOT / "data" / "resumes" / "01_cn_standard_single_page.pdf"
SAMPLE_JOBS_DIR = PROJECT_ROOT / "data" / "jobs"
DEFAULT_JOB = SAMPLE_JOBS_DIR / "01_baidu_llm_algorithm_graduate.md"
DEFAULT_RESUME_GOLD = PROJECT_ROOT / "data" / "ground_truth" / "resume_parser_gold.json"
DEFAULT_VECTOR_STORE = PROJECT_ROOT / "data" / "vector_store" / "chroma"


def load_text(path: Path) -> str:
    """读取 UTF-8 文本，并拒绝没有有效内容的文件。"""
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"文件为空：{path}")
    return content


def discover_job_files() -> list[Path]:
    """查找全部岗位实验文件，并按文件名返回稳定顺序。"""
    return sorted(SAMPLE_JOBS_DIR.glob("*.md"))


def run_check() -> int:
    """保留阶段 0 的基础就绪检查。"""
    resume = load_text(SAMPLE_RESUME)
    job_files = discover_job_files()
    if len(job_files) != 30:
        raise RuntimeError(f"期望 30 份岗位 JD，实际找到 {len(job_files)} 份。")
    for job_file in job_files:
        load_text(job_file)

    print("JobLens 项目检查成功！")
    print(f"项目目录：{PROJECT_ROOT}")
    print(f"示例简历：{SAMPLE_RESUME.name}（{len(resume)} 个字符）")
    print(f"岗位数据：{len(job_files)} 份")
    print("阶段 1 已就绪：可以执行结构化岗位匹配分析。")
    return 0


def run_analysis(args: argparse.Namespace) -> int:
    """执行阶段 1 的固定分析流程，并展示报告摘要。"""
    resume_path = Path(args.resume).resolve()
    job_path = Path(args.job).resolve()
    output_path = Path(args.output).resolve() if args.output else None

    report, saved_to = generate_report(
        resume_path=resume_path,
        job_path=job_path,
        provider=args.provider,
        output_path=output_path,
    )

    print("JobLens 阶段 1 分析完成！")
    print(f"岗位：{report.job_company} — {report.job_title}")
    print(f"匹配：{report.summary.matched}")
    print(f"部分匹配：{report.summary.partially_matched}")
    print(f"证据不足：{report.summary.insufficient_evidence}")
    print(f"明确不匹配：{report.summary.not_matched}")
    print(f"证据覆盖率：{report.summary.evidence_coverage:.1%}")
    print(f"报告：{saved_to}")
    return 0


def _write_structured_result(result, output: str | None, default_name: str) -> Path:
    """写出富结构解析结果，供调试、评测和后续 Agent 工具复用。"""
    destination = (
        Path(output).expanduser().resolve()
        if output
        else PROJECT_ROOT / "reports" / default_name
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return destination


def run_parse_resume(args: argparse.Namespace) -> int:
    """解析命令行指定的简历文件，保存 JSON 并打印字段数量摘要。"""
    result = parse_resume_document(Path(args.resume).expanduser().resolve())
    saved_to = _write_structured_result(result, args.output, "parsed_resume.json")
    data = result.data
    print("简历解析完成！")
    print(f"姓名：{data.basic_information.name if data else '未识别'}")
    print(f"教育：{len(data.education) if data else 0} 项")
    print(f"工作/实习：{len(data.work_experiences) if data else 0} 项")
    print(f"技能：{len(data.skills) if data else 0} 项")
    print(f"项目：{len(data.projects) if data else 0} 项")
    if result.metadata.page_count:
        print(f"PDF 页数：{result.metadata.page_count}")
    print(f"警告：{len(result.warnings)} 项")
    print(f"结果：{saved_to}")
    return 0


def run_parse_job(args: argparse.Namespace) -> int:
    """解析命令行指定的岗位文件，保存 JSON 并打印职责、要求数量。"""
    result = parse_job_document(Path(args.job).expanduser().resolve())
    saved_to = _write_structured_result(result, args.output, "parsed_job.json")
    data = result.data
    print("岗位解析完成！")
    print(f"职责：{len(data.responsibilities) if data else 0} 项")
    print(f"要求：{len(data.requirements) if data else 0} 项")
    print(f"警告：{len(result.warnings)} 项")
    print(f"结果：{saved_to}")
    return 0


def run_match_evidence(args: argparse.Namespace) -> int:
    """对一份简历和一个岗位执行富结构证据匹配。"""
    result = match_resume_job_files(
        Path(args.resume).expanduser().resolve(),
        Path(args.job).expanduser().resolve(),
        mode=args.mode,
    )
    saved_to = _write_structured_result(result, args.output, "evidence_match.json")
    if not result.success or result.summary is None:
        print("证据匹配失败：" + "；".join(result.errors), file=sys.stderr)
        print(f"结果：{saved_to}")
        return 1
    summary = result.summary
    print("简历—岗位证据匹配完成！")
    print(f"岗位：{result.job_company or '未知公司'} — {result.job_title or '未知岗位'}")
    print(f"匹配：{summary.matched}")
    print(f"部分匹配：{summary.partially_matched}")
    print(f"证据不足：{summary.insufficient_evidence}")
    print(f"明确不匹配：{summary.not_matched}")
    print(f"硬约束失败：{summary.hard_constraints_failed}")
    print(f"证据覆盖率：{summary.evidence_coverage:.1%}")
    print(f"加权得分：{summary.weighted_score:.2f}")
    print(f"需要复核：{summary.review_required}")
    print(f"结果：{saved_to}")
    return 0


def run_match_jobs(args: argparse.Namespace) -> int:
    """将一份简历与岗位目录批量匹配并输出排名。"""
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else PROJECT_ROOT / "reports" / "evidence_matches"
    )
    result = match_resume_to_job_directory(
        Path(args.resume).expanduser().resolve(),
        Path(args.jobs_dir).expanduser().resolve(),
        mode=args.mode,
        output_dir=output_dir,
    )
    saved_to = _write_structured_result(
        result,
        args.summary,
        "evidence_match_batch.json",
    )
    print("批量证据匹配完成！")
    print(f"成功岗位：{result.successful_jobs}/{result.job_count}")
    for item in result.rankings:
        print(
            f"{item.rank}. {item.job_company} — {item.job_title} | "
            f"得分 {item.weighted_score:.2f} | 硬约束失败 {item.hard_constraints_failed}"
        )
    if result.errors:
        print("失败：" + "；".join(result.errors), file=sys.stderr)
    print(f"逐岗位结果：{output_dir}")
    print(f"排名汇总：{saved_to}")
    return 0 if result.success else 1


def run_zhipin_search_url(args: argparse.Namespace) -> int:
    """生成可由用户在普通浏览器中打开的官方搜索地址。"""
    url = build_zhipin_search_url(args.query, args.city_code)
    print("BOSS直聘官方搜索链接：")
    print(url)
    print("请在浏览器中正常访问并手动保存单个岗位详情页；本工具不会自动登录或爬取。")
    return 0


def run_parse_zhipin_export(args: argparse.Namespace) -> int:
    """解析用户手动保存的 BOSS直聘单个岗位页面或复制文本。"""
    result = parse_zhipin_export(
        Path(args.input).expanduser().resolve(),
        source_url=args.url,
    )
    saved_to = _write_structured_result(result, args.output, "parsed_zhipin_job.json")
    data = result.data
    print("BOSS直聘岗位导入完成！")
    print(f"岗位：{data.basic_information.title if data else '未识别'}")
    print(f"公司：{data.company.name if data and data.company.name else '未识别'}")
    print(f"职责：{len(data.responsibilities) if data else 0} 项")
    print(f"要求：{len(data.requirements) if data else 0} 项")
    print(f"警告：{len(result.warnings)} 项")
    print(f"结果：{saved_to}")
    return 0


def run_evaluate_resumes(args: argparse.Namespace) -> int:
    """对标准答案集运行字段级 Precision、Recall 和 F1 评测。"""
    gold_path = Path(args.gold).expanduser().resolve()
    report = evaluate_resume_corpus(gold_path, PROJECT_ROOT)
    destination = (
        Path(args.output).expanduser().resolve()
        if args.output
        else PROJECT_ROOT / "reports" / "resume_parser_evaluation.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("简历解析器评测完成！")
    print(f"样本：{report['successful_samples']}/{report['sample_count']}")
    print(f"完全匹配：{report['exact_match_samples']}/{report['sample_count']}")
    print(f"Precision：{report['overall']['precision']:.2%}")
    print(f"Recall：{report['overall']['recall']:.2%}")
    print(f"F1：{report['overall']['f1']:.2%}")
    print(f"结果：{destination}")
    return 0


def run_build_job_index(args: argparse.Namespace) -> int:
    """使用本地 BGE 为全部岗位分块生成 Chroma 向量索引。"""
    settings = BgeEmbeddingSettings(
        model_name=args.model,
        device=args.device,
        batch_size=args.embedding_batch_size,
        local_files_only=args.local_files_only,
    )
    embedding = BgeLocalEmbedding(settings)
    result = build_job_index(
        Path(args.jobs_dir).expanduser().resolve(),
        Path(args.persist_dir).expanduser().resolve(),
        embedding=embedding,
        collection_name=args.collection,
        chroma_batch_size=args.chroma_batch_size,
    )
    saved_to = _write_structured_result(result, args.output, "job_index_build.json")
    print("岗位向量索引构建完成！")
    print(f"岗位：{result.job_count} 份")
    print(f"Chunk：{result.stored_chunk_count} 条")
    print(f"Embedding：{result.embedding_model}（{result.embedding_dimension} 维）")
    print(f"Chroma：{result.persist_directory}")
    print(f"构建报告：{saved_to}")
    return 0


def run_generate_resume_queries(args: argparse.Namespace) -> int:
    """从简历结构化字段生成多个互补岗位检索查询。"""
    result = generate_resume_queries_from_document(
        Path(args.resume).expanduser().resolve(),
        max_queries=args.max_queries,
        max_projects=args.max_projects,
        max_work_experiences=args.max_work_experiences,
    )
    saved_to = _write_structured_result(result, args.output, "resume_queries.json")
    print("简历多查询生成完成！")
    for query in result.queries:
        print(f"{query.query_id} [{query.query_type.value}] 权重 {query.weight:.2f}")
        print(f"  {query.text}")
    print(f"结果：{saved_to}")
    return 0


def run_recommend_jobs_rag(args: argparse.Namespace) -> int:
    """运行本地混合检索、证据匹配与最终岗位排序。"""
    embedding = BgeLocalEmbedding(
        BgeEmbeddingSettings(
            model_name=args.model,
            device=args.device,
            batch_size=args.embedding_batch_size,
            local_files_only=args.local_files_only,
        )
    )
    result = recommend_jobs_with_rag(
        Path(args.resume).expanduser().resolve(),
        Path(args.jobs_dir).expanduser().resolve(),
        Path(args.persist_dir).expanduser().resolve(),
        embedding=embedding,
        collection_name=args.collection,
        top_k_retrieval=args.top_k_retrieval,
        top_k_final=args.top_k,
        top_k_per_query=args.top_k_per_query,
    )
    saved_to = _write_structured_result(result, args.output, "rag_job_recommendations.json")
    if not result.success:
        print("RAG岗位推荐失败：" + "；".join(result.errors), file=sys.stderr)
        print(f"结果：{saved_to}")
        return 1
    print("RAG岗位推荐完成！")
    for item in result.recommendations:
        print(
            f"{item.rank}. {item.company or '未知公司'} — {item.title} | "
            f"最终 {item.final_score:.2f} | RAG {item.retrieval_score:.3f} | "
            f"证据 {item.evidence_score:.2f} | 硬约束失败 {item.hard_constraints_failed}"
        )
    print(f"结果：{saved_to}")
    return 0


def run_web(args: argparse.Namespace) -> int:
    """启动只监听本机的JobLens对话网页。"""
    try:
        import uvicorn
        from joblens.web import create_app
    except ImportError as exc:
        raise JobLensError('缺少网页依赖，请运行 pip install -e ".[rag,web]"') from exc
    web_app = create_app(PROJECT_ROOT)
    print(f"JobLens 网页已启动：http://{args.host}:{args.port}")
    print("按 Ctrl+C 可以停止服务。")
    uvicorn.run(web_app, host=args.host, port=args.port, log_level="info")
    return 0


def run_agent(args: argparse.Namespace) -> int:
    """运行阶段二模型—工具循环，并保存完整的安全执行轨迹。"""

    registry = build_default_registry(PROJECT_ROOT)
    model_client = OpenAIResponsesClient()
    runner = AgentRunner(
        model_client,
        registry,
        max_steps=args.max_steps,
        max_tool_calls=args.max_tool_calls,
    )
    result = runner.run(args.message)
    saved_to = _write_structured_result(result, args.output, "agent_run.json")

    print("JobLens Agent 运行完成！" if result.success else "JobLens Agent 已安全停止。")
    print(result.answer)
    print(f"模型轮数：{result.model_turns}")
    print(f"工具调用：{result.tool_call_count}")
    print(f"停止原因：{result.stop_reason.value}")
    print(f"执行轨迹：{saved_to}")
    if result.errors:
        print("错误摘要：" + "；".join(result.errors), file=sys.stderr)
    return 0 if result.success else 1


def build_parser() -> argparse.ArgumentParser:
    """定义固定流程、解析工具和阶段二 Agent 命令。"""
    parser = argparse.ArgumentParser(
        description="JobLens：实习岗位与简历证据匹配分析器"
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("check", help="检查项目目录和阶段 0 数据")

    analyze = subparsers.add_parser("analyze", help="执行阶段 1 固定分析流程")
    analyze.add_argument(
        "--resume", default=str(SAMPLE_RESUME), help="简历 PDF/Markdown/TXT 文件路径"
    )
    analyze.add_argument(
        "--job", default=str(DEFAULT_JOB), help="岗位 JD Markdown/TXT 文件路径"
    )
    analyze.add_argument(
        "--provider",
        choices=("local", "openai"),
        default="local",
        help="local 为可复现规则解析；openai 为真实结构化输出 API",
    )
    analyze.add_argument("--output", help="报告 JSON 路径；不填写时保存到 reports/")

    parse_resume = subparsers.add_parser(
        "parse-resume", help="将简历解析为包含原文证据的富结构 JSON"
    )
    parse_resume.add_argument(
        "--resume", default=str(DEFAULT_PDF_RESUME), help="简历 PDF/Markdown/TXT 文件路径"
    )
    parse_resume.add_argument("--output", help="解析结果 JSON 路径")

    parse_job = subparsers.add_parser(
        "parse-job", help="将岗位 JD 解析为包含匹配规则的富结构 JSON"
    )
    parse_job.add_argument(
        "--job", default=str(DEFAULT_JOB), help="岗位 JD Markdown/TXT 文件路径"
    )
    parse_job.add_argument("--output", help="解析结果 JSON 路径")

    match_evidence = subparsers.add_parser(
        "match-evidence", help="对简历和单个岗位执行可追溯证据匹配"
    )
    match_evidence.add_argument(
        "--resume", default=str(DEFAULT_PDF_RESUME), help="简历 PDF/Markdown/TXT 文件路径"
    )
    match_evidence.add_argument(
        "--job", default=str(DEFAULT_JOB), help="岗位 JD Markdown/TXT 文件路径"
    )
    match_evidence.add_argument(
        "--mode",
        choices=("local", "hybrid"),
        default="local",
        help="local 为确定性规则；hybrid 对模糊项使用模型复核",
    )
    match_evidence.add_argument("--output", help="证据匹配 JSON 路径")

    match_jobs = subparsers.add_parser(
        "match-jobs", help="将一份简历与岗位目录批量匹配并生成排名"
    )
    match_jobs.add_argument(
        "--resume", default=str(DEFAULT_PDF_RESUME), help="简历 PDF/Markdown/TXT 文件路径"
    )
    match_jobs.add_argument(
        "--jobs-dir", default=str(SAMPLE_JOBS_DIR), help="岗位文件目录"
    )
    match_jobs.add_argument(
        "--mode", choices=("local", "hybrid"), default="local", help="匹配模式"
    )
    match_jobs.add_argument("--output-dir", help="逐岗位匹配 JSON 输出目录")
    match_jobs.add_argument("--summary", help="批量排名汇总 JSON 路径")

    zhipin_url = subparsers.add_parser(
        "zhipin-search-url", help="生成 BOSS直聘官方岗位搜索链接（不自动抓取）"
    )
    zhipin_url.add_argument("--query", required=True, help="岗位搜索关键词")
    zhipin_url.add_argument(
        "--city-code", default="100010000", help="9 位 BOSS直聘城市代码"
    )

    zhipin_import = subparsers.add_parser(
        "parse-zhipin-export", help="解析手动保存的单个 BOSS直聘 HTML/Markdown/TXT"
    )
    zhipin_import.add_argument("--input", required=True, help="手动保存的岗位文件路径")
    zhipin_import.add_argument("--url", help="原始岗位详情页 https://*.zhipin.com/ 链接")
    zhipin_import.add_argument("--output", help="解析结果 JSON 路径")

    evaluate_resumes = subparsers.add_parser(
        "evaluate-resumes", help="使用标准答案集评测简历解析器"
    )
    evaluate_resumes.add_argument(
        "--gold", default=str(DEFAULT_RESUME_GOLD), help="标准答案 JSON 路径"
    )
    evaluate_resumes.add_argument("--output", help="评测报告 JSON 路径")

    build_index = subparsers.add_parser(
        "build-job-index", help="使用本地 BGE 和 Chroma 构建岗位向量索引"
    )
    build_index.add_argument(
        "--jobs-dir", default=str(SAMPLE_JOBS_DIR), help="岗位 Markdown/TXT 目录"
    )
    build_index.add_argument(
        "--persist-dir", default=str(DEFAULT_VECTOR_STORE), help="Chroma 持久化目录"
    )
    build_index.add_argument(
        "--collection", default="joblens_job_chunks", help="Chroma 集合名称"
    )
    build_index.add_argument(
        "--model", default="BAAI/bge-small-zh-v1.5", help="本地BGE模型名称或路径"
    )
    build_index.add_argument(
        "--device", default="auto", help="运行设备：auto、cpu 或 cuda"
    )
    build_index.add_argument(
        "--embedding-batch-size", type=int, default=32, help="BGE 编码批大小"
    )
    build_index.add_argument(
        "--chroma-batch-size", type=int, default=64, help="Chroma 写入批大小"
    )
    build_index.add_argument(
        "--local-files-only",
        action="store_true",
        help="只使用本机已缓存的BGE模型，不尝试联网下载",
    )
    build_index.add_argument("--output", help="索引构建报告 JSON 路径")

    resume_queries = subparsers.add_parser(
        "generate-resume-queries", help="解析简历并生成多条岗位检索查询"
    )
    resume_queries.add_argument(
        "--resume", default=str(DEFAULT_PDF_RESUME), help="简历 PDF/Markdown/TXT 路径"
    )
    resume_queries.add_argument("--max-queries", type=int, default=10)
    resume_queries.add_argument("--max-projects", type=int, default=3)
    resume_queries.add_argument("--max-work-experiences", type=int, default=2)
    resume_queries.add_argument("--output", help="多查询结果 JSON 路径")

    recommend_rag = subparsers.add_parser(
        "recommend-jobs-rag", help="使用混合RAG和证据匹配推荐本地岗位"
    )
    recommend_rag.add_argument(
        "--resume", default=str(DEFAULT_PDF_RESUME), help="简历 PDF/Markdown/TXT 路径"
    )
    recommend_rag.add_argument("--jobs-dir", default=str(SAMPLE_JOBS_DIR))
    recommend_rag.add_argument("--persist-dir", default=str(DEFAULT_VECTOR_STORE))
    recommend_rag.add_argument("--collection", default="joblens_job_chunks")
    recommend_rag.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    recommend_rag.add_argument("--device", default="auto")
    recommend_rag.add_argument("--embedding-batch-size", type=int, default=32)
    recommend_rag.add_argument("--top-k-retrieval", type=int, default=10)
    recommend_rag.add_argument("--top-k", type=int, default=5)
    recommend_rag.add_argument("--top-k-per-query", type=int, default=30)
    recommend_rag.add_argument("--local-files-only", action="store_true")
    recommend_rag.add_argument("--output", help="完整推荐结果 JSON 路径")

    web = subparsers.add_parser("web", help="启动本地对话式岗位推荐网页")
    web.add_argument(
        "--host",
        default="127.0.0.1",
        help="监听地址，默认只允许本机访问",
    )
    web.add_argument("--port", type=int, default=8000, help="网页端口，默认8000")

    agent = subparsers.add_parser(
        "agent", help="运行阶段二 Agent，由模型选择并调用三个白名单工具"
    )
    agent.add_argument(
        "--message",
        required=True,
        help="交给 Agent 的任务，例如解析简历或比较简历与岗位",
    )
    agent.add_argument(
        "--max-steps",
        type=int,
        default=6,
        help="最大模型决策轮数，默认 6",
    )
    agent.add_argument(
        "--max-tool-calls",
        type=int,
        default=6,
        help="最大工具调用次数，默认 6",
    )
    agent.add_argument("--output", help="Agent 执行轨迹 JSON 路径")
    return parser


def main() -> int:
    """解析命令行参数，并将可预期错误转成简洁提示。"""
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command in (None, "check"):
            return run_check()
        if args.command == "analyze":
            return run_analysis(args)
        if args.command == "parse-resume":
            return run_parse_resume(args)
        if args.command == "parse-job":
            return run_parse_job(args)
        if args.command == "match-evidence":
            return run_match_evidence(args)
        if args.command == "match-jobs":
            return run_match_jobs(args)
        if args.command == "zhipin-search-url":
            return run_zhipin_search_url(args)
        if args.command == "parse-zhipin-export":
            return run_parse_zhipin_export(args)
        if args.command == "evaluate-resumes":
            return run_evaluate_resumes(args)
        if args.command == "build-job-index":
            return run_build_job_index(args)
        if args.command == "generate-resume-queries":
            return run_generate_resume_queries(args)
        if args.command == "recommend-jobs-rag":
            return run_recommend_jobs_rag(args)
        if args.command == "web":
            return run_web(args)
        if args.command == "agent":
            return run_agent(args)
        parser.error(f"未知命令：{args.command}")
    except (JobLensError, OSError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
