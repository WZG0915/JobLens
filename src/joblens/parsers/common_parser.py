"""Markdown 规则解析器共用的小型文本工具。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TypeVar

from joblens.schemas import SourceReference


@dataclass(frozen=True)
class LocatedLine:
    """保留章节、行号和字符范围的一行原文。"""

    section: str
    text: str
    line_number: int
    start_char: int
    end_char: int

    def reference(self, text: str | None = None) -> SourceReference:
        """把当前行或其中一段文本转换为可追溯的来源引用。"""
        evidence = (text if text is not None else self.text).strip()
        relative_start = self.text.find(evidence)
        start = self.start_char + max(relative_start, 0)
        return SourceReference(
            section=self.section,
            text=evidence,
            start_char=start,
            end_char=start + len(evidence),
            line_number=self.line_number,
        )


@dataclass(frozen=True)
class SkillDefinition:
    """技能词典条目：标准名称、类别以及可能出现的别名。"""

    canonical: str
    category: str
    aliases: tuple[str, ...]


SKILL_DEFINITIONS: tuple[SkillDefinition, ...] = (
    SkillDefinition("Python", "programming_language", ("python",)),
    SkillDefinition("Java", "programming_language", ("java",)),
    SkillDefinition("Go", "programming_language", ("golang", "go")),
    SkillDefinition("C++", "programming_language", ("c++",)),
    SkillDefinition("R", "programming_language", ("r语言",)),
    SkillDefinition("JavaScript", "programming_language", ("javascript", "js")),
    SkillDefinition("TypeScript", "programming_language", ("typescript", "ts")),
    SkillDefinition("NumPy", "ai_ml", ("numpy",)),
    SkillDefinition("Pandas", "ai_ml", ("pandas",)),
    SkillDefinition("PyTorch", "ai_ml", ("pytorch",)),
    SkillDefinition("TensorFlow", "ai_ml", ("tensorflow",)),
    SkillDefinition("scikit-learn", "ai_ml", ("scikit-learn", "sklearn")),
    SkillDefinition("OpenCV", "ai_ml", ("opencv",)),
    SkillDefinition("CNN", "ai_ml", ("卷积神经网络", "cnn")),
    SkillDefinition("Flask", "framework", ("flask",)),
    SkillDefinition("FastAPI", "framework", ("fastapi",)),
    SkillDefinition("Django", "framework", ("django",)),
    SkillDefinition("Spring Boot", "framework", ("spring boot", "springboot")),
    SkillDefinition("JPA", "framework", ("jpa",)),
    SkillDefinition("Node.js", "framework", ("node.js", "nodejs")),
    SkillDefinition("React", "framework", ("react",)),
    SkillDefinition("Vue", "framework", ("vue",)),
    SkillDefinition("SQL", "database", ("sql",)),
    SkillDefinition("MySQL", "database", ("mysql",)),
    SkillDefinition("SQLite", "database", ("sqlite",)),
    SkillDefinition("PostgreSQL", "database", ("postgresql", "postgres")),
    SkillDefinition("Redis", "database", ("redis",)),
    SkillDefinition("Neo4j", "database", ("neo4j",)),
    SkillDefinition("Git", "tool", ("git",)),
    SkillDefinition("Docker", "tool", ("docker",)),
    SkillDefinition("Kubernetes", "platform", ("kubernetes", "k8s")),
    SkillDefinition("Linux", "platform", ("linux",)),
    SkillDefinition("CUDA", "platform", ("cuda",)),
    SkillDefinition("pytest", "tool", ("pytest",)),
    SkillDefinition("Postman", "tool", ("postman",)),
    SkillDefinition("Tableau", "tool", ("tableau",)),
    SkillDefinition("Excel", "tool", ("excel",)),
    SkillDefinition("Hadoop", "platform", ("hadoop",)),
    SkillDefinition("API testing", "tool", ("接口测试", "api测试", "api 测试", "api testing")),
    SkillDefinition("data analysis", "domain_knowledge", ("数据分析", "data analysis")),
    SkillDefinition("data visualization", "tool", ("数据可视化", "结果可视化", "可视化")),
    SkillDefinition("LLM", "ai_ml", ("大语言模型", "llm")),
    SkillDefinition("LLM API", "ai_ml", ("大语言模型api", "模型api", "llm api")),
    SkillDefinition("Prompt Engineering", "ai_ml", ("prompt engineering", "prompt")),
    SkillDefinition("RAG", "ai_ml", ("rag", "检索增强生成")),
    SkillDefinition("Agent", "ai_ml", ("agent", "智能体")),
    SkillDefinition("LangChain", "framework", ("langchain",)),
    SkillDefinition("LangGraph", "framework", ("langgraph",)),
    SkillDefinition("REST API", "framework", ("rest api", "restful api", "web api")),
    SkillDefinition("Transformers", "ai_ml", ("transformers",)),
    SkillDefinition("Whisper", "ai_ml", ("whisper",)),
    SkillDefinition("BERT", "ai_ml", ("bert",)),
    SkillDefinition("GPT", "ai_ml", ("gpt",)),
    SkillDefinition("LoRA", "ai_ml", ("lora",)),
    SkillDefinition("PEFT", "ai_ml", ("peft",)),
    SkillDefinition("DeepSpeed", "ai_ml", ("deepspeed",)),
    SkillDefinition("PCA", "ai_ml", ("pca",)),
    SkillDefinition("XGBoost", "ai_ml", ("xgboost",)),
    SkillDefinition("Random Forest", "ai_ml", ("随机森林", "random forest")),
    SkillDefinition("TensorRT", "ai_ml", ("tensorrt",)),
    SkillDefinition("ONNX", "ai_ml", ("onnx",)),
    SkillDefinition("vLLM", "ai_ml", ("vllm",)),
    SkillDefinition("Ray", "platform", ("ray",)),
    SkillDefinition("ViT", "ai_ml", ("vit", "vision transformer")),
    SkillDefinition("CLIP", "ai_ml", ("clip",)),
    SkillDefinition("fastai", "ai_ml", ("fastai",)),
    SkillDefinition("Adapter", "ai_ml", ("adapter",)),
    SkillDefinition("Kafka", "platform", ("kafka",)),
    SkillDefinition("Spark", "platform", ("spark",)),
    SkillDefinition("Flink", "platform", ("flink",)),
    SkillDefinition("Prometheus", "tool", ("prometheus",)),
    SkillDefinition("Grafana", "tool", ("grafana",)),
    SkillDefinition("Jenkins", "tool", ("jenkins",)),
    SkillDefinition("Shell", "programming_language", ("shell",)),
    SkillDefinition("software testing", "domain_knowledge", ("软件测试",)),
    SkillDefinition("computer vision", "domain_knowledge", ("计算机视觉", "图像分类", "目标检测")),
    SkillDefinition("natural language processing", "domain_knowledge", ("自然语言处理", "nlp")),
    SkillDefinition("speech recognition", "domain_knowledge", ("语音识别", "asr")),
    SkillDefinition("multimodal learning", "domain_knowledge", ("多模态学习", "多模态")),
    SkillDefinition("recommendation system", "domain_knowledge", ("推荐系统",)),
    SkillDefinition("data engineering", "domain_knowledge", ("数据工程", "data engineering")),
    SkillDefinition("cloud native", "domain_knowledge", ("云原生", "cloud native")),
    SkillDefinition("paper reading", "soft_skill", ("论文阅读", "阅读论文")),
    SkillDefinition("data communication", "soft_skill", ("数据表达",)),
)


def clean_text(text: str) -> str:
    """统一换行、清除行尾空白和过多空行，但不改写正文。"""
    normalized = (
        text.replace("\ufeff", "")
        .replace("\u200b", "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )
    lines = [re.sub(r"[ \t]+$", "", line) for line in normalized.split("\n")]
    cleaned = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", cleaned)


def iter_located_lines(text: str) -> list[LocatedLine]:
    """扫描 Markdown，并将每行绑定到最近的二级标题。"""
    result: list[LocatedLine] = []
    section = ""
    offset = 0
    for line_number, line in enumerate(text.splitlines(keepends=True), start=1):
        raw = line.rstrip("\n")
        heading = re.match(r"^##\s+(.+?)\s*$", raw)
        if heading:
            section = heading.group(1).strip()
        result.append(
            LocatedLine(
                section=section,
                text=raw,
                line_number=line_number,
                start_char=offset,
                end_char=offset + len(raw),
            )
        )
        offset += len(line)
    return result


def section_lines(text: str, aliases: set[str]) -> list[LocatedLine]:
    """返回章节名匹配任一别名的所有带位置文本行。"""
    wanted = {normalize_heading(alias) for alias in aliases}
    return [
        line
        for line in iter_located_lines(text)
        if normalize_heading(line.section) in wanted
    ]


def normalize_heading(value: str) -> str:
    """移除标题中的排版符号并转小写，便于中英文标题比较。"""
    return re.sub(r"[\s：:（）()_-]", "", value).lower()


def content_text(line: LocatedLine) -> str:
    """移除 Markdown 列表/标题标记，返回证据正文。"""
    return re.sub(
        r"^\s*(?:[-*+•●▪◦]\s*|#{1,6}\s+|\d+[.)、]\s*)",
        "",
        line.text,
    ).strip()


def content_lines(lines: list[LocatedLine]) -> list[LocatedLine]:
    """过滤空行和二级标题，只保留章节中的实际内容行。"""
    return [line for line in lines if content_text(line) and not line.text.startswith("## ")]


def first_h1(text: str) -> LocatedLine | None:
    """寻找第一个 Markdown 一级标题，简历中通常用于表示姓名。"""
    for line in iter_located_lines(text):
        if re.match(r"^#\s+[^#]", line.text):
            return line
    return None


def _alias_present(alias: str, lowered_text: str) -> bool:
    """判断技能别名是否出现，并避免短别名造成子串误匹配。"""
    escaped = re.escape(alias.lower())
    if re.fullmatch(r"[a-z0-9+#. -]+", alias.lower()):
        # 短别名必须是独立 token，避免 js 在 Node.js 中、r 在 RAG 中误命中。
        if len(alias) <= 2 and alias.lower() not in {"c++"}:
            return bool(
                re.search(
                    rf"(?<![a-z0-9_.+#-]){escaped}(?![a-z0-9_.+#-])",
                    lowered_text,
                )
            )
        return bool(re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", lowered_text))
    return alias.lower() in lowered_text


def extract_skills(text: str) -> list[SkillDefinition]:
    """按词表顺序提取技能并去重，避免 Python/python 产生两项。"""
    lowered = re.sub(r"\s+", " ", text.lower())
    result: list[SkillDefinition] = []
    for definition in SKILL_DEFINITIONS:
        if any(_alias_present(alias, lowered) for alias in definition.aliases):
            result.append(definition)
    return result


def split_list(value: str) -> list[str]:
    """切分中文枚举，同时保留技术名内部的连字符等符号。"""
    return [item.strip() for item in re.split(r"[、，,；;]|\s+/\s+", value) if item.strip()]


ValueT = TypeVar("ValueT")


def unique(items: list[ValueT]) -> list[ValueT]:
    """按原始顺序去重，字符串比较不区分大小写。"""
    seen: set[str] = set()
    result: list[ValueT] = []
    for item in items:
        key = str(item).casefold()
        if item and key not in seen:
            seen.add(key)
            result.append(item)
    return result
