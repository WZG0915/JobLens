"""简历解析器的可重复字段级评测。"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from joblens.parsers import parse_resume_document
from joblens.schemas.resume import ResumeData


def _normalize(value: Any) -> str:
    """统一大小写、空白和常见分隔符，使字段比较不受排版差异影响。"""
    if value is None:
        return ""
    text = str(getattr(value, "value", value)).casefold().strip()
    return re.sub(r"[\s|｜,，;；]+", "", text)


def _record(values: list[Any]) -> str:
    """把多个字段拼成一个稳定的复合比较键。"""
    return "¦".join(_normalize(value) for value in values)


def _actual_values(data: ResumeData, field: str, expected: Any) -> set[str]:
    """从解析结果中读取指定字段，并转换为可进行集合比较的值。"""
    if field == "basic_information":
        contact = data.basic_information.contact
        source = {
            "name": data.basic_information.name,
            "phone": contact.phone,
            "email": contact.email,
            "city": contact.city,
            "github": contact.github,
        }
        return {_record([key, source.get(key)]) for key in expected}
    if field == "target_roles":
        return {_normalize(value) for value in (data.job_preference.target_roles if data.job_preference else [])}
    if field == "education":
        return {
            _record([item.school, item.major, item.degree])
            for item in data.education
        }
    if field == "work_experiences":
        return {
            _record([item.organization, item.position, item.experience_type])
            for item in data.work_experiences
        }
    if field == "projects":
        return {_normalize(item.name) for item in data.projects}
    if field == "skills":
        return {_normalize(item.normalized_name) for item in data.skills}
    if field == "publications":
        return {_normalize(item.title) for item in data.publications}
    raise ValueError(f"不支持的评测字段：{field}")


def _expected_values(field: str, expected: Any) -> set[str]:
    """把标准答案中的指定字段转换为与实际结果相同的集合格式。"""
    if field == "basic_information":
        return {_record([key, value]) for key, value in expected.items()}
    if field == "target_roles":
        return {_normalize(value) for value in expected}
    if field == "education":
        return {
            _record([item.get("school"), item.get("major"), item.get("degree", "unknown")])
            for item in expected
        }
    if field == "work_experiences":
        return {
            _record(
                [
                    item.get("organization"),
                    item.get("position"),
                    item.get("experience_type", "unknown"),
                ]
            )
            for item in expected
        }
    if field in {"projects", "skills", "publications"}:
        return {_normalize(value) for value in expected}
    raise ValueError(f"不支持的评测字段：{field}")


def _metric(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    """根据 TP、FP、FN 计算 Precision、Recall 和 F1。"""
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def evaluate_resume_corpus(gold_path: str | Path, project_root: str | Path) -> dict[str, Any]:
    """解析标准答案中的全部样本并计算逐字段及总体 P/R/F1。"""
    gold_document = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    root = Path(project_root)
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    samples: list[dict[str, Any]] = []

    for sample in gold_document["samples"]:
        result = parse_resume_document(root / sample["source"])
        if not result.success or result.data is None:
            samples.append({"source": sample["source"], "success": False, "errors": result.errors})
            continue
        field_results: dict[str, Any] = {}
        sample_exact = True
        for field, expected in sample["expected"].items():
            expected_set = _expected_values(field, expected)
            actual_set = _actual_values(result.data, field, expected)
            tp = len(expected_set & actual_set)
            fp = len(actual_set - expected_set)
            fn = len(expected_set - actual_set)
            totals[field][0] += tp
            totals[field][1] += fp
            totals[field][2] += fn
            field_results[field] = {
                **_metric(tp, fp, fn),
                "missing": sorted(expected_set - actual_set),
                "unexpected": sorted(actual_set - expected_set),
            }
            sample_exact = sample_exact and fp == 0 and fn == 0
        samples.append(
            {
                "source": sample["source"],
                "success": True,
                "exact_match": sample_exact,
                "warnings": [warning.code for warning in result.warnings],
                "fields": field_results,
            }
        )

    field_metrics = {
        field: _metric(*counts)
        for field, counts in sorted(totals.items())
    }
    overall_counts = [sum(values[index] for values in totals.values()) for index in range(3)]
    return {
        "schema_version": "1.0",
        "sample_count": len(gold_document["samples"]),
        "successful_samples": sum(bool(item.get("success")) for item in samples),
        "exact_match_samples": sum(bool(item.get("exact_match")) for item in samples),
        "overall": _metric(*overall_counts),
        "fields": field_metrics,
        "samples": samples,
    }
