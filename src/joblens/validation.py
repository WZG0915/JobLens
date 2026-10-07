"""Pydantic 结构化输出的统一验证入口。"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from joblens.exceptions import StructuredOutputError


ModelT = TypeVar("ModelT", bound=BaseModel)


def validate_payload(
    model_type: type[ModelT], payload: Any, context: str = "数据"
) -> ModelT:
    """验证 Python/JSON 数据，并返回简洁且带上下文的错误。"""
    try:
        if isinstance(payload, str):
            return model_type.model_validate_json(payload)
        return model_type.model_validate(payload)
    except (ValidationError, ValueError, TypeError) as exc:
        raise StructuredOutputError(f"{context}的结构化输出无效：{exc}") from exc

