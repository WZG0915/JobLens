"""JobLens 简历与岗位 JD 共用的数据模型。"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictBaseModel(BaseModel):
    """所有结构化数据模型的公共基类。

    禁止未声明字段进入模型，避免拼写错误或大模型生成的额外字段
    被系统静默接受。
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class SourceReference(StrictBaseModel):
    """结构化字段在原始简历或 JD 中的可追溯位置。"""

    section: str = Field(
        default="",
        description="来源章节，例如：专业技能、项目经历、任职要求",
    )

    text: str = Field(
        min_length=1,
        description="未经改写的原始文本证据",
    )

    start_char: Optional[int] = Field(
        default=None,
        ge=0,
        description="证据在完整原文中的起始字符位置",
    )

    end_char: Optional[int] = Field(
        default=None,
        ge=0,
        description="证据在完整原文中的结束字符位置",
    )

    line_number: Optional[int] = Field(
        default=None,
        ge=1,
        description="证据在原文中的行号",
    )

    @model_validator(mode="after")
    def validate_character_range(self) -> "SourceReference":
        """同时提供字符范围时，结束位置必须晚于开始位置。"""
        if (
            self.start_char is not None
            and self.end_char is not None
            and self.end_char <= self.start_char
        ):
            raise ValueError("end_char 必须大于 start_char")
        return self


class DateRange(StrictBaseModel):
    """简历或岗位信息中的时间范围。

    简历通常只提供年月，也可能出现“至今”等表达，因此这里保留
    标准化字符串和原始日期文本，而不强制转换为 Python date。
    """

    start_date: Optional[str] = Field(
        default=None,
        description="标准化开始时间，推荐格式：YYYY-MM",
    )

    end_date: Optional[str] = Field(
        default=None,
        description="标准化结束时间，推荐格式：YYYY-MM",
    )

    is_current: bool = Field(
        default=False,
        description="当前是否仍在进行",
    )

    raw_text: str = Field(
        default="",
        description="原始日期表达，例如：2024.03—至今",
    )
