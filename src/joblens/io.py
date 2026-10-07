"""文档读取边界，负责安全读取文本和带文本层的 PDF。"""

from __future__ import annotations

import re
import unicodedata
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from joblens.exceptions import DocumentReadError


SUPPORTED_TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
SUPPORTED_RESUME_SUFFIXES = SUPPORTED_TEXT_SUFFIXES | {".pdf"}
MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_PDF_PAGES = 10
MIN_PDF_TEXT_CHARACTERS = 20
MIN_PAGE_TEXT_CHARACTERS = 12


@dataclass(frozen=True)
class PdfDocumentText:
    """PDF 文本抽取结果，包含版面与 OCR 回退信息。"""

    text: str
    page_count: int
    empty_page_numbers: tuple[int, ...]
    extraction_method: str = "pdfplumber_layout"
    ocr_page_numbers: tuple[int, ...] = ()
    layout_page_numbers: tuple[int, ...] = ()


def _normalize_pdf_text(value: str) -> str:
    """统一 PDF 文本中的全角字符、兼容字符和不可见空字符。"""
    normalized = unicodedata.normalize("NFKC", value or "")
    return normalized.replace("\x00", "").strip()


def _visible_character_count(value: str) -> int:
    """统计去除空白后的有效字符数，用于判断文本层是否可用。"""
    return len(re.sub(r"\s+", "", value))


def _extract_layout_text(page: Any) -> str:
    """使用单词坐标恢复视觉行；不支持坐标的页面回退到普通文本层。"""
    try:
        words = page.extract_words(
            x_tolerance=2,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=False,
        )
    except (AttributeError, TypeError):
        words = []
    except Exception:
        words = []
    if not words:
        return ""

    clean_words = [
        word
        for word in words
        if str(word.get("text", "")).strip()
        and word.get("top") is not None
        and word.get("x0") is not None
    ]
    if not clean_words:
        return ""
    clean_words.sort(key=lambda item: (float(item["top"]), float(item["x0"])))

    visual_lines: list[list[dict[str, Any]]] = []
    line_tops: list[float] = []
    for word in clean_words:
        top = float(word["top"])
        target_index = next(
            (index for index, value in enumerate(line_tops) if abs(value - top) <= 5.0),
            None,
        )
        if target_index is None:
            visual_lines.append([word])
            line_tops.append(top)
        else:
            visual_lines[target_index].append(word)
            count = len(visual_lines[target_index])
            line_tops[target_index] = ((line_tops[target_index] * (count - 1)) + top) / count

    ordered = sorted(zip(line_tops, visual_lines), key=lambda item: item[0])
    rendered_lines: list[str] = []
    for _, line_words in ordered:
        line_words.sort(key=lambda item: float(item["x0"]))
        pieces: list[str] = []
        previous_x1: float | None = None
        for word in line_words:
            text = str(word["text"]).strip()
            x0 = float(word["x0"])
            if previous_x1 is not None and x0 - previous_x1 >= 48:
                pieces.append(" | ")
            elif pieces:
                pieces.append(" ")
            pieces.append(text)
            previous_x1 = float(word.get("x1", x0))
        rendered_lines.append("".join(pieces).strip())
    return _normalize_pdf_text("\n".join(rendered_lines))


def _ocr_pdf_page(document_path: Path, page_number: int) -> str | None:
    """可选 OCR 回退；依赖 Poppler、Tesseract、Pillow 和 pytesseract。"""
    pdftoppm = shutil.which("pdftoppm")
    configured_tesseract = os.getenv("JOBLENS_TESSERACT_CMD")
    tesseract = configured_tesseract or shutil.which("tesseract")
    if not pdftoppm or not tesseract:
        return None
    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        return None

    try:
        pytesseract.pytesseract.tesseract_cmd = tesseract
        with tempfile.TemporaryDirectory(prefix="joblens_ocr_") as temp_dir:
            output_prefix = Path(temp_dir) / f"page_{page_number}"
            completed = subprocess.run(
                [
                    pdftoppm,
                    "-f",
                    str(page_number),
                    "-l",
                    str(page_number),
                    "-singlefile",
                    "-r",
                    "240",
                    "-png",
                    str(document_path),
                    str(output_prefix),
                ],
                capture_output=True,
                check=False,
                timeout=60,
            )
            image_path = output_prefix.with_suffix(".png")
            if completed.returncode != 0 or not image_path.exists():
                return None
            languages = os.getenv("JOBLENS_OCR_LANG", "chi_sim+eng")
            try:
                text = pytesseract.image_to_string(Image.open(image_path), lang=languages)
            except Exception:
                text = pytesseract.image_to_string(Image.open(image_path), lang="eng")
            normalized = _normalize_pdf_text(text)
            return normalized if _visible_character_count(normalized) >= MIN_PAGE_TEXT_CHARACTERS else None
    except (OSError, subprocess.SubprocessError):
        return None


def _validate_document_path(path: str | Path) -> Path:
    """确认输入路径存在且指向文件，并转换为 Path 对象。"""
    document_path = Path(path).expanduser()
    if not document_path.exists():
        raise DocumentReadError(f"文件不存在：{document_path}")
    if not document_path.is_file():
        raise DocumentReadError(f"路径不是文件：{document_path}")
    return document_path


def read_pdf_document(path: str | Path, *, enable_ocr: bool = True) -> PdfDocumentText:
    """按坐标读取 PDF；文本层不足时自动尝试本地 OCR。"""
    document_path = _validate_document_path(path)
    if document_path.suffix.lower() != ".pdf":
        raise DocumentReadError(f"文件不是 PDF：{document_path}")
    try:
        file_size = document_path.stat().st_size
    except OSError as exc:
        raise DocumentReadError(f"无法读取文件信息 {document_path}：{exc}") from exc
    if file_size <= 0:
        raise DocumentReadError(f"文件为空：{document_path}")
    if file_size > MAX_PDF_BYTES:
        raise DocumentReadError(
            f"PDF 大小超过 {MAX_PDF_BYTES // (1024 * 1024)} MB 限制：{document_path}"
        )

    try:
        import pdfplumber
    except ImportError as exc:
        raise DocumentReadError(
            "读取 PDF 需要 pdfplumber，请执行：pip install -e ."
        ) from exc

    # 部分合法 PDF 缺少可选的 FontBBox；pdfminer 会重复输出无害告警。
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    logging.getLogger("pdfminer.pdffont").setLevel(logging.ERROR)

    page_texts: list[str] = []
    empty_pages: list[int] = []
    ocr_pages: list[int] = []
    layout_pages: list[int] = []
    try:
        with pdfplumber.open(document_path) as pdf:
            page_count = len(pdf.pages)
            if page_count == 0:
                raise DocumentReadError(f"PDF 不包含页面：{document_path}")
            if page_count > MAX_PDF_PAGES:
                raise DocumentReadError(
                    f"PDF 页数超过 {MAX_PDF_PAGES} 页限制：{page_count} 页"
                )
            for page_number, page in enumerate(pdf.pages, start=1):
                # dedupe_chars 可去除部分软件导出时产生的重叠文字副本。
                try:
                    page = page.dedupe_chars(tolerance=1)
                except Exception:
                    pass
                layout_text = _extract_layout_text(page)
                try:
                    plain_text = _normalize_pdf_text(
                        page.extract_text(x_tolerance=2, y_tolerance=3, layout=False) or ""
                    )
                except Exception:
                    plain_text = ""
                # 坐标文本信息量接近普通文本时优先使用，避免内容流顺序打乱字段。
                if (
                    _visible_character_count(layout_text) >= MIN_PAGE_TEXT_CHARACTERS
                    and _visible_character_count(layout_text)
                    >= int(_visible_character_count(plain_text) * 0.82)
                ):
                    extracted = layout_text
                    layout_pages.append(page_number)
                else:
                    extracted = plain_text
                if _visible_character_count(extracted) < MIN_PAGE_TEXT_CHARACTERS and enable_ocr:
                    ocr_text = _ocr_pdf_page(document_path, page_number)
                    if ocr_text:
                        extracted = ocr_text
                        ocr_pages.append(page_number)
                if _visible_character_count(extracted) < MIN_PAGE_TEXT_CHARACTERS:
                    empty_pages.append(page_number)
                page_texts.append(extracted)
    except DocumentReadError:
        raise
    except Exception as exc:
        message = str(exc).lower()
        if "password" in message or "encrypt" in message:
            detail = "PDF 已加密或需要密码"
        else:
            detail = "PDF 文件损坏、格式异常或无法读取文本层"
        raise DocumentReadError(f"{detail}：{document_path}（{exc}）") from exc

    text = "\n\n".join(part for part in page_texts if part).strip()
    visible_character_count = _visible_character_count(text)
    if visible_character_count < MIN_PDF_TEXT_CHARACTERS:
        raise DocumentReadError(
            "PDF 未检测到足够的可提取文本，可能是扫描件，且本地 OCR 不可用或识别失败；"
            "请安装 Tesseract 与 pytesseract（可通过 JOBLENS_TESSERACT_CMD 指定路径），"
            "或上传带文本层的 PDF"
        )
    if ocr_pages and layout_pages:
        extraction_method = "pdfplumber_layout+ocr"
    elif ocr_pages:
        extraction_method = "ocr"
    elif layout_pages:
        extraction_method = "pdfplumber_layout"
    else:
        extraction_method = "pdfplumber_text_layer"
    return PdfDocumentText(
        text=text,
        page_count=page_count,
        empty_page_numbers=tuple(empty_pages),
        extraction_method=extraction_method,
        ocr_page_numbers=tuple(ocr_pages),
        layout_page_numbers=tuple(layout_pages),
    )


def read_document(path: str | Path) -> str:
    """读取 UTF-8 文本或 PDF，并把底层错误转换为领域异常。"""
    document_path = _validate_document_path(path)
    suffix = document_path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf_document(document_path).text
    if suffix not in SUPPORTED_TEXT_SUFFIXES:
        supported = "、".join(sorted(SUPPORTED_RESUME_SUFFIXES))
        raise DocumentReadError(
            f"暂不支持 {document_path.suffix or '无扩展名'} 文件；支持：{supported}"
        )
    try:
        text = document_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentReadError(f"文件不是有效的 UTF-8 文本：{document_path}") from exc
    except OSError as exc:
        raise DocumentReadError(f"无法读取文件 {document_path}：{exc}") from exc
    text = text.strip()
    if not text:
        raise DocumentReadError(f"文件为空：{document_path}")
    return text
