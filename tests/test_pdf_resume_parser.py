"""中文 PDF 简历解析的端到端回归测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from joblens.parsers import parse_resume_document  # noqa: E402
from joblens.exceptions import DocumentReadError  # noqa: E402
from joblens.io import read_pdf_document  # noqa: E402


RESUME_DIR = PROJECT_ROOT / "data" / "resumes"


class PdfResumeParserTests(unittest.TestCase):
    """验证 PDF 文本层、OCR 回退、字段解析和来源范围。"""

    def test_scanned_pdf_without_text_layer_has_actionable_error(self) -> None:
        """扫描 PDF 且 OCR 不可用时应返回可操作错误。"""
        class EmptyPage:
            """模拟没有文本层的单页 PDF。"""

            def dedupe_chars(self, tolerance: int):
                """模拟 pdfplumber 页面去重接口。"""
                return self

            def extract_text(self, **kwargs):
                """返回空字符串，表示页面没有可提取文字。"""
                return ""

        class EmptyPdf:
            """模拟可作为上下文管理器打开的单页 PDF。"""

            pages = [EmptyPage()]

            def __enter__(self):
                """进入上下文时返回模拟 PDF 自身。"""
                return self

            def __exit__(self, exc_type, exc, traceback):
                """退出上下文且不吞掉异常。"""
                return False

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scanned.pdf"
            path.write_bytes(b"%PDF-test")
            with patch("pdfplumber.open", return_value=EmptyPdf()), patch(
                "joblens.io._ocr_pdf_page", return_value=None
            ):
                with self.assertRaisesRegex(DocumentReadError, "扫描件"):
                    read_pdf_document(path)

    def test_scanned_pdf_uses_ocr_fallback_when_available(self) -> None:
        """文本层为空但 OCR 可用时应成功提取文字。"""
        class EmptyPage:
            """再次模拟无文本层页面，用于验证 OCR 成功分支。"""

            def dedupe_chars(self, tolerance: int):
                """模拟 pdfplumber 页面去重接口。"""
                return self

            def extract_text(self, **kwargs):
                """返回空文本以触发 OCR 回退。"""
                return ""

        class EmptyPdf:
            """为 OCR 成功分支提供最小 PDF 上下文管理器。"""

            pages = [EmptyPage()]

            def __enter__(self):
                """进入上下文时返回模拟 PDF 自身。"""
                return self

            def __exit__(self, exc_type, exc, traceback):
                """退出上下文且不吞掉异常。"""
                return False

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scanned.pdf"
            path.write_bytes(b"%PDF-test")
            with patch("pdfplumber.open", return_value=EmptyPdf()), patch(
                "joblens.io._ocr_pdf_page",
                return_value="张三\n邮箱：zhangsan@example.com\n求职意向：Python 工程师",
            ):
                result = read_pdf_document(path)
        self.assertIn("张三", result.text)
        self.assertEqual(result.extraction_method, "ocr")
        self.assertEqual(result.ocr_page_numbers, (1,))

    def test_all_pdf_samples_have_text_and_expected_page_counts(self) -> None:
        """五份 PDF 样本都应提取足够文本并识别正确页数。"""
        expected_pages = {
            "01_cn_standard_single_page.pdf": 1,
            "02_cn_dense_two_page.pdf": 2,
            "03_cn_data_analyst_single_page.pdf": 1,
            "04_cn_backend_with_avatar.pdf": 1,
            "05_cn_ai_researcher_two_page.pdf": 2,
        }
        for name, page_count in expected_pages.items():
            with self.subTest(name=name):
                result = parse_resume_document(RESUME_DIR / name)
                self.assertTrue(result.success)
                self.assertIsNotNone(result.data)
                self.assertEqual(result.metadata.source_file_type, "pdf")
                self.assertEqual(result.metadata.page_count, page_count)
                self.assertIn(
                    result.metadata.extraction_method,
                    {"pdfplumber_layout", "pdfplumber_text_layer", "pdfplumber_layout+ocr"},
                )
                self.assertGreater(result.metadata.character_count, 500)
                self.assertEqual(result.metadata.empty_page_numbers, [])

    def test_data_analyst_resume_extracts_core_fields(self) -> None:
        """数据分析简历应识别个人信息、经历、项目和核心技能。"""
        result = parse_resume_document(RESUME_DIR / "03_cn_data_analyst_single_page.pdf")
        assert result.data is not None
        data = result.data
        self.assertEqual(data.basic_information.name, "陈思雨")
        self.assertEqual(data.basic_information.contact.email, "chensiyu@example.com")
        self.assertIn("上海", data.basic_information.contact.city or "")
        self.assertIn("数据分析师", data.job_preference.target_roles if data.job_preference else [])
        self.assertGreaterEqual(len(data.work_experiences), 2)
        self.assertTrue(any("电商用户" in item.name for item in data.projects))
        skill_names = {skill.normalized_name for skill in data.skills}
        self.assertTrue({"Python", "MySQL", "Tableau"}.issubset(skill_names))

    def test_backend_resume_extracts_work_projects_and_languages(self) -> None:
        """后端简历应识别工作、项目、技术栈和语言考试。"""
        result = parse_resume_document(RESUME_DIR / "04_cn_backend_with_avatar.pdf")
        assert result.data is not None
        data = result.data
        self.assertGreaterEqual(len(data.work_experiences), 2)
        self.assertGreaterEqual(len(data.projects), 2)
        skill_names = {skill.normalized_name for skill in data.skills}
        self.assertTrue({"Spring Boot", "MySQL", "Docker"}.issubset(skill_names))
        examinations = {item.examination for item in data.languages}
        self.assertTrue(any(item and "CET-6" in item for item in examinations))

    def test_research_resume_extracts_multi_page_sections(self) -> None:
        """双页科研简历应恢复多段工作、教育和论文。"""
        result = parse_resume_document(RESUME_DIR / "05_cn_ai_researcher_two_page.pdf")
        assert result.data is not None
        data = result.data
        self.assertEqual(data.basic_information.name, "无名氏")
        self.assertGreaterEqual(len(data.work_experiences), 4)
        self.assertGreaterEqual(len(data.education), 2)
        self.assertGreaterEqual(len(data.publications), 4)
        self.assertIn("RAG", {skill.normalized_name for skill in data.skills})

    def test_evidence_ranges_still_point_to_normalized_pdf_text(self) -> None:
        """PDF 标准化后技能证据范围仍应精确指向原文。"""
        result = parse_resume_document(RESUME_DIR / "01_cn_standard_single_page.pdf")
        assert result.data is not None
        self.assertEqual(result.data.job_preference.target_roles, ["AI Agent Engineer"])
        self.assertEqual(
            [(item.school, item.major) for item in result.data.education],
            [("星河大学", "计算机科学与技术"), ("云杉实验室", "人机交互与软件工程")],
        )
        for skill in result.data.skills:
            for source in skill.source:
                if source.start_char is not None and source.end_char is not None:
                    self.assertEqual(
                        result.data.raw_text[source.start_char : source.end_char],
                        source.text,
                    )

    def test_dense_pdf_does_not_use_template_marker_as_company(self) -> None:
        """密集 PDF 中的模板标记不能被误识别为公司名称。"""
        result = parse_resume_document(RESUME_DIR / "02_cn_dense_two_page.pdf")
        assert result.data is not None
        organizations = {item.organization for item in result.data.work_experiences}
        self.assertNotIn("Mock", organizations)
        self.assertTrue(any("字节跳动" in item for item in organizations))
        self.assertTrue(all(item.position for item in result.data.work_experiences))
        self.assertEqual(result.data.education[0].school, "东海科技大学")
        self.assertEqual(result.data.education[0].major, "人工智能实验班")
        self.assertEqual(result.data.education[0].ranking, "6/120")


if __name__ == "__main__":
    unittest.main()
