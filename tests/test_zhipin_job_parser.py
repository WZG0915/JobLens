"""BOSS直聘单页手动导入适配器测试；不访问网络。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from joblens.parsers import (  # noqa: E402
    build_zhipin_search_url,
    parse_zhipin_export,
)
from joblens.schemas import SalaryPeriod  # noqa: E402


class ZhipinJobParserTests(unittest.TestCase):
    """验证搜索链接构造、来源限制和本地 HTML 导入。"""

    def test_build_search_url_encodes_query(self) -> None:
        """中文搜索词应被正确编码到 BOSS直聘官方链接。"""
        url = build_zhipin_search_url("Python 实习生")
        self.assertEqual(
            url,
            "https://www.zhipin.com/c100010000/?query=Python+%E5%AE%9E%E4%B9%A0%E7%94%9F",
        )

    def test_rejects_non_zhipin_source_url(self) -> None:
        """手动导入器应拒绝非 BOSS直聘域名的来源地址。"""
        fixture = PROJECT_ROOT / "tests" / "fixtures" / "zhipin_job_detail.html"
        with self.assertRaises(ValueError):
            parse_zhipin_export(fixture, source_url="https://example.com/job/1")

    def test_parse_saved_job_detail_html(self) -> None:
        """本地保存的详情页应被解析为完整岗位结构。"""
        fixture = PROJECT_ROOT / "tests" / "fixtures" / "zhipin_job_detail.html"
        source_url = "https://www.zhipin.com/job_detail/example.html"
        result = parse_zhipin_export(fixture, source_url=source_url)

        self.assertTrue(result.success)
        self.assertIsNotNone(result.data)
        assert result.data is not None
        basic = result.data.basic_information
        self.assertEqual(basic.title, "Python开发实习生")
        self.assertEqual(result.data.company.name, "示例智能科技（虚构）")
        self.assertEqual(result.data.company.industry, "人工智能")
        self.assertEqual(result.data.company.company_size, "20-99人")
        self.assertEqual(result.data.company.financing_stage, "天使轮")
        self.assertEqual(basic.cities, ["杭州市"])
        self.assertIn("余杭区", basic.address or "")
        self.assertEqual(basic.published_date, "2026-09-01")
        self.assertEqual(basic.source_url, source_url)
        self.assertEqual(result.metadata.source_url, source_url)
        self.assertIsNotNone(basic.salary)
        assert basic.salary is not None
        self.assertEqual(basic.salary.minimum, 180)
        self.assertEqual(basic.salary.maximum, 250)
        self.assertEqual(basic.salary.period, SalaryPeriod.DAILY)
        self.assertGreaterEqual(len(result.data.responsibilities), 2)
        self.assertGreaterEqual(len(result.data.requirements), 3)
        self.assertEqual(len(result.data.benefits), 2)
        self.assertIsNotNone(result.data.availability)
        assert result.data.availability is not None
        self.assertEqual(result.data.availability.minimum_days_per_week, 4)
        self.assertEqual(result.data.availability.minimum_months, 3)
        self.assertTrue(result.data.availability.source)
        requirement_keywords = {
            keyword
            for requirement in result.data.requirements
            for keyword in requirement.normalized_keywords
        }
        self.assertIn("Python", requirement_keywords)
        self.assertIn("MySQL", requirement_keywords)


if __name__ == "__main__":
    unittest.main()
