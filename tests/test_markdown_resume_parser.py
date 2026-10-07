"""多结构 Markdown 简历的端到端回归测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from joblens.parsers import parse_resume_document  # noqa: E402
from joblens.schemas.resume import ExperienceType  # noqa: E402


RESUME_DIR = PROJECT_ROOT / "data" / "resumes"


class MarkdownResumeParserTests(unittest.TestCase):
    """验证多种 Markdown 排版都能恢复关键简历字段。"""

    def test_standard_backend_resume_has_exact_core_fields(self) -> None:
        """标准后端简历应准确识别核心字段且避免短技能误匹配。"""
        result = parse_resume_document(RESUME_DIR / "06_cn_backend_standard.md")
        assert result.data is not None
        data = result.data
        self.assertEqual(data.basic_information.name, "周子航")
        self.assertEqual(data.job_preference.target_roles, ["Java 后端开发工程师"])
        self.assertEqual(len(data.education), 1)
        self.assertEqual(data.education[0].school, "江南信息大学")
        self.assertEqual(data.education[0].major, "软件工程")
        self.assertEqual(data.education[0].ranking, "前 10%")
        self.assertEqual(data.work_experiences[0].organization, "云启科技有限公司")
        self.assertEqual(data.projects[0].name, "智慧订单履约平台")
        self.assertNotIn("JavaScript", {item.normalized_name for item in data.skills})

    def test_research_resume_parses_publication_authors(self) -> None:
        """科研简历应解析论文标题和作者。"""
        result = parse_resume_document(RESUME_DIR / "07_cn_ai_researcher.md")
        assert result.data is not None
        data = result.data
        self.assertEqual(data.education[0].major, "人工智能")
        self.assertEqual(data.education[0].ranking, "3/82")
        self.assertEqual(data.publications[0].authors, ["林若曦、陈明远"])
        skill_names = {item.normalized_name for item in data.skills}
        self.assertIn("natural language processing", skill_names)
        self.assertIn("multimodal learning", skill_names)

    def test_markdown_tables_are_normalized_to_records(self) -> None:
        """Markdown 表格应被规范化为教育、经历等结构化记录。"""
        result = parse_resume_document(RESUME_DIR / "08_cn_frontend_table.md")
        assert result.data is not None
        data = result.data
        self.assertEqual(len(data.education), 1)
        self.assertEqual(data.education[0].school, "海川大学")
        self.assertEqual(data.education[0].major, "计算机科学与技术")
        self.assertEqual(data.education[0].gpa, "3.68/4.0")
        self.assertEqual(len(data.work_experiences), 1)
        self.assertEqual(data.work_experiences[0].organization, "云帆网络有限公司")
        self.assertEqual(len(data.projects), 1)
        self.assertEqual(data.projects[0].name, "FlowBoard 低代码工作台")

    def test_mixed_section_classifies_each_entry(self) -> None:
        """混合经历章节中的工作和项目条目应分别归类。"""
        result = parse_resume_document(RESUME_DIR / "09_cn_devops_mixed.md")
        assert result.data is not None
        data = result.data
        self.assertEqual(len(data.work_experiences), 2)
        by_org = {item.organization: item for item in data.work_experiences}
        self.assertEqual(by_org["拓云计算有限公司"].experience_type, ExperienceType.FULL_TIME)
        self.assertEqual(by_org["校园开源镜像站"].experience_type, ExperienceType.CAMPUS)
        self.assertEqual([item.name for item in data.projects], ["ClusterLens 云资源巡检平台"])

    def test_bilingual_headings_are_canonicalized(self) -> None:
        """中英文章节标题应被统一到内部标准章节名。"""
        result = parse_resume_document(RESUME_DIR / "10_cn_data_engineer_bilingual.md")
        assert result.data is not None
        data = result.data
        self.assertEqual(data.basic_information.contact.city, "成都")
        self.assertIn("数据工程师", data.job_preference.target_roles)
        self.assertEqual(data.education[0].school, "西陆大学")
        self.assertEqual(data.education[0].major, "数据科学与大数据技术")
        self.assertEqual(data.work_experiences[0].organization, "经纬数据有限公司")
        self.assertEqual(data.projects[0].name, "MetricFlow 数据质量平台")
        self.assertIn("Data Engineering Professional Certificate", {item.name for item in data.certificates})
        self.assertEqual(result.warnings, [])


if __name__ == "__main__":
    unittest.main()
