"""JobLens 阶段 0 冒烟测试应用。

本程序只使用 Python 标准库，用于验证项目是否可以找到并读取
合成的简历数据和岗位 JD 数据。
"""

from pathlib import Path  # 导入标准库中的路径处理工具


# 当前脚本文件所在目录。
# 在本项目中，假定这个脚本位于项目根目录，所以它就是项目根目录。
# __file__：当前 Python 文件路径
# resolve()：转为绝对路径，并解析符号链接
# parent：取所在目录
PROJECT_ROOT = Path(__file__).resolve().parent

# 示例简历路径：项目根目录 / data / resumes / sample_resume.md
SAMPLE_RESUME = PROJECT_ROOT / "data" / "resumes" / "sample_resume.md"

# 模拟岗位 JD 所在目录：项目根目录 / data / jobs
SAMPLE_JOBS_DIR = PROJECT_ROOT / "data" / "jobs"


def load_text(path: Path) -> str:
    """读取 UTF-8 文本文件，并拒绝空文档。"""
    # 以 UTF-8 编码读取文件内容
    # strip() 会去掉首尾空白字符和换行符
    content = path.read_text(encoding="utf-8").strip()

    # 如果去掉首尾空白后内容为空，说明文件没有有效文本
    if not content:
        raise ValueError(f"文件为空：{path}")

    # 返回读取到的有效文本
    return content


def discover_job_files() -> list[Path]:
    """按稳定顺序返回所有模拟 Markdown 岗位 JD。"""
    # glob("*.md") 只查找 SAMPLE_JOBS_DIR 当前层级的 .md 文件
    # 不会递归查找子目录
    # sorted(...) 保证返回顺序稳定，方便测试和展示
    return sorted(SAMPLE_JOBS_DIR.glob("*.md"))


def main() -> None:
    """打印阶段 0 就绪报告。"""
    # 读取示例简历，并验证其非空
    resume = load_text(SAMPLE_RESUME)

    # 查找所有模拟岗位 JD 文件
    job_files = discover_job_files()

    # 冒烟测试约定：必须正好有 5 份岗位 JD
    if len(job_files) != 5:
        raise RuntimeError(f"期望 5 份岗位 JD，实际找到 {len(job_files)} 份。")

    # 逐个读取岗位 JD 文件，确保每个文件都能读取且非空
    # 这里不保存返回值，只做可读性和非空校验
    for job_file in job_files:
        load_text(job_file)

    # 打印阶段 0 成功信息
    print("JobLens 阶段 0 启动成功！")

    # 打印项目根目录
    print(f"项目目录：{PROJECT_ROOT}")

    # 打印示例简历文件名，以及读取后文本的字符数
    # 注意：这里是 strip() 之后的字符数，不是原始文件字节数
    print(f"示例简历：{SAMPLE_RESUME.name}（{len(resume)} 个字符）")

    # 打印岗位 JD 数量
    print(f"模拟岗位：{len(job_files)} 份")

    # 编号列出所有岗位 JD 文件名
    for index, job_file in enumerate(job_files, start=1):
        print(f"  {index}. {job_file.name}")

    # 提示下一步开发任务
    print("下一步：实现岗位 JD 与简历的结构化解析。")


# 仅当该文件被直接运行时，才调用 main()
# 如果该文件被其他模块 import，则不会自动执行 main()
if __name__ == "__main__":
    main()