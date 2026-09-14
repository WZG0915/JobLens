# JobLens

JobLens 是一个面向实习求职场景的学习型 AI Agent 项目。它将逐步实现岗位 JD 解析、简历证据匹配、面试准备、岗位管理、RAG 检索和 Agent 评测。

当前进度：**阶段 0——环境、目录和模拟数据准备完成**。

## 阶段 0 已包含

- 标准 Python 项目目录
- 一份完全虚构的示例简历
- 五份完全虚构的实习岗位 JD
- 安全的环境变量模板和 `.gitignore`
- 无需 API Key 的最小启动程序
- 基础 smoke test

## 运行环境

- Conda 环境：`ai_env`
- Python：3.10.20
- 阶段 0 不依赖任何第三方 Python 包

## 快速开始（PowerShell）

```powershell
cd E:\Ai_agent\project1
conda activate ai_env
python app.py
python -m unittest discover -s tests -v
```

如果当前终端尚未初始化 Conda，可以不激活，直接使用 `ai_env` 中的解释器：

```powershell
E:\Anaconda3\envs\ai_env\python.exe app.py
E:\Anaconda3\envs\ai_env\python.exe -m unittest discover -s tests -v
```

说明：这台 Windows 机器上的 `conda run` 在转发中文输出时可能触发 GBK 编码错误，因此备用命令直接调用环境中的 Python；它仍然使用 `ai_env`，不是系统 Python。

## 隐私和安全

- `.env` 已被 Git 忽略，禁止在代码中硬编码 API Key。
- `data/resumes/sample_resume.md` 是虚构数据，可以进入公开仓库。
- 真实简历请放入 `data/private/`，该目录已被 Git 忽略。
- 公开代码前仍应运行 `git status --ignored` 检查敏感文件。

## 下一阶段

阶段 1 将实现：

1. 简历结构化解析。
2. 岗位 JD 结构化解析。
3. 岗位要求与简历证据匹配。
4. 固定 JSON 结果和 Pydantic 校验。
