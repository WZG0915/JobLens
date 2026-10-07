# JobLens

JobLens 是一个面向实习求职场景的学习型 AI Agent 项目。它将逐步实现岗位 JD 解析、简历证据匹配、面试准备、岗位管理、RAG 检索和 Agent 评测。

当前进度：**阶段 2——模型驱动的安全 Agent 核心闭环已完成**。

## 当前已包含

- 标准 Python 项目目录
- 一份完全虚构的示例简历
- 30份AI岗位JD：前10份来自企业官方招聘页，后20份为明确标记的合成测试数据
- 安全的环境变量模板和 `.gitignore`
- 无需 API Key 的最小启动程序
- Pydantic 简历、岗位、匹配结果和报告模型
- 带原文行号/字符范围的富结构简历与岗位解析模型
- 中文单页/双页 PDF 文本层提取、章节恢复与结构化简历解析
- PDF 单词坐标版面恢复，以及可选的本地 Tesseract OCR 回退
- Markdown 表格、双语章节标题和“实习与项目经历”混合章节解析
- 10 份标准答案简历与字段级 Precision/Recall/F1 自动评测
- 本地可复现解析模式
- 可选的 OpenAI Structured Outputs 解析模式
- 固定的证据匹配流程和 JSON 报告
- 简历/JD 独立解析命令，以及阶段 0 与阶段 1 自动测试
- OpenAI Responses API 模型客户端与可替换的客户端协议
- 模型选择工具、本地安全执行、结果回传和最终回答的 Agent 循环
- 最大步数、最大工具调用数、重复调用保护和结构化执行轨迹
- 本地 BGE 中文 Embedding、Chroma 岗位向量库和结构化岗位 Chunk
- 从简历求职意向、技能、项目、经历与教育生成可追溯的多查询
- FastAPI本地对话网页：上传PDF后完成解析、岗位推荐、证据展示和会话清理

## 运行环境

- Conda 环境：`ai_env`
- Python：3.10.20
- 阶段 1 使用 Pydantic、pdfplumber 和 python-dotenv；真实 API 模式使用 OpenAI SDK

## 快速开始（PowerShell）

```powershell
cd E:\Ai_agent\project1
conda activate ai_env
python app.py
python app.py analyze --provider local
python app.py parse-resume
python app.py parse-resume --resume data/resumes/03_cn_data_analyst_single_page.pdf
python app.py parse-job
python app.py match-evidence --resume data/resumes/sample_resume.md --job data/jobs/04_baidu_llm_application_engineer.md
python app.py match-jobs --resume data/resumes/sample_resume.md --jobs-dir data/jobs
python app.py zhipin-search-url --query "Python 实习生"
python app.py parse-zhipin-export --input data/private/zhipin_job.html
python app.py evaluate-resumes
python app.py build-job-index
python app.py generate-resume-queries --resume data/resumes/01_cn_standard_single_page.pdf
python app.py recommend-jobs-rag --resume data/resumes/01_cn_standard_single_page.pdf --local-files-only
python app.py web
python app.py agent --message "请比较 data/resumes/01_cn_standard_single_page.pdf 与 data/jobs/04_baidu_llm_application_engineer.md"
python -m pytest -q
```

如果当前终端尚未初始化 Conda，可以不激活，直接使用 `ai_env` 中的解释器：

```powershell
E:\Anaconda3\envs\ai_env\python.exe app.py
E:\Anaconda3\envs\ai_env\python.exe app.py analyze --provider local
E:\Anaconda3\python.exe -m pytest -q
```

说明：这台 Windows 机器上的 `conda run` 在转发中文输出时可能触发 GBK 编码错误，因此备用命令直接调用环境中的 Python；它仍然使用 `ai_env`，不是系统 Python。

## 隐私和安全

- `.env` 已被 Git 忽略，禁止在代码中硬编码 API Key。
- `data/resumes/sample_resume.md` 是虚构数据，可以进入公开仓库。
- 真实简历请放入 `data/private/`，该目录已被 Git 忽略。
- 公开代码前仍应运行 `git status --ignored` 检查敏感文件。

## 阶段 1 的运行模式

### 本地模式（默认、免费、可复现）

```powershell
python app.py analyze --provider local
```

默认读取示例简历和第一份岗位样本，将 JSON 报告保存到 `reports/`。也可以指定文件：

```powershell
python app.py analyze --provider local `
  --resume data/resumes/sample_resume.md `
  --job data/jobs/04_ai_test_intern.md
```

如果要单独检查抽取结果（包含标准化字段、原文证据、行号、字符范围、PDF 页数、警告和解析元数据）：

```powershell
python app.py parse-resume --resume data/resumes/03_cn_data_analyst_single_page.pdf
python app.py parse-job --job data/jobs/01_baidu_llm_algorithm_graduate.md
```

默认分别写入 `reports/parsed_resume.json` 和 `reports/parsed_job.json`；可用 `--output` 指定路径。

### 简历—岗位证据匹配工具

本地模式对技能、项目、工作、教育、专业、经验年限和到岗条件进行可复现匹配，
每个判断都会保留岗位要求原文、简历证据 ID、原文位置、熟练度、约束检查、置信度和复核标记：

```powershell
# 单个岗位
python app.py match-evidence `
  --resume data/resumes/sample_resume.md `
  --job data/jobs/04_baidu_llm_application_engineer.md `
  --mode local

# 批量匹配并生成岗位排名
python app.py match-jobs `
  --resume data/resumes/sample_resume.md `
  --jobs-dir data/jobs `
  --mode local
```

`local` 完全使用确定性规则。`hybrid` 只将需要语义复核的候选证据发送给配置好的
OpenAI 兼容模型；模型只能选择工具已经生成的 `evidence_id`，不能新增简历证据，
也不能覆盖规则确认的硬约束失败。真实简历使用混合模式前应先确认隐私和数据合规要求。

默认单岗位结果为 `reports/evidence_match.json`；批量结果保存在
`reports/evidence_matches/`，排名汇总为 `reports/evidence_match_batch.json`。

### 模型工具注册中心

简历 PDF 解析、岗位解析、单岗位证据匹配和RAG岗位推荐已封装为四个严格白名单工具：
`parse_resume_pdf`、`parse_job_description`、`match_resume_job_evidence`、
`recommend_jobs_with_rag`。
注册中心可为 OpenAI Responses API 或 Chat Completions 导出严格 JSON Schema，
并统一完成未知工具拦截、Pydantic 参数校验、项目目录边界检查、文件类型与大小限制、
异常脱敏和结构化结果校验：

```python
from joblens.agent import build_default_registry

registry = build_default_registry()
model_tools = registry.openai_tools(api="responses")

# 收到模型 function call 后，只能通过注册中心执行
result = registry.execute(
    "parse_resume_pdf",
    {"file_path": "data/resumes/01_cn_standard_single_page.pdf"},
)
tool_output = result.to_model_message()
```

四个工具均为只读且默认不访问网络。证据匹配与推荐工具固定使用本地模式，
不会把简历个人信息发送给外部模型；输出为模型友好的精简摘要，不包含联系方式和整份原文。

### 阶段二 Agent 核心

阶段二运行器让模型从上述四个白名单工具中自主选择能力，但模型不会直接执行 Python、
访问任意路径或修改文件。每次函数调用都经过注册中心的参数、扩展名、大小和目录边界检查，
工具结果再通过 `call_id` 返回给模型，直到生成最终答复或触发安全停止条件。

先安装可选依赖，并在 `.env` 中配置 `LLM_PROVIDER`、`LLM_MODEL`、`LLM_API_KEY`。
`LLM_PROVIDER=deepseek` 时会自动使用官方兼容地址 `https://api.deepseek.com`；其他
OpenAI 兼容服务可以显式配置 `LLM_BASE_URL`，且显式地址的优先级最高：

```powershell
pip install -e ".[llm]"
python app.py agent `
  --message "请比较 data/resumes/01_cn_standard_single_page.pdf 与 data/jobs/04_baidu_llm_application_engineer.md"
```

默认最多执行 6 轮模型决策和 6 次工具调用；同一工具与参数重复超过 2 次会自动停止。
完整结果保存在 `reports/agent_run.json`，包括停止原因、每轮响应、工具名、参数、耗时与
脱敏错误摘要。Agent 使用无状态历史回传，并设置 `store=False`。模型会看到用户问题与
工具的精简结构化输出；真实简历在接入任何外部模型前仍应确认隐私和合规要求。

### 本地岗位 RAG 数据层

首次使用需要安装可选依赖并下载一次中文 BGE 模型：

```powershell
pip install -e ".[rag]"
python app.py build-job-index
```

默认模型为 `BAAI/bge-small-zh-v1.5`，将30份岗位解析为结构化 Chunk 后写入
`data/vector_store/chroma`。索引包含岗位、章节、要求类别、硬约束、原文来源与内容
哈希，目录已加入 `.gitignore`，可以随岗位数据重复构建。模型下载完成后，可以增加
`--local-files-only` 强制离线加载。

简历多查询不包含姓名、手机和邮箱，并过滤“尚未完成”“只有概念性了解”等明确负向
技能证据：

```powershell
python app.py generate-resume-queries `
  --resume data/resumes/01_cn_standard_single_page.pdf `
  --output reports/resume_queries.json
```

当前阶段已完成向量数据层和查询生成层；下一层将负责执行每条查询、聚合重复岗位并把
候选 Chunk 交给简历证据匹配工具。

混合推荐已经接通完整链路：每条简历查询同时执行BGE向量检索与BM25检索，并增加
技能精确匹配分；随后按岗位聚合重复Chunk，只对召回靠前的岗位执行逐要求证据匹配。
最终分数由35%召回分和65%证据匹配分组成，每个失败的硬约束还会乘以0.7惩罚，且
最终排序优先展示没有明确硬约束失败的岗位：

```powershell
python app.py recommend-jobs-rag `
  --resume data/resumes/01_cn_standard_single_page.pdf `
  --top-k 5 `
  --local-files-only
```

同一能力已注册为 `recommend_jobs_with_rag` 模型工具。Agent收到“从岗位库推荐/筛选/
排名岗位”一类请求时，可以直接调用它；工具固定读取本地岗位库和本地Chroma索引，
不把简历发送到外部Embedding服务。

### 本地对话网页

网页使用FastAPI提供本地接口，前端为无需构建工具的原生HTML/CSS/JavaScript。安装依赖
并确认岗位索引已经建立：

```powershell
pip install -e ".[rag,web]"
python app.py build-job-index --local-files-only
python app.py web
```

然后访问 [http://127.0.0.1:8000](http://127.0.0.1:8000)。点击输入框左侧“＋”上传
PDF简历，可以输入“解析我的简历”或“推荐5个岗位”。服务端使用随机文件名将上传内容
暂存在已忽略的 `data/private/web_uploads/`；点击“清除会话”会立即删除对应文件。
默认只监听 `127.0.0.1`，不要在没有身份认证的情况下改为公网地址。

### BOSS直聘岗位导入

本项目采用“官方搜索链接 + 用户手动保存单个岗位页 + 本地结构化解析”的方式，
不会执行后台爬取、自动登录、验证码规避或批量请求：

```powershell
# 1. 生成官方搜索链接，并在普通浏览器中自行打开
python app.py zhipin-search-url --query "Python 实习生"

# 2. 打开所需岗位详情页，将网页另存为 HTML；真实岗位建议放入已忽略的 data/private/
# 3. 解析保存的页面，并可附上原始详情页地址
python app.py parse-zhipin-export `
  --input data/private/zhipin_job.html `
  --url "https://www.zhipin.com/job_detail/实际页面地址.html"
```

导入器优先读取页面中的 `JobPosting` JSON-LD，再回退到常见 HTML 语义类名，
最终统一输出公司、岗位、薪资、地点、发布日期、职责、要求、福利和原文证据。
也可以导入手动复制并整理为本项目章节格式的 Markdown/TXT。网站页面结构可能变化，
因此真实页面样本需要通过回归测试持续维护。

### PDF 简历解析

- 支持中文单页和双页 PDF，并保留 Markdown/TXT 兼容能力。
- 使用 `pdfplumber` 的单词坐标恢复视觉行，执行 Unicode 归一化、重复字符清理和章节标题恢复。
- 本地规则解析联系方式、求职意向、教育、工作/实习、项目、技能、证书、语言、奖项和论文。
- 扫描型 PDF 会自动尝试本地 OCR；安装方式为 `pip install -e ".[ocr]"`，并安装 Tesseract。非标准路径可通过 `JOBLENS_TESSERACT_CMD` 指定。
- 未安装 OCR 环境时会返回可操作的错误，不会把空文本误判为解析成功。
- 默认限制为 25 MB、10 页，避免异常或超大 PDF 消耗过多资源。

### 简历解析器评测

```powershell
python app.py evaluate-resumes
```

命令读取 `data/ground_truth/resume_parser_gold.json`，批量解析 10 份虚构简历，并把逐字段及总体 Precision、Recall、F1 写入 `reports/resume_parser_evaluation.json`。

### 真实大模型模式（显式启用）

先在 `.env` 中填写 `LLM_MODEL` 和 `LLM_API_KEY`，再运行：

```powershell
python app.py analyze --provider openai
```

只有显式使用 `--provider openai` 才会调用外部 API。请先使用虚构简历测试，不要把真实个人信息发送到不了解的数据服务。

## 关键代码

- `src/joblens/models/schemas.py`：Pydantic 数据结构
- `src/joblens/schemas/`：富结构解析模型与来源引用
- `src/joblens/io.py`：PDF/Markdown/TXT 安全读取与扫描件检测
- `src/joblens/parsers/`：可复现的中文简历与 JD 结构化解析器
- `src/joblens/evaluation/resume_evaluator.py`：标准答案字段级自动评测
- `src/joblens/extractors.py`：本地与 API 解析器
- `src/joblens/matcher.py`：确定性证据匹配
- `src/joblens/pipeline.py`：阶段 1 固定流程
- `src/joblens/agent/contracts.py`：模型工具公共契约、安全策略和统一执行结果
- `src/joblens/agent/builtin_tools.py`：四个业务能力的只读模型工具适配器
- `src/joblens/agent/registry.py`：工具白名单注册、schema 导出和安全执行入口
- `src/joblens/agent/model_client.py`：可替换模型客户端协议和 Responses API 实现
- `src/joblens/agent/runner.py`：带循环上限、失败边界和轨迹的 Agent 核心
- `src/joblens/agent/prompts.py`：可版本化的 Agent 系统指令
- `src/joblens/rag/embedding.py`：本地 BGE Embedding 封装
- `src/joblens/rag/vector_store.py`：Chroma 持久化、重建与查询封装
- `src/joblens/rag/indexer.py`：30份岗位的向量索引构建流程
- `src/joblens/rag/resume_query_generator.py`：隐私最小化的简历多查询生成
- `src/joblens/rag/bm25.py`：轻量中文/英文BM25索引
- `src/joblens/rag/retriever.py`：BGE、BM25与技能精确匹配的混合召回和岗位聚合
- `src/joblens/rag/recommender.py`：接入证据匹配器并生成最终岗位排序
- `src/joblens/web/app.py`：上传安全、对话路由和本地推荐API
- `src/joblens/web/static/`：响应式对话页面、岗位卡片和浏览器交互
- `docs/stage1-guide.md`：面向新手的实现说明
- `docs/technical-stack.md`：技术栈、架构取舍和后续演进建议

## 阶段二后续

1. 增加解析结果缓存，避免同一文件在一次或多次任务中重复解析。
2. 建立 Agent 场景评测集，验证工具选择、失败恢复和证据忠实度。
3. 增加多轮会话存储与前端接口，并继续坚持个人信息最小化原则。
