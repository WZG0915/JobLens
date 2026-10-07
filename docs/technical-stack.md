# JobLens 技术栈与解析架构

## 当前技术栈

| 技术 | 在项目中的职责 | 当前选择的理由 |
|---|---|---|
| Python 3.10 | CLI、解析、匹配、报告生成 | 生态成熟，适合逐步扩展 Agent、RAG 和数据处理 |
| Pydantic v2 | 简历、JD、匹配报告的强类型契约 | 能同时校验规则解析和大模型 Structured Outputs，拒绝多余字段 |
| PDF / Markdown / TXT | 简历与岗位输入格式 | PDF 覆盖真实投递场景，文本格式保留为可复现基线 |
| pdfplumber | PDF 文本层、页面与阅读顺序提取 | 对中文字体和复杂导出 PDF 的兼容性优于简单字符串读取 |
| OpenAI Python SDK（可选） | 非固定格式文档的结构化抽取 | 只在显式使用 `--provider openai` 时调用，默认本地模式零成本可复现 |
| python-dotenv | 本地加载模型配置 | API Key 不进入源码和 Git |
| unittest / pytest | 回归测试 | 当前测试不依赖网络，运行稳定；pytest 可直接执行 unittest 测试 |
| JSON | 解析产物和匹配报告 | 便于调试、评测、SQLite 入库以及后续 Agent 工具调用 |

## 当前数据流

```text
PDF / Markdown / TXT
    │
    ├─ io.py：UTF-8/PDF 文本层、安全限制、扫描件识别和可读错误
    │
    ├─ parsers/resume_parser.py、parsers/job_parser.py
    │      └─ PDF 章节恢复 + 富结构数据 + 原文证据 + warnings + metadata
    │
    ├─ extractors.py：适配为阶段 1 的稳定轻量模型
    │
    ├─ matcher.py：确定性证据匹配，不生成“录用概率”
    │
    └─ pipeline.py：输出 JSON 报告
```

## 两层 Pydantic 模型为什么同时保留

- `joblens.schemas` 是解析层模型，字段完整，适合保存和后续精细匹配。例如技能类别、要求逻辑、原文位置、解析警告和运行元数据。
- `joblens.models` 是阶段 1 报告层的稳定模型，字段少，保证现有 CLI、测试和报告格式不被一次结构升级破坏。
- `extractors.py` 是两层之间的适配边界。后续匹配器完全迁移到富结构模型后，可以删除这层兼容，而不是让业务代码同时理解两套结构。

## 规则解析与大模型解析的分工

本地规则解析适合带文本层的中文 PDF 和格式受控的 Markdown/TXT，优势是便宜、可解释、可测试。PDF 先恢复姓名、章节和条目层级，再进入与文本简历相同的结构化解析契约。大模型解析适合标题缺失、表达多样、职责和要求混排的真实文档，但必须通过 Pydantic 校验，并且不能凭空推断技能水平或经历。

证据匹配当前保持确定性：岗位结论必须能回指简历原文。大模型后续可以用于解释和改写建议，但不应替代证据边界。

## 暂不建议现在引入的组件

- LangChain / LangGraph：当前流程固定，普通函数更清楚；阶段 2 出现真正的工具选择和循环后再引入。
- 向量数据库：短简历和单份 JD 不需要 RAG；等学习资料达到几十份以上再评估 FAISS、Chroma 或 pgvector。
- OCR：当前扫描件会返回明确错误；建立扫描件评测集后再加入按需 OCR，避免所有 PDF 都承担识别误差。

## 推荐的后续顺序

1. 用 10–20 份不同格式的匿名 JD 建立解析评测集，记录字段级 precision/recall。
2. 为扫描型 PDF 增加可选 OCR，并记录 OCR 置信度与页级告警。
3. 将富结构 `JobRequirement` 直接接入匹配器，支持学历、年限、熟练度与 `ALL/ANY/AT_LEAST`。
4. 使用 SQLite 保存原文、解析版本、结构化结果和投递状态。
5. 再把读取、解析、匹配、保存包装为 Agent 工具，并加入最大步数和调用轨迹。
