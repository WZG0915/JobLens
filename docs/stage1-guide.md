# 阶段 1 学习说明：固定流程的岗位分析器

## 1. 为什么阶段 1 还不是 Agent

当前调用顺序由程序写死：

```text
读取简历 → 解析简历 → 读取 JD → 解析 JD → 匹配证据 → 保存报告
```

程序不会根据用户问题决定使用哪些函数，因此它是一个固定工作流。阶段 2 才会把函数变成可由模型选择的工具。

## 2. 四个核心函数

对外流程函数位于 `src/joblens/pipeline.py`：

- `parse_resume()`：读取简历并返回 `ResumeProfile`。
- `parse_job_description()`：读取 JD 并返回 `JobDescription`。
- `match_requirements()`：将每条岗位要求与简历证据匹配。
- `generate_report()`：运行完整流程并写出 JSON 报告。

底层的 `src/joblens/parsers/` 还提供富结构解析：它不仅返回技能和要求，
还保留原文证据、行号、字符范围、解析警告与耗时元数据。`extractors.py`
负责把富结构结果适配为阶段 1 的稳定报告模型。

## 3. Pydantic 在这里解决什么问题

`src/joblens/models/schemas.py` 定义了程序允许的数据形状。例如匹配状态只能是：

```text
matched
partially_matched
insufficient_evidence
not_matched
```

如果本地解析器或大模型返回字符串代替技能列表、遗漏必要字段或添加未知字段，验证层会抛出可读错误，不让错误数据继续流入匹配逻辑。

## 4. local 与 openai 两种模式

### local

使用简单、确定性的 Markdown 规则提取信息。优点是免费、可重复、适合调试和运行测试；不足是只适合当前约定的文档格式。

### openai

使用 Responses API 的 Structured Outputs，将同一套 Pydantic 模型作为输出结构。必须显式传入 `--provider openai` 才会发起请求，避免意外产生费用。

简历和 JD 被放在用户输入中；系统提示强调它们是待分析数据，并要求忽略文档中的指令。请求设置 `store=False`。

## 5. 为什么提取可以用大模型，匹配暂时不用

自然语言岗位描述格式变化很大，适合由模型抽取成统一结构；而“证据是否覆盖关键词、熟练程度是否满足要求”可以先用确定性代码完成，这样每个结论更容易复现和测试。

## 6. 报告字段如何阅读

- `requirement_evidence`：来自岗位 JD 的原始要求。
- `resume_evidence`：来自简历的支持证据，可能为空。
- `matched_keywords`：两边匹配到的标准化技能名。
- `status`：匹配状态。
- `explanation`：为什么得到该状态。
- `evidence_coverage`：找到简历证据的岗位要求占比，不等同于录用概率。

## 7. 当前局限

- local 模式支持常见标题别名，但仍主要依赖 Markdown 二级标题和项目三级标题。
- 技能词表有限，不能识别所有同义词。
- 当前结果是简历证据覆盖分析，不是求职成功概率。
- PDF、Word、图片简历将在后续阶段考虑。
- 阶段 1 不包含对话、记忆、工具选择、RAG 或 LangGraph。
