"""无需外部服务的轻量 BM25 中文/英文关键词检索。"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence


TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9+#.\-]*|[\u4e00-\u9fff]+")


def tokenize_for_bm25(text: str) -> list[str]:
    """提取英文技术词，并把连续中文转换成二元/三元短语。"""
    tokens: list[str] = []
    for value in TOKEN_PATTERN.findall(text.casefold()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", value):
            # 完整短语对短标题有用，二元和三元片段则减少没有分词器时的漏召回。
            tokens.append(value)
            if len(value) == 1:
                tokens.append(value)
            else:
                tokens.extend(value[index : index + 2] for index in range(len(value) - 1))
                tokens.extend(value[index : index + 3] for index in range(len(value) - 2))
        else:
            tokens.append(value.strip(".-"))
    return [token for token in tokens if token]


@dataclass(frozen=True)
class Bm25Hit:
    """BM25 命中文档在原列表中的位置和归一化得分。"""

    index: int
    raw_score: float
    score: float


class Bm25Index:
    """适合当前数百个岗位 Chunk 的内存 BM25 索引。"""

    def __init__(self, documents: Sequence[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        if not documents:
            raise ValueError("BM25 文档不能为空")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 参数不合法")
        self.k1 = k1
        self.b = b
        self._tokens = [tokenize_for_bm25(document) for document in documents]
        self._term_frequencies = [Counter(tokens) for tokens in self._tokens]
        self._lengths = [len(tokens) for tokens in self._tokens]
        self._average_length = sum(self._lengths) / len(self._lengths) or 1.0
        document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            document_frequency.update(set(tokens))
        count = len(documents)
        self._idf = {
            term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def scores(self, query: str) -> list[float]:
        """计算查询对全部文档的原始 BM25 得分。"""
        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return [0.0] * len(self._tokens)
        scores: list[float] = []
        for frequencies, length in zip(self._term_frequencies, self._lengths):
            score = 0.0
            for term in query_tokens:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / self._average_length
                )
                score += self._idf.get(term, 0.0) * frequency * (self.k1 + 1) / denominator
            scores.append(score)
        return scores

    def search(self, query: str, *, n_results: int = 20) -> list[Bm25Hit]:
        """返回按相关度排序的文档；分数除以本次查询最高分归一化。"""
        if n_results < 1:
            raise ValueError("n_results 必须大于 0")
        raw_scores = self.scores(query)
        ranked = sorted(
            ((index, score) for index, score in enumerate(raw_scores) if score > 0),
            key=lambda item: (-item[1], item[0]),
        )[:n_results]
        maximum = ranked[0][1] if ranked else 0.0
        return [
            Bm25Hit(index=index, raw_score=score, score=score / maximum)
            for index, score in ranked
        ]
