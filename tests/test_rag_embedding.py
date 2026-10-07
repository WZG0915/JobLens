"""BGE 封装测试，不下载真实模型。"""

from __future__ import annotations

import numpy as np

from joblens.rag import BgeEmbeddingSettings, BgeLocalEmbedding


class FakeSentenceTransformer:
    """只记录输入并返回固定维度向量的测试替身。"""

    def __init__(self) -> None:
        self.inputs: list[list[str]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts, **kwargs):
        self.inputs.append(list(texts))
        return np.asarray([[float(index + 1), 0.5, 0.25] for index, _ in enumerate(texts)])


def test_bge_wrapper_separates_document_and_query_encoding() -> None:
    model = FakeSentenceTransformer()
    embedding = BgeLocalEmbedding(
        BgeEmbeddingSettings(model_name="fake-bge", query_instruction="检索："),
        model=model,
    )

    documents = embedding.embed_documents(["Python Agent", "RAG开发"])
    query = embedding.embed_query("大模型应用")

    assert embedding.model_name == "fake-bge"
    assert embedding.dimension == 3
    assert len(documents) == 2
    assert len(query) == 3
    assert model.inputs[0] == ["Python Agent", "RAG开发"]
    assert model.inputs[1] == ["检索：大模型应用"]
