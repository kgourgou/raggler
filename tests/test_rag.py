import numpy as np
from raggler import RAG, create_index
from raggler.rag import EXT_TO_LOADER
from raggler.base_classes.base_classes import BaseIndex, BaseLLM
from raggler.llm_context import RAG_TEMPLATE


class MockEmbedder:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, content):
        return np.array([1, 1, 1, 1])


class MockIndex(BaseIndex):
    def __init__(self, *args, **kwargs):
        self.content = None
        self.index = None

    def add(self, vectors, content):
        self.content = content
        self.index = vectors

    def retrieve(self, query_embedding, k: int = 1):
        return np.array([0]), np.array([0])

    def save(self, path_to_save_index: str):
        pass

    def load(self, path_to_index: str):
        pass


class MockMLM(BaseLLM):
    def __call__(self, **kwargs) -> str:
        return "response"


def test_rag():
    index = MockIndex()
    index.content = ["content"]
    index.index = np.array([1, 1, 1, 1])

    rag = RAG(MockEmbedder(), index, MockMLM())
    assert rag("query") == "response"
    assert rag.retrieve("query") == (np.array([0]), np.array([0]))


def test_create_index():
    index = create_index(
        paths_to_directories=["tests/test_index/"],
        embedder=MockEmbedder(),
        index=MockIndex(),
    )
    assert index.content == ["content"]


def test_wrong_path():
    try:
        create_index(
            paths_to_directories=["no_such_directory__/"],
            embedder=MockEmbedder(),
            index=MockIndex(),
        )
    except FileNotFoundError:
        pass


def test_loader():
    assert len(EXT_TO_LOADER) > 0


def test_rag_template():
    assert "{context}" in RAG_TEMPLATE
    assert "{question}" in RAG_TEMPLATE
