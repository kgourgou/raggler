"""
Tests for the NPIndex class.
"""

import numpy as np
from raggler.indexes import NPIndex


def test_add():
    index = NPIndex()
    vectors = np.array([[1, 2, 3], [4, 5, 6]])
    content = ["a", "b"]
    index.add(vectors, content)
    assert np.array_equal(index.index, vectors)
    assert index.content == content


def test_retrieve():
    index = NPIndex()
    vectors = np.array([[1, 2, 3], [4, 5, 6]])
    content = ["a", "b"]
    index.add(vectors, content)
    query_embedding = np.array([1, 2, 3])
    _, ind = index.retrieve(query_embedding, 1)
    assert content[ind[0]] == "a"


def test_save(tmp_path):
    index = NPIndex()
    vectors = np.array([[1, 2, 3], [4, 5, 6]])
    content = ["a", "b"]
    index.add(vectors, content)

    tmp_path = str(tmp_path)
    index.save(tmp_path)
    index.load(tmp_path)
    assert np.array_equal(index.index, vectors)
    assert index.content == content
