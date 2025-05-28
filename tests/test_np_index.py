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


def test_hybrid_retrieval():
    index = NPIndex()
    # Sample data with some overlapping terms for BM25
    vectors = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    content = [
        "apple banana orange",
        "apple pie recipe",
        "orange juice benefits",
    ]
    index.add(vectors, content)

    # Sample query
    query_text = "apple recipe"
    # Dummy embedding for the query (actual embedding values don't matter for this test)
    query_embedding = np.array([0.15, 0.25, 0.35]) 
    k = 1

    distances, indices = index.retrieve(
        query_embedding=query_embedding, query_text=query_text, k=k
    )

    # Assert that results are not empty
    assert len(distances) > 0, "Distances should not be empty"
    assert len(indices) > 0, "Indices should not be empty"

    # Assert shape of the results
    assert distances.shape == (k,), f"Distances shape should be ({k},), but got {distances.shape}"
    assert indices.shape == (k,), f"Indices shape should be ({k},), but got {indices.shape}"

    # Assert that indices are within the valid range
    for idx in indices:
        assert 0 <= idx < len(content), f"Index {idx} is out of bounds"
