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
    vectors = np.array([
        [0.8, 0.2],  # Doc 0: "old knowledge"
        [0.2, 0.8],  # Doc 1: "ancient wisdom traditions"
        [0.1, 0.9]   # Doc 2: "modern technology"
    ])
    content = [
        "old knowledge",
        "ancient wisdom traditions",
        "modern technology",
    ]
    index.add(vectors, content)

    query_text = "ancient wisdom"
    query_embedding = np.array([0.9, 0.1]) 
    k = 2

    combined_scores, indices = index.retrieve(
        query_embedding=query_embedding, query_text=query_text, k=k
    )

    # Assert that results are not empty
    assert len(combined_scores) > 0, "Scores should not be empty"
    assert len(indices) > 0, "Indices should not be empty"

    # Assert shape of the results
    assert combined_scores.shape == (k,), f"Scores shape should be ({k},), but got {combined_scores.shape}"
    assert indices.shape == (k,), f"Indices shape should be ({k},), but got {indices.shape}"

    # Assert that indices are within the valid range
    for idx in indices:
        assert 0 <= idx < len(content), f"Index {idx} is out of bounds"

    # Assert specific content based on expected order
    # Doc1 ("ancient wisdom traditions") should be first, Doc0 ("old knowledge") second.
    assert content[indices[0]] == "ancient wisdom traditions"
    assert content[indices[1]] == "old knowledge"

    # Assert relative scores
    # Score for "ancient wisdom traditions" (indices[0]) should be higher than for "old knowledge" (indices[1])
    assert combined_scores[0] > combined_scores[1]
    
    # Optional: Check if scores are roughly in expected range (might be brittle due to BM25 variance)
    # For example, both scores should be > 0 and < 1 if there's effective normalization
    assert combined_scores[0] > 0 and combined_scores[0] <= 1.0
    assert combined_scores[1] > 0 and combined_scores[1] <= 1.0
