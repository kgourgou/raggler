import pytest
import numpy as np
from raggler.faiss_index import FAISSIndex

# Test for adding vectors and content, then retrieving
def test_add_and_retrieve_simple():
    index = FAISSIndex(d=3)
    vectors = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    content = ["apple", "banana"]
    index.add(vectors, content)

    assert index.content == content, "Content not stored correctly"
    assert index.bm25 is not None, "BM25 not initialized"
    assert index.index.ntotal == 2, "Incorrect number of vectors in FAISS index"

    query_embedding = np.array([1, 2, 3], dtype=np.float32)
    query_text = "apple"
    scores, indices = index.retrieve(query_embedding, query_text, k=1)

    assert len(scores) == 1, "Incorrect number of scores returned"
    assert len(indices) == 1, "Incorrect number of indices returned"
    assert index.content[indices[0]] == "apple", "Retrieved content mismatch"

# Test for saving and loading the index
def test_save_load(tmp_path):
    index = FAISSIndex(d=2)
    vectors = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    content = ["doc1", "doc2"]
    index.add(vectors, content)

    save_path = str(tmp_path) + "/" # FAISS save/load might need directory-like path
    index.save(save_path)

    loaded_index = FAISSIndex(d=2)
    loaded_index.load(save_path)

    assert loaded_index.content == content, "Loaded content mismatch"
    assert loaded_index.index.ntotal == 2, "Loaded FAISS index ntotal mismatch"
    assert loaded_index.bm25 is not None, "Loaded BM25 is None"
    # Check if BM25 has indexed the correct number of documents by checking corpus size or similar
    assert len(loaded_index.bm25.get_scores("doc1".split())) == len(content), "BM25 doc count mismatch after load"


    query_embedding = np.array([0.1, 0.2], dtype=np.float32)
    query_text = "doc1"
    scores, indices = loaded_index.retrieve(query_embedding, query_text, k=1)

    assert len(scores) == 1, "Retrieval from loaded index failed (scores)"
    assert len(indices) == 1, "Retrieval from loaded index failed (indices)"
    assert loaded_index.content[indices[0]] == "doc1", "Retrieved content mismatch from loaded index"

# Test for hybrid retrieval ranking logic
def test_hybrid_retrieval_ranking():
    index = FAISSIndex(d=2)
    vectors = np.array([
        [0.9, 0.1],  # Doc A: Semantically like "query_sem"
        [0.1, 0.9],  # Doc B: Semantically different, but lexically like "query_lex"
        [0.8, 0.2],  # Doc C: Semantically similar, also some lexical overlap
    ], dtype=np.float32)
    content = [
        "semantic match here",      # Doc A (index 0)
        "lexical match there",      # Doc B (index 1)
        "semantic and lexical too", # Doc C (index 2)
    ]
    index.add(vectors, content)

    query_embedding = np.array([0.95, 0.05], dtype=np.float32)  # Targets Doc A and C semantically
    query_text = "lexical too"  # Targets Doc B and C lexically

    scores, indices = index.retrieve(query_embedding, query_text, k=3)
    
    retrieved_content_ordered = [content[i] for i in indices]

    assert len(scores) == 3, "Should return 3 scores"
    assert len(indices) == 3, "Should return 3 indices"
    assert len(set(retrieved_content_ordered)) == 3, "All unique documents should be returned"

    # Expected: Doc C is top due to combined semantic and lexical match.
    assert retrieved_content_ordered[0] == "semantic and lexical too", \
        f"Doc C ('semantic and lexical too') expected first. Got: {retrieved_content_ordered}"

    # Doc A ("semantic match here") and Doc B ("lexical match there") are next.
    # We need to ensure Doc C's score is higher than both A and B.
    score_C = scores[0] # Since Doc C is asserted to be first.

    try:
        idx_A_in_results = retrieved_content_ordered.index("semantic match here")
        score_A = scores[idx_A_in_results]
        assert score_C >= score_A, f"Score of C ({score_C}) should be >= score of A ({score_A})"
    except ValueError:
        assert False, "Doc A ('semantic match here') not found in results"

    try:
        idx_B_in_results = retrieved_content_ordered.index("lexical match there")
        score_B = scores[idx_B_in_results]
        assert score_C >= score_B, f"Score of C ({score_C}) should be >= score of B ({score_B})"
    except ValueError:
        assert False, "Doc B ('lexical match there') not found in results"

    # Ensure the order of A and B (if they are not C) is such that higher score comes first
    if "semantic match here" in retrieved_content_ordered[1:] and \
       "lexical match there" in retrieved_content_ordered[1:]:
        idx_A_in_results = retrieved_content_ordered.index("semantic match here")
        idx_B_in_results = retrieved_content_ordered.index("lexical match there")
        
        # If A is ranked higher than B in results, its score should be >= B's score
        if idx_A_in_results < idx_B_in_results:
            assert scores[idx_A_in_results] >= scores[idx_B_in_results], \
                f"Doc A ranked before Doc B, but score A ({scores[idx_A_in_results]}) < score B ({scores[idx_B_in_results]})"
        # If B is ranked higher than A in results, its score should be >= A's score
        else: # idx_B_in_results < idx_A_in_results
            assert scores[idx_B_in_results] >= scores[idx_A_in_results], \
                f"Doc B ranked before Doc A, but score B ({scores[idx_B_in_results]}) < score A ({scores[idx_A_in_results]})"

# Test retrieving from an empty index
def test_retrieve_empty_index():
    index = FAISSIndex(d=2)
    query_embedding = np.array([0.1, 0.2], dtype=np.float32)
    query_text = "query"
    scores, indices = index.retrieve(query_embedding, query_text, k=1)

    assert len(scores) == 0, "Scores should be empty for empty index"
    assert len(indices) == 0, "Indices should be empty for empty index"

# Test retrieving when only a single document is in the index
def test_retrieve_single_document_in_index():
    index = FAISSIndex(d=2)
    vectors = np.array([[0.1, 0.2]], dtype=np.float32)
    content = ["single doc"]
    index.add(vectors, content)

    query_embedding = np.array([0.1, 0.2], dtype=np.float32)
    query_text = "single" 
    scores, indices = index.retrieve(query_embedding, query_text, k=1)

    assert len(scores) == 1, "Should return 1 score for single doc"
    assert np.isclose(scores[0], 1.0), f"Score for single doc should be approx 1.0, got {scores[0]}"
    assert len(indices) == 1, "Should return 1 index for single doc"
    assert indices[0] == 0, "Index for single doc should be 0"
    assert index.content[indices[0]] == "single doc"

# Test retrieving when k is greater than the number of documents
def test_retrieve_k_greater_than_docs():
    index = FAISSIndex(d=2)
    vectors = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    content = ["doc1", "doc2"]
    index.add(vectors, content)

    query_embedding = np.array([0.1, 0.2], dtype=np.float32) 
    query_text = "doc1" 
    scores, indices = index.retrieve(query_embedding, query_text, k=5)

    assert len(scores) == 2, "Should return 2 scores when k > num_docs (actual number of docs)"
    assert len(indices) == 2, "Should return 2 indices when k > num_docs (actual number of docs)"

    # Check content and order (doc1 should be first due to stronger match with query_embedding and query_text)
    # This assumes 'doc1' will have a higher combined score than 'doc2' for the given query.
    # If 'doc2' could rank higher, this assertion would need to be more flexible.
    if len(scores) == 2: # Ensure we have two scores to compare
        assert index.content[indices[0]] == "doc1", "doc1 expected first"
        assert index.content[indices[1]] == "doc2", "doc2 expected second"
        assert scores[0] >= scores[1], "Scores should be in descending order"
    elif len(scores) == 1: # Only one doc returned
        assert index.content[indices[0]] == "doc1", "If only one doc, it should be doc1"
