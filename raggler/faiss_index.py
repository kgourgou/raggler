"""
TODO: This won't work as is, need to fix the retrieve method, and test the load / save methods.

When ready to fix, add faiss to the requirements.txt file.
"""

import faiss
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from raggler.base_classes.base_classes import BaseIndex


class FAISSIndex(BaseIndex):
    def __init__(self, d: int):
        self.index = faiss.IndexIDMap2(faiss.IndexFlatL2(d))
        self.content = []
        self.bm25 = None

    def add(self, vectors: np.ndarray, content: list[str]):
        """
        Add the given vectors to the index, alongside the given content.
        """
        assert vectors.shape[0] == len(content)
        self.content = content
        ids = np.arange(len(content))
        self.index.add_with_ids(vectors, ids)
        tokenized_content = [doc.split() for doc in self.content]
        self.bm25 = BM25Okapi(tokenized_content)

    def save(self, path_to_save_index):
        """
        Save the index to the given path.
        """
        faiss.write_index(self.index, path_to_save_index + "index.faiss")

        with open(path_to_save_index + "content.pk", "wb") as f:
            pickle.dump(self.content, f)
        
        if self.bm25:
            with open(path_to_save_index + "bm25.pk", "wb") as f:
                pickle.dump(self.bm25, f)

    def load(self, path_to_index):
        """
        Load the index from the given path.
        """
        self.index = faiss.read_index(path_to_index + "index.faiss")

        with open(path_to_index + "content.pk", "rb") as f:
            self.content = pickle.load(f)
        
        try:
            with open(path_to_index + "bm25.pk", "rb") as f:
                self.bm25 = pickle.load(f)
        except FileNotFoundError:
            self.bm25 = None # Consistent with __init__

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Min-max normalize scores to the 0-1 range.
        Handles cases where all scores are the same.
        """
        min_val = np.min(scores)
        max_val = np.max(scores)
        delta = max_val - min_val
        if delta == 0:
            # All scores are the same. Return array of 0.5s as a neutral value,
            # indicating no discriminative power from this score type but contributing neutrally.
            return np.full_like(scores, 0.5, dtype=np.float32)
        return (scores - min_val) / delta

    def retrieve(self, query_embedding: np.ndarray, query_text: str, k: int = 2) -> tuple[np.ndarray, list[int]]:
        """
        Retrieve the most similar documents to the given query using a combination of
        semantic (FAISS) and lexical (BM25) search.

        Args:
            query_embedding: The embedding of the query.
            query_text: The text of the query.
            k: The number of documents to retrieve.

        Returns:
            A tuple containing an array of combined scores and a list of document indices.
        """
        if not self.content or self.bm25 is None:
            return np.array([]), []

        # Prepare Query Embedding
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype(np.float32)

        # Handle single document case separately for simplicity in normalization
        if len(self.content) == 1:
            # FAISS search to check if the query matches the dimension
            # This also implicitly checks if self.index is not empty or malformed for search
            try:
                # We don't strictly need the result, just that search doesn't error
                _ = self.index.search(query_embedding, k=1)
            except Exception: # Catch potential FAISS errors with malformed query/index
                return np.array([]), [] # Or handle error more gracefully
            return np.array([1.0], dtype=np.float32), [0]


        # 1. Semantic Search (FAISS)
        # Search for all documents to get a complete score list
        distances_flat, indices_flat = self.index.search(query_embedding, k=len(self.content))
        
        distances_flat = distances_flat[0]
        indices_flat = indices_flat[0]

        valid_mask = indices_flat != -1
        valid_indices = indices_flat[valid_mask]
        valid_distances = distances_flat[valid_mask]

        semantic_distance_map = {idx: dist for idx, dist in zip(valid_indices, valid_distances)}
        
        all_semantic_scores_raw = np.array([semantic_distance_map.get(i, float('inf')) for i in range(len(self.content))], dtype=np.float32)
        
        # Convert distances to similarity scores (higher is better)
        # L2 distances are non-negative, so 1 + distance is > 0 (unless distance is -1, which we filtered)
        all_semantic_scores = 1.0 / (1.0 + all_semantic_scores_raw)

        # 2. Lexical Search (BM25)
        tokenized_query = query_text.split()
        all_lexical_scores = self.bm25.get_scores(tokenized_query)
        all_lexical_scores = np.array(all_lexical_scores, dtype=np.float32)

        # 3. Normalize Scores
        norm_semantic_scores = self._normalize_scores(all_semantic_scores)
        norm_lexical_scores = self._normalize_scores(all_lexical_scores)

        # 4. Combine Scores
        combined_scores = 0.5 * norm_semantic_scores + 0.5 * norm_lexical_scores

        # 5. Get Top K Results
        actual_k = min(k, len(self.content))
        
        # Argsort sorts in ascending order, so we take the last k and reverse them for descending order
        top_k_indices = np.argsort(combined_scores)[-actual_k:][::-1]
        
        final_scores = combined_scores[top_k_indices]
        final_indices = top_k_indices.tolist()

        return final_scores, final_indices
