import pickle

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from raggler.base_classes.base_classes import BaseIndex


class NPIndex(BaseIndex):
    """
    Tracking the index and content of documents.
    Storing embeddings as numpy arrays.
    """

    def __init__(
        self,
    ):
        self.index = None
        self.content = None
        self.bm25 = None

    def add(self, vectors: np.ndarray, content: list[str]):
        """
        Store vectors and content in the index.
        """
        assert vectors.shape[0] == len(
            content
        ), "The number of vectors and content do not match."

        self.index = vectors
        self.content = content
        tokenized_content = [c.split() for c in self.content]
        self.bm25 = BM25Okapi(tokenized_content)

    def save(self, path_to_save_index: str):
        """
        Save the index to the given path.
        """

        with open(path_to_save_index + "index.pk", "wb") as f:
            pickle.dump(self.index, f)

        with open(path_to_save_index + "content.pk", "wb") as f:
            pickle.dump(self.content, f)

        with open(path_to_save_index + "bm25.pk", "wb") as f:
            pickle.dump(self.bm25, f)

    def load(self, path_to_index: str):
        """
        Load the index from the given path.
        """
        with open(path_to_index + "index.pk", "rb") as f:
            self.index = pickle.load(f)

        with open(path_to_index + "content.pk", "rb") as f:
            self.content = pickle.load(f)

        with open(path_to_index + "bm25.pk", "rb") as f:
            self.bm25 = pickle.load(f)

    def retrieve(
        self, query_embedding: np.ndarray, query_text: str, k: int
    ) -> tuple[np.ndarray, list[int]]:
        """
        Retrieve the most similar documents to the given query.
        Use cosine similarity for semantic search and BM25 for lexical search.

        Args:
            query_embedding: The embedding of the query.
            query_text: The text of the query.
            k: The number of documents to retrieve.

        Returns:
            tuple[np.ndarray, np.ndarray]: The combined scores and indices of the most similar documents.
        """
        if len(query_embedding.shape) == 1:
            # single query
            query_embedding = query_embedding.reshape(1, -1)

        # Semantic search
        semantic_scores = cosine_similarity(query_embedding, self.index)[0]

        # Lexical search
        tokenized_query = query_text.split()
        lexical_scores = self.bm25.get_scores(tokenized_query)

        # Normalize scores (min-max scaling to 0-1 range)
        # Add a small epsilon to avoid division by zero if all scores are the same
        epsilon = 1e-9
        normalized_semantic_scores = (semantic_scores - np.min(semantic_scores)) / (
            np.max(semantic_scores) - np.min(semantic_scores) + epsilon
        )
        normalized_lexical_scores = (lexical_scores - np.min(lexical_scores)) / (
            np.max(lexical_scores) - np.min(lexical_scores) + epsilon
        )
        
        # Combine scores
        combined_scores = 0.5 * normalized_semantic_scores + 0.5 * normalized_lexical_scores
        
        actual_k = min(k, len(combined_scores))
        indices = combined_scores.argsort()[-actual_k:][::-1]  # Sort in descending order
        return combined_scores[indices], indices.tolist()
