import faiss
import pickle

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .base_classes.base_classes import BaseIndex


class NPIndex(BaseIndex):
    """

    Tracking the index and content of documents.
    Storing embeddings as mlx arrays.

    """

    def __init__(
        self,
    ):
        self.index = None
        self.content = None

    def add(self, vectors: np.ndarray, content: list[str]):
        """
        Store vectors and content in the index.
        """
        assert vectors.shape[0] == len(
            content
        ), "The number of vectors and content do not match."

        self.index = vectors
        self.content = content

    def save(self, path_to_save_index):
        """
        Save the index to the given path.
        """

        with open(path_to_save_index + "index.pk", "wb") as f:
            pickle.dump(self.index, f)

        with open(path_to_save_index + "content.pk", "wb") as f:
            pickle.dump(self.content, f)

    def load(self, path_to_index):
        """
        Load the index from the given path.
        """
        with open(path_to_index + "index.pk", "rb") as f:
            self.index = pickle.load(f)

        with open(path_to_index + "content.pk", "rb") as f:
            self.content = pickle.load(f)

    def retrieve(self, query_embedding, k: int):
        """
        Retrieve the most similar documents to the given query.
        Use cosine similarity to compare the query to
            the documents in the index.
        """
        if len(query_embedding.shape) == 1:
            # single query
            query_embedding = query_embedding.reshape(1, -1)

        distances = cosine_similarity(query_embedding, self.index)[0]

        actual_k = min(k, len(distances))
        indices = distances.argsort()[-actual_k:]
        return distances[indices], indices


class FAISSIndex(BaseIndex):
    def __init__(self, d: int):
        self.index = faiss.IndexIDMap2(faiss.IndexFlatL2(d))
        self.content = []

    def add(self, vectors, content):
        """
        Add the given vectors to the index, alongside the given content.
        """
        self.index.add_with_ids(vectors, range(len(content)))
        self.content.append(content)

    def save(self, path_to_save_index):
        """
        Save the index to the given path.
        """
        faiss.write_index(self.index, path_to_save_index + "index.faiss")

        with open(path_to_save_index + "content.pk", "wb") as f:
            pickle.dump(self.content, f)

    def load(self, path_to_index):
        """
        Load the index from the given path.
        """
        self.index = faiss.read_index(path_to_index + "index.faiss")

        with open(path_to_index + "content.pk", "rb") as f:
            self.content = pickle.load(f)

    def retrieve(self, query_embedding, k):
        """
        Retrieve the most similar documents to the given query.
        """
        raise NotImplementedError
