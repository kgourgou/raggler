"""
TODO: This won't work as is, need to fix the retrieve method, and test the load / save methods.

When ready to fix, add faiss to the requirements.txt file.
"""

import faiss
import pickle
from raggler.base_classes.base_classes import BaseIndex


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

    def retrieve(self, query_embedding, k: int = 2):
        """
        Retrieve the most similar documents to the given query.

        Args:
            query_embedding: The query to retrieve the most similar documents to.
            k: The number of documents to retrieve.

        Should return the distances and indices of the k most similar documents.
        """
        raise NotImplementedError("Sorry, this method is not implemented yet.")
