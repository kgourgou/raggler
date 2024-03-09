"""
Base class for language models.
"""

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """
    Base class for language models.
    """

    @abstractmethod
    def __call__(self, text: str, **kwargs) -> str:
        """
        Query the language model with the given text.

        Args:
            text (str): The text to query the language model with.
            **kwargs: Additional arguments to pass to the language model.

        Returns:
            str: The response from the language model.
        """
        pass


class BaseIndex(ABC):
    @abstractmethod
    def add(self, vectors, content):
        """
        Add the given vectors to the index, alongside the given content.
        """
        pass

    @abstractmethod
    def save(self, path_to_save_index):
        """
        Save the index to the given path.
        """
        pass

    @abstractmethod
    def load(self, path_to_index):
        """
        Load the index from the given path.
        """
        pass

    @abstractmethod
    def retrieve(self, query, k):
        """
        Retrieve the most similar documents to the given query.
        """
        pass
