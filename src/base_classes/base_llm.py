"""
Base class for language models.
"""

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """
    Base class for language models.
    """

    @abstractmethod
    def query(self, text: str) -> str:
        """
        Query the language model with the given text.

        Args:
            text (str): The text to query the language model with.

        Returns:
            str: The response from the language model.
        """
        pass
