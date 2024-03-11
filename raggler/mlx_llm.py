import os
import pickle
import logging

from pathlib import Path
from mlx_lm import generate, load
from .base_classes.base_classes import BaseLLM

DEFAULT_OPTIONS = {"verbose": False, "max_tokens": 1024}

logger = logging.getLogger(__name__)


class MLXLLM(BaseLLM):
    """
    Wrapper for an MLX language model.
    """

    def __init__(self, model_name: str, prompt_template: str = None):
        """
        Initialize the MLX language model.

        Args:
            model_name (str): The name of the model to use.
            prompt_template (str): The prompt_template to use when querying the language model. Should contain a {context} and a {text} placeholder. If not provided, a default template will be used.

        """
        self.model = None
        self.tokenizer = None

        self.load_model_and_tokenizer(model_name)
        self.prompt_template = prompt_template if prompt_template else "{text}"

    def load_model_and_tokenizer(self, model_name):
        self.model, self.tokenizer = load(model_name)

    def __call__(self, **kwargs) -> str:
        """
        Query the language model with the given text.

        Args:
            **kwargs: arguments with which to populate the prompt template.

        Returns:
            str: The response from the language model.
        """
        prompt = self.prompt_template.format(**kwargs)
        return generate(self.model, self.tokenizer, prompt, **DEFAULT_OPTIONS)
