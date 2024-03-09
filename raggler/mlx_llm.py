import os
import pickle
import logging

from pathlib import Path
from mlx_lm import generate, load
from .base_classes.base_classes import BaseLLM

DEFAULT_OPTIONS = {"verbose": True, "max_tokens": 512}

CACHE_PATH = "data/models"

logger = logging.getLogger(__name__)


class MLXLLM(BaseLLM):
    """
    Wrapper for an MLX language model.
    """

    def __init__(
        self, model_name: str, prompt_template: str = None, cache_model: bool = True
    ):
        """
        Initialize the MLX language model.

        Args:
            model_name (str): The name of the model to use.
            prompt_template (str): The prompt_template to use when querying the language model. Should contain a {context} and a {text} placeholder. If not provided, a default template will be used.
            cache_model (bool): Whether to cache the model. Defaults to True. Will cache in the current directory under data/models
        """
        self.model = None
        self.tokenizer = None

        model_path = Path(CACHE_PATH) / model_name
        self.load_model_and_tokenizer(model_name, cache_model, model_path)

        self.prompt_template = prompt_template if prompt_template else "{text}"

    def load_model_and_tokenizer(self, model_name, cache_model, model_path):
        if os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            with open(model_path / "model.pkl", "rb") as f:
                self.model = pickle.load(f)
            with open(model_path / "tokenizer.pkl", "rb") as f:
                self.tokenizer = pickle.load(f)
        else:
            logger.info(f"Loading model from HuggingFace. This may take a while.")
            self.model, self.tokenizer = load(model_name)
            if cache_model:
                os.makedirs(model_path, exist_ok=True)
                with open(model_path / "model.pkl", "wb") as f:
                    pickle.dump(self.model, f)
                with open(model_path / "tokenizer.pkl", "wb") as f:
                    pickle.dump(self.tokenizer, f)

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
