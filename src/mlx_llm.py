from src.base_classes.base_llm import BaseLLM
from mlx_lm import load, generate


class MLXLLM(BaseLLM):
    """
    Wrapper for the MLX language model.
    """

    def __init__(self, model_name: str):
        """
        Initialize the MLX language model.

        Args:
            model_name (str): The name of the model to use.
        """
        self.model, self.tokenizer = load(model_name)

    def query(self, text: str, **kwargs) -> str:
        """
        Query the language model with the given text.

        Args:
            text (str): The text to query the language model with.
            **kwargs: Additional arguments to pass to the language model.

        Returns:
            str: The response from the language model.
        """
        return generate(self.model, self.tokenizer, text, **kwargs)
