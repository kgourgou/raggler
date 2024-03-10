#!/usr/bin/env python3
import os
import fire
from raggler import RAG, create_index
from raggler.llm_context import RAG_TEMPLATE
from raggler.mlx_llm import MLXLLM
from raggler.indexes import NPIndex
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


def main(
    query: str,
    k: int = 2,
    show_context: bool = False,
    mlx_llm_name: str = "mlx-community/NeuralBeagle14-7B-4bit-mlx",
    embedder: str = "paraphrase-albert-small-v2",
    refresh_index: bool = False,
):
    """
    Retrieve the most similar documents to the given query.

    Args:
        query: The query to retrieve and generate a response for.
        k: The number of documents to retrieve.
        show_context: Whether to print the context.
        mlx_llm_name: The name of the mlx language model to use.
        embedder: The name of the sentence transformer model to use.
        refresh_index: Whether to refresh the index. False by default.

    Returns:
        str: The response from the language model.
    """

    load_dotenv()
    embedder = SentenceTransformer(embedder)

    path_to_files = os.getenv("RAGGLER_DIR")
    if path_to_files is None:
        raise ValueError("RAGGLER_DIR environment variable not set.")

    default_path_for_index = os.path.join("data/indexes/")
    if os.path.exists(default_path_for_index) and not refresh_index:
        index = NPIndex()
        index.load(default_path_for_index)
    else:
        # create an index
        index = create_index(
            paths_to_directories=[path_to_files],
            embedder=embedder,
            index=NPIndex(),
            path_to_save_index=default_path_for_index,
        )

    rag = RAG(embedder, index, MLXLLM(mlx_llm_name, RAG_TEMPLATE))
    response = rag(query, k=k, show_context=show_context)
    return response


if __name__ == "__main__":
    fire.Fire(main)
