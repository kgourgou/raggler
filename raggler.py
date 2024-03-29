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
    ctx: bool = False,
    mlx_llm_name: str = "mlx-community/AlphaMonarch-7B-mlx-4bit",
    embedder: str = "all-MiniLM-L12-v2",
    rfr: bool = False,
    files: str = None,
    thr: float = 0.0,
):
    """
    Retrieve the most similar documents to the given query
    and generate a response from the language model.

    Args:
        query: The query to retrieve and generate a response for.
        k: The number of documents to retrieve.
        ctx: Whether to print the context found.
        mlx_llm_name: The name of the mlx language model to use.
        embedder: The name of the sentence transformer model to use.
        rfr: Whether to refresh the index. False by default.
        files: The path to the directory containing the documents to index.
            This takes precedence over the RAGGLER_DIR environment variable.
        thr: The similarity threshold for the retrieved chunks.
            Set it to > 0.0 to filter out chunks with similarity < thr.

    Returns:
        str: The response from the language model.
    """

    load_dotenv()
    embedder = SentenceTransformer(embedder)

    files = files or os.getenv("RAGGLER_DIR")

    default_path_for_index = os.path.join("data/indexes/")
    if os.path.exists(default_path_for_index) and not rfr:
        index = NPIndex()
        index.load(default_path_for_index)
    else:
        # create an index
        index = create_index(
            paths_to_directories=[files],
            embedder=embedder,
            index=NPIndex(),
            path_to_save_index=default_path_for_index,
        )

    rag = RAG(embedder, index, MLXLLM(mlx_llm_name, RAG_TEMPLATE))
    return rag(query, k=k, show_context=ctx, thr=thr)


if __name__ == "__main__":
    fire.Fire(main)
