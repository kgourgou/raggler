"""
Pipeline for retrieval augmented generation.
"""

import logging
import os

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from raggler.indexes import NPIndex
from raggler.mlx_llm import MLXLLM

logger = logging.getLogger(__name__)


EXT_TO_LOADER = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}


def create_index(
    paths_to_directories: list[str],
    embedder: SentenceTransformer,
    path_to_save_index: str = None,
    index=NPIndex(),
) -> tuple[NPIndex, list[str]]:
    """
    Create an NPIndex with content from each file in the given directories
    and embeddings from the given embedder.

    Args:
        paths_to_directories: List of paths to directories containing files.
        embedder: SentenceTransformer model to use for embedding.
        path_to_save_index: str, self-explanatory and optional.
        index: the index class to use to store embeddings and content.

    Returns:
        The updated index.
    """

    assert isinstance(
        paths_to_directories, list
    ), "paths_to_directories should be a list"

    for path in paths_to_directories:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path '{path}' does not exist.")

    logger.info("paths_to_directories", paths_to_directories)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=0.5, length_function=len
    )

    all_content = []
    all_vectors = []
    for directory in tqdm(paths_to_directories):
        for root, _, files in os.walk(
            directory,
            topdown=True,
            followlinks=True,
        ):
            logger.info(f'Processing directory: "{root}"')
            for file in tqdm(files, desc="Looking into files ..."):
                file_path = os.path.join(root, file)
                logger.info(f"Processing: {file_path}")

                _, ext = os.path.splitext(file_path)
                logger.info(f"Extension: {ext}")

                loader = EXT_TO_LOADER.get(ext)
                if loader:
                    logger.info("We can process this file.")
                    sentences = loader(file_path).load_and_split(text_splitter)
                    content = [x.page_content for x in sentences]

                    if len(content) == 0:
                        logger.info(f"No content in {file_path}")
                        continue

                    vectors = embedder.encode(content)

                    logger.debug("embeddings: ", vectors.shape)

                    all_content += content
                    all_vectors.append(vectors)

    all_vectors = np.concatenate(all_vectors, axis=0)
    index.add(all_vectors, all_content)

    if path_to_save_index:
        logger.info(f"Saving the index to {path_to_save_index}")
        index.save(path_to_save_index)

    return index


class RAG:
    def __init__(
        self,
        embedder: SentenceTransformer,
        index: NPIndex,
        language_model: MLXLLM,
    ):
        self.embedder = embedder
        self.index = index
        self.language_model = language_model

    def retrieve(self, query: str, k: int = 2):
        """
        Retrieve the most similar documents to the given query.
        """
        query_vector = self.embedder.encode([query])
        distances, indices = self.index.retrieve(query_embedding=query_vector, query_text=query, k=k)

        return distances, list(indices.flatten())

    def __call__(
        self, query: str, k: int = 2, show_context: bool = False, thr: float = 0.0
    ):
        """
        Retrieve the most similar documents to the given query,
        concatenate them to a single string, then generate a response
          using the language model.

        Args:
            query: The query to retrieve and generate a response for.
            k: The number of documents to retrieve.
            show_context: Whether to print the context.
            thr: The similarity threshold for the retrieved chunks.

        Returns:
            str: The response from the language model.
        """

        distances, indices = self.retrieve(query, k)
        indices = [ind for ind, dist in zip(indices, distances) if dist > thr]

        if len(indices) == 0:
            context = "No context found in the index."
            query = ""
        else:
            context = " ".join([self.index.content[i] for i in indices])

        if show_context:
            sep = "-" * 100
            print(f"context: {context}\n{sep}\n")

        return self.language_model(context=context, question=query)
