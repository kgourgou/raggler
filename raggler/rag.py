"""
Pipeline for retrieval augmented generation. 
"""

import logging
import os

import faiss
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
) -> tuple[NPIndex, list[str]]:
    """
    Create an NPIndex wit content from each file in the given directories
    and embeddings from the given embedder.

    Args:
        paths_to_directories: List of paths to directories containing files.
        embedder: SentenceTransformer model to use for embedding.
        path_to_save_index: str, self-explanatory and optional.

    Returns:
        index
    """

    assert isinstance(
        paths_to_directories, list
    ), "paths_to_directories should be a list"

    print("paths_to_directories", paths_to_directories)

    text_splitter = RecursiveCharacterTextSplitter()
    index = NPIndex()

    all_content = []
    all_vectors = []
    for directory in tqdm(paths_to_directories):
        for root, _, files in os.walk(
            directory,
            topdown=True,
            followlinks=True,
        ):
            print(f'Processing directory: "{root}"')
            for file in files:
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")

                _, ext = os.path.splitext(file_path)
                print(f"Extension: {ext}")

                loader = EXT_TO_LOADER.get(ext)
                if loader:
                    print("We can process this file.")
                    sentences = loader(file_path).load_and_split(text_splitter)
                    content = [x.page_content for x in sentences]
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


def load_index(path_to_index: str):
    """
    Load the FAISS index from the given path.
    """
    faiss_index = faiss.read_index(path_to_index)
    return faiss_index


class RAG:
    def __init__(
        self,
        embedder: SentenceTransformer,
        index: NPIndex,
        language_model: MLXLLM,
    ):
        self.embedder = embedder
        self.index = index
        self.content = index.content
        self.language_model = language_model

    def retrieve(self, query: str, k: int = 2):
        """
        Retrieve the most similar documents to the given query.
        """
        query_vector = self.embedder.encode([query])
        distances, indices = self.index.retrieve(query_vector, k)

        return distances, list(indices.flatten())

    def __call__(self, query: str, k: int = 2, print_context: bool = False):
        """
        Retrieve the most similar documents to the given query,
        concatenate them to a single string, then generate a response using the language model.
        """

        _, indices = self.retrieve(query, k)
        context = " ".join([self.content[i] for i in indices])

        if print_context:
            print(f"context: {context}")

        return self.language_model(context=context, question=query)
