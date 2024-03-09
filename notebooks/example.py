import logging

logging.basicConfig(level=logging.INFO)

from sentence_transformers import SentenceTransformer
from raggler.rag import create_index


model = SentenceTransformer("paraphrase-albert-small-v2")

index = create_index(
    ["../tests/fake_files/", "../tests/more_fake_files/"],
    embedder=model,
    path_to_save_index="../tests/test_index/",
)

from sentence_transformers import SentenceTransformer

from raggler.rag import RAG

from raggler.mlx_llm import MLXLLM
from raggler.llm_context import RAG_TEMPLATE

model = SentenceTransformer("paraphrase-albert-small-v2")

test = MLXLLM("mlx-community/NeuralBeagle14-7B-4bit-mlx", prompt_template=RAG_TEMPLATE)


# TODO Too much input; we can give one path for both index and content.
rag = RAG(embedder=model, index=index, language_model=test)
