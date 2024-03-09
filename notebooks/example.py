import logging
from sentence_transformers import SentenceTransformer

from raggler.llm_context import RAG_TEMPLATE
from raggler.mlx_llm import MLXLLM
from raggler.rag import RAG


from raggler.rag import create_index

logging.basicConfig(level=logging.INFO)
model = SentenceTransformer("paraphrase-albert-small-v2")

index = create_index(
    ["../tests/fake_files/", "../tests/more_fake_files/"],
    embedder=model,
    path_to_save_index="../tests/test_index/",
)

model = SentenceTransformer("paraphrase-albert-small-v2")
test = MLXLLM("mlx-community/NeuralBeagle14-7B-4bit-mlx", prompt_template=RAG_TEMPLATE)


rag = RAG(embedder=model, index=index, language_model=test)
