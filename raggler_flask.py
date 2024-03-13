"""
From the great work of Micky Multani: 

https://github.com/mickymultani/Streaming-LLM-Chat/tree/main

"""

from flask import Flask, render_template, request, Response
from flask_cors import CORS  # Import CORS
from sentence_transformers import SentenceTransformer
import os
from raggler import RAG
from raggler.llm_context import RAG_TEMPLATE
from raggler.mlx_llm import MLXLLM
from raggler.indexes import NPIndex

app = Flask(__name__)
CORS(app)
rag = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    query = request.json.get("message")
    response = rag(query, k=2, show_context=False, thr=0.0)
    return Response(response, content_type="text/plain")


def load_models(
    embedder="all-MiniLM-L12-v2",
    mlx_llm_name="mlx-community/AlphaMonarch-7B-mlx-4bit",
):
    embedder = SentenceTransformer(embedder)

    default_path_for_index = os.path.join("data/indexes/")
    index = NPIndex()
    index.load(default_path_for_index)
    rag = RAG(embedder, index, MLXLLM(mlx_llm_name, RAG_TEMPLATE))
    return rag


if __name__ == "__main__":
    rag = load_models()
    app.run(debug=True)
