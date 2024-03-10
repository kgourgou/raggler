# raggler_dev

Point this at your files and enjoy some simple RAG (retrieval augmented generation). I mainly build this to help me quickly send questions to my obsidian vault, but it should work for any text files.

Why build this?

1. Most RAG systems are built for large-scale retrieval and generation, but this is a simple, lightweight system that should work well for small-scale retrieval and generation and should be out-of-the-box.
2. Most RAG systems start with the assumption of access to a SOTA LLM like GPT-4, Claude, etc., but this system uses a simple LLM that is fast and works well on Apple Silicon (7b quantized models for the win).

There is no chat interface to this system, but it should be easy to add one. I mainly use this as a command-line tool and as a library.

## Installing

**The code currently uses MLX language models, so you will need Apple Silicon (M1, M1 Pro, etc.).** However, it should be simple to change the code to use other language models.

Make a virtual environmen first. Then install the package with:

```bash
pip install -e . 
```

or, if you have the [*just* command runner](https://github.com/casey/just) installed,

```bash
just install
```

To get the dev requirements, run

```bash
pip install -r dev_requirements.txt
```

or, if you are impatient and just want to install everything,

```bash
just install-dev
```

## Usage

### Point-and-Rag in CLI

Raggler is mostly a "point at your files and rag" library.

If you have all of your files in the same directory, do something like this:

```bash
export RAGGLER_DIR=/path/to/your/files
./raggler.py 'A query for your files'
```

You can also store that environment variable in a local .env file.

The first time you run this, it will take a while to index your files. After that, it should be pretty fast as the language-model will be cached locally as a pickle file. Your index will also be saved locally as a pickle file for fast retrieval. All of that will be stored under `data/`.

### As a library

You can also use Raggler as a library. Here's an example:

```python
from raggler import create_index, rag 
```

## Development

This is on purpose a very simple system. At the moment it uses very small models and is not very sophisticated. However, it should be easy to add more sophisticated models and features for your use-case.

## Acknowledgements

- [Hugging Face](https://huggingface.co/) for hosting language models and embedders.
- The [MLX team](https://github.com/ml-explore/mlx) and community for wrapping models nicely and allowing for fast inference on Apple Silicon.
- Maxime Labonne for creating the [NeuralBeagle](https://huggingface.co/mlabonne/NeuralBeagle14-7B) model which handles the query-answering part of the pipeline.
- [Chat-with-MLX](https://github.com/qnguyen3/chat-with-mlx) and [MLX-RAG](https://github.com/vegaluisjose/mlx-rag) for inspiration and code snippets.
