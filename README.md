# raggler

Point this at your files and enjoy some simple RAG (retrieval augmented generation). I mainly buildt this to quickly send questions to my obsidian vault, but it should work for any plain text files / markdown. Raggler will also try to chunk your PDFs.

## See what this thing can do

**The code currently uses MLX language models, so you will need Apple Silicon (M1, M1 Pro, etc.).** However, it should be simple to change the code to use other language models.

Make a virtualenv (optional, but recommended) and then run:

```bash
pip install -e .
python3 raggler.py 'Give me a geometry problem about planes.' --path_to_files "tests/fake_files/" --show_contex
t --refresh_index
```

## Installing

Make a virtual environment first. Then install the package with:

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
python raggler.py 'A query for your files' --refresh_index 
```

You can also store RAGGLER_DIR in a local .env file within the project directory.

```bash
echo "RAGGLER_DIR=/path/to/your/files" > .env
```

The first time you run raggler.py, it will take a while to index your files. After that, it should be pretty fast as the language-model will be cached locally as a pickle file. Your index will also be saved locally as a pickle file for fast retrieval. All of that will be stored under `data/`.

A few pointers:

1. You don't need to refresh the index every time, but it also won't happen automatically when the files change.
2. You can use the `--show_context` flag to see the context of the answer.

### As a library

You can also use Raggler as a library; see the corresponding notebook in `notebooks/`.

## Development

This is on purpose a very simple system. At the moment it uses small models and is not very sophisticated.

- There is no chat interface, but nothing is stopping you from building one.
- Embeddings are from vanilla sentence-transformers; there is no fine-tuning or query / instructor embeddings used.
- Chunking is done with a recursive splitter.
- The index is a collection of list indexes and embeddings held in a numpy array (but watch this space for more exotic index methods soon). There are more scalable ways for index storage and index search (see FAISS).

## Why build this?

In the taxonomy of RAG systems, raggler is simple enough that you can write it in a few hours, but it can still process a set of directories and give you a respectable RAG system without having to reach for your API keys.

I've tested it on a 16Gb M1 Macbook Pro with a few hundred files and it works OK.

## Acknowledgements

- All chunking is possible thanks to [LangChain](https://www.langchain.com/).
- [Hugging Face](https://huggingface.co/) for hosting language models and embedders.
- The [MLX team](https://github.com/ml-explore/mlx) and community for wrapping models nicely and allowing for fast inference on Apple Silicon.
- Maxime Labonne for creating the [NeuralBeagle](https://huggingface.co/mlabonne/NeuralBeagle14-7B) model which handles the query-answering part of the pipeline.
- [Chat-with-MLX](https://github.com/qnguyen3/chat-with-mlx) and [MLX-RAG](https://github.com/vegaluisjose/mlx-rag) for the inspiration.
