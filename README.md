# raggler

Point this at your files and enjoy some simple RAG (retrieval augmented generation). I mainly built this to quickly send questions to my [obsidian](https://obsidian.md/) vault, but it should work for any plain text files / markdown. Raggler will also try to chunk your PDFs.

## See what this thing can do

**The code currently uses MLX language models, so you will need Apple Silicon (M1, M1 Pro, etc.).** However, it should be simple to change the code to use other language models.

Make a virtualenv (optional, but recommended), clone this repo, navigate to the cloned directory, and then run:

```bash
pip install .
python3 raggler.py 'Give me a geometry problem and then suggest a variation of it.' --files "tests/fake_files/" --ctx --rfr
```

It will take a while to download the LLM, but it will be cached for future use by the HF library.

## Installing

Make a virtual environment first. Clone this repo, navigate to the cloned directory, then install the package in editable mode with:

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

The first time you run raggler.py, it will take a while to index your files. Your index will also be saved locally as a pickle file for fast retrieval under `data/`.

A few pointers:

1. You don't need to refresh the index every time (except if your files have changed), but it also won't happen automatically when the files change.
2. You can use the `--show_context` flag to see the context of the answer.

### As a library

You can also use raggler as a library; see `notebooks/`.

I've tested it on a 16Gb M1 Pro with a few hundred files and python 3.11 and it works OK.

## Acknowledgements

- All chunking is possible thanks to [LangChain](https://www.langchain.com/).
- [Hugging Face](https://huggingface.co/) for hosting language models and embedders.
- The [MLX team](https://github.com/ml-explore/mlx) and community for allowing for fast inference on Apple Silicon.
- Maxime Labonne for creating the [AlphaMonarch](https://huggingface.co/mlabonne/AlphaMonarch-7B) model which handles the query-answering part of the pipeline.
- [Chat-with-MLX](https://github.com/qnguyen3/chat-with-mlx) and [MLX-RAG](https://github.com/vegaluisjose/mlx-rag) for the inspiration.
