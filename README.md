# raggler_dev

My own experiments with RAG

# Installing

The code currently uses MLX language models, but the code can be hacked to remove this dependency.

I personally use `uv` to manage python requirements, so I would do the following:

```bash
pip install uv 
uv venv 
source .venv/bin/activate 
uv pip install pip -U 
uv pip install -e .
```

You can also just use `pip` directly, but I would recommend using a virtual environment: `pip install -e .`

# Usage
