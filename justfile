TESTS := "tests/"
SRC_DIR := "raggler/"
EXAMPLES := "notebooks/"

format: 
    ruff format {{SRC_DIR}} {{TESTS}} {{EXAMPLES}}

check: 
    ruff check {{SRC_DIR}} {{TESTS}} {{EXAMPLES}} --fix

test: 
    pytest {{TESTS}} --cov={{SRC_DIR}} --cov-report=term

install: 
    uv pip install pip -U 
    uv pip install -e . 

install_dev:
    uv pip install pip -U 
    uv pip install -e . 
    uv pip install -r dev_requirements.txt