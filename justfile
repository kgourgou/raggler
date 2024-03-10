TESTS := "tests/"
SRC_DIR := "raggler/"
EXAMPLES := "notebooks/"

format: 
    ruff format {{SRC_DIR}} {{TESTS}} {{EXAMPLES}}

check: 
    ruff check {{SRC_DIR}} {{TESTS}} {{EXAMPLES}} 

test: 
    pytest {{TESTS}} --cov={{SRC_DIR}} --cov-report=term

install: 
    pip install pip -U 
    pip install -e . 

install_dev:
    pip install pip -U 
    pip install -e . 
    pip install -r dev_requirements.txt