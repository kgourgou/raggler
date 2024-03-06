import pytest
from raggler.rag import create_index


# TODO segmentation error
@pytest.mark.parametrize("path_to_dir", ["fake_files/"])
def test_create_index(path_to_dir):
    create_index(path_to_dir, None)
    assert True
