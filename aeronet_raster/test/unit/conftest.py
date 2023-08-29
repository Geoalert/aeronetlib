import shutil
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from generate_files import create_tiff_file


@pytest.fixture(scope='session')
def get_file():
    tempdir = TemporaryDirectory()
    path = Path(tempdir.name)
    filename = path / 'input.tif'

    height = 2673
    width = 128
    count = 1
    gen_mode = 'gradient'
    dtype = 'uint8'

    create_tiff_file(filename, width, height, mode=gen_mode, count=count, dtype=dtype)

    yield path

    try:
        shutil.rmtree(path)
    except OSError:
        pass
