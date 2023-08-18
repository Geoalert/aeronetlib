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
    sample_size0 = 938
    sample_size = (sample_size0, sample_size0)
    bound = 150
    bound_mode = 'weight'
    gen_mode = 'gradient'
    count = 1
    dst_dtype = 'float32'
    input_channels = ['input']
    output_labels = ['output']

    create_tiff_file(filename, width, height, mode=gen_mode, count=count)

    yield path, sample_size, bound, bound_mode, dst_dtype, input_channels, output_labels

    try:
        shutil.rmtree(path)
    except OSError:
        pass