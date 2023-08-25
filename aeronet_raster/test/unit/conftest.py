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
    sample_size = (938, 938)
    bound = 150
    bound_mode = 'drop'
    gen_mode = 'gradient'
    input_channels = ['input']
    output_labels = ['output']
    count = len(input_channels)
    dst_dtype = 'float32'
    src_nodata = 0
    dst_nodata = 0
    padding = 'none'

    create_tiff_file(filename, width, height, mode=gen_mode, count=count)

    yield path, sample_size, bound, bound_mode, dst_dtype, input_channels, output_labels, padding, src_nodata

    try:
        shutil.rmtree(path)
    except OSError:
        pass
