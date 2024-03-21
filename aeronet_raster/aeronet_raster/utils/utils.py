import os
from typing import Final, Tuple, List, Union, Optional
import string
import random
import numpy as np

TMP_DIR: Final[str] = '/tmp/raster'
IntCoords = Union[Tuple[int, int], List[int], np.ndarray]
IntBox = Union[Tuple[IntCoords, IntCoords], List[IntCoords], np.ndarray]


def parse_directory(directory: str, names: Tuple[str],
                    extensions: Tuple[str] = ('tif', 'tiff', 'TIF', 'TIFF')) -> List[str]:
    """
    Extract necessary filenames
    Args:
        directory: str
        names: tuple of str, band or file names, e.g. ['RED', '101']
        extensions: tuple of str, allowable file extensions

    Returns:
        list of matched paths
    """
    paths = []
    for name in names:
        for ext in extensions:
            path = os.path.join(directory, f"{name}.{ext}")
            if os.path.exists(path):
                paths.append(path)
                break
    if len(paths) != len(names):
        raise ValueError('Not all files found')
    return paths


def band_shape_guard(raster: np.ndarray) -> np.ndarray:
    raster = raster.squeeze()
    if raster.ndim != 2:
        raise ValueError('Raster file has wrong shape {}. '.format(raster.shape) +
                         'Should be 2 dim np.array')
    return raster


def random_name(length: int = 10) -> str:
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))


def validate_coord(coord: Union[int, tuple, list, slice, None], dim_size: int) -> Union[List, slice]:
    """
    :param coord: Anything that can be used as an index (single value, sequence or slice)
    :param dim_size: size of current data dimension
    :return: tuple of indexes or valid slice
    """
    if coord is None:
        return slice(0, dim_size, 1)
    if isinstance(coord, int):
        coord = [coord, ]
    if isinstance(coord, tuple):
        coord = list(coord)
    if isinstance(coord, list):
        for i, c in enumerate(coord):
            if -dim_size <= c < 0:
                coord[i] = dim_size + c
            elif 0 <= c < dim_size:
                pass
            else:
                raise IndexError(f'{c} is out of bounds for axis size {dim_size}')
        return sorted(coord)
    if isinstance(coord, slice):
        start, stop, step = coord.start or 0, coord.stop or dim_size, coord.step or 1
        #if not(-dim_size <= start < dim_size) or \
        #   not(-dim_size-1 <= stop <= dim_size) or \
        #   not((step > 0) ^ (stop-start <= 0)) or stop == start:
        #    raise IndexError(f'Invalid slice {start, stop, step} for axis size {dim_size}')
        return slice(start, stop, step)


def to_np_2(value) -> np.ndarray:
    """
    converts any value to (2) shaped np array if possible
    """
    if isinstance(value, slice):
        value = list(range(value.start, value.stop))
    if isinstance(value, (int, float)):
        value = (value, value)
    assert len(value) == 2, f"Length {value} = {len(value)} != 2"
    return np.array(value).astype(int)
