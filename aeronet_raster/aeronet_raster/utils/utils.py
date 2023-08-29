import os
from typing import Final, Tuple, List
import string
import random
import numpy as np

TMP_DIR: Final[str] = '/tmp/raster'


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
