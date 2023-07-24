import os
import re
import glob
from warnings import warn
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
    paths = glob.glob(os.path.join(directory, '*'))
    extensions = '|'.join(extensions)
    res = []
    for name in names:
        # the channel name must be either full filename (that is, ./RED.tif) or a part after '_' (./dse_channel_RED.tif)
        sep = os.sep if os.sep != '\\' else '\\\\'
        pattern = '.*(^|{}|_)({})\.({})$'.format(sep, name, extensions)
        band_path = [path for path in paths if re.match(pattern, path) is not None]

        # Normally with our datasets it will never be the case, and may indicate wrong file naming
        if len(band_path) > 1:
            warn(RuntimeWarning(
                "There are multiple files matching the channel {}. "
                "It can cause ambiguous behavior later.".format(name)))
        res += band_path

    return res


def band_shape_guard(raster: np.ndarray) -> np.ndarray:
    raster = raster.squeeze()
    if raster.ndim != 2:
        raise ValueError('Raster file has wrong shape {}. '.format(raster.shape) +
                         'Should be 2 dim np.array')
    return raster


def random_name(length: int = 10) -> str:
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))
