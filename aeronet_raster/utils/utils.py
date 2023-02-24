import os
import re
import glob
import sys
from warnings import warn
from numbers import Number
import numpy as np
from typing import Union, Optional, Final
import string
import random


NumericalSeq = Union[list, tuple, np.ndarray]
TMP_DIR: Final[str] = '/tmp/raster'


class Logger:
    @staticmethod
    def debug(s):
        print(s, file=sys.stdout)

    @staticmethod
    def warning(s):
        print(s, file=sys.stderr)


def to_np_2(value) -> np.ndarray:
    """
    converts any value to (2) shaped np array if possoble
    """
    assert isinstance(value, (Number, NumericalSeq)), f"{value} must be convertable to np.array of shape=(2)"
    if isinstance(value, Number):
        value = (value, value)
    assert len(value) == 2, f"Length {value} = {len(value)} != 2"
    return np.array(value).astype(int)


def to_np_2_2(value) -> np.ndarray:
    """
    converts any value to (2, 2) shaped np array if possoble
    """
    assert isinstance(value, (Number, NumericalSeq))
    if isinstance(value, Number):
        value = (value, value)
    if isinstance(value, (list, tuple)):
        if len(value) == 2:
            value = np.array([[0, 0], [value[0], value[1]]])
        elif len(value) == 4:
            value = np.array([[value[0], value[1]], [value[2], value[3]]])
        else:
            raise AssertionError(f"{value} must be convertable to np.array of shape=(2,2)")
    if value.ndim == 1:
        value = np.array([[0, 0], [value[0], value[1]]])
    assert value.shape == (2, 2), f"Shape {value} = {value.shape} != (2, 2)"
    return value.astype(int)


def to_tuple(value):
    """
    Returns empty tuple if value is none, tuple(value) if value is Number
    """
    if not value:
        return tuple()
    if isinstance(value, Number):
        return value,
    if isinstance(value, np.ndarray):
        return tuple(map(tuple, value))
    return value


def parse_directory(directory: str, names: tuple[str], extensions: tuple[str] = ('tif', 'tiff', 'TIF', 'TIFF')):
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


def band_shape_guard(raster):
    raster = raster.squeeze()
    if raster.ndim != 2:
        raise ValueError('Raster file has wrong shape {}. '.format(raster.shape) +
                         'Should be 2 dim np.array')
    return raster


def random_name(length=10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))
