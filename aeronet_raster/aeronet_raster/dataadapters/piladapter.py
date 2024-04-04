from .imageadapter import ImageReader, ImageWriter
from .filemixin import FileMixin
import numpy as np
import pkg_resources

if 'pillow' in {pkg.key for pkg in pkg_resources.working_set}:
    from PIL import Image


class PilReader(FileMixin, ImageReader):
    """Provides numpy array-like interface to PIL-compatible image file."""

    def __init__(self, path, padding_mode='constant', **kwargs):
        super().__init__(path=path, padding_mode=padding_mode)

    def open(self):
        self._descriptor = Image.open(self._path)
        self._shape = len(self._descriptor.getbands()), self._descriptor.height, self._descriptor.width

    def fetch(self, item):
        if not self._descriptor:
            raise ValueError(f'File {self._path} is not opened')
        channels, y, x = item
        return np.array(self._descriptor.crop((x.start, y.start, x.stop, y.stop))).transpose(2, 0, 1)[channels]


class PilWriter(FileMixin, ImageWriter):
    """Provides numpy array-like interface to PIL-compatible image file."""

    def __init__(self, path, shape):
        super().__init__(path=path)
        if not shape[0] in (1, 3, 4):
            raise ValueError(f'Only 1, 3, and 4 channels supported, got {shape}')
        self._shape = shape

    def open(self):
        self._descriptor = np.zeros(self._shape, dtype=np.uint8)

    def close(self):
        Image.fromarray(self._descriptor.transpose(1, 2, 0)).save(self._path)

    def write(self, item, data):
        if not self._descriptor:
            raise ValueError(f'Writer is not opened')
        channels, y, x = item
        self._descriptor[channels, y.start:y.stop, x.start:x.stop] = data
