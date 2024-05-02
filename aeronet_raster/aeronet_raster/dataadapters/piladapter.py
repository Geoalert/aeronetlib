from .imageadapter import ImageAdapter
from .filemixin import FileMixin
import numpy as np
import pkg_resources

if 'pillow' in {pkg.key for pkg in pkg_resources.working_set}:
    from PIL import Image


class PilAdapter(FileMixin, ImageAdapter):
    """Provides numpy array-like interface to PIL-compatible image file."""

    def open(self):
        self._descriptor = Image.open(self._path)
        self._shape = len(self._descriptor.getbands()), self._descriptor.height, self._descriptor.width

    def fetch(self, item):
        if not self._descriptor:
            raise ValueError(f'File {self._path} is not opened')
        channels, y, x = item
        return np.array(self._descriptor.crop((x.start, y.start, x.stop, y.stop))).transpose(2, 0, 1)[channels]

    def write(self, key, value):
        raise AttributeError('PIL Image is not writable. Use NumpyAdapter and save it as Image ')

    @property
    def dtype(self):
        return np.uint8

    @property
    def ndim(self):
        return 3

    @property
    def shape(self):
        return self._shape

    def __len__(self):
        return self._shape[0]

