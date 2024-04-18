from .imageadapter import ImageAdapter
from .filemixin import FileMixin
import numpy as np
import rasterio

RASTERIO_OPEN_MODES = {'r', 'r+', 'w', 'w+'}


class RasterioAdapter(FileMixin, ImageAdapter):
    """Provides numpy array-like interface to geotiff file via rasterio"""

    def __init__(self, path, mode='r', profile=None, padding_mode: str = 'constant', **kwargs):
        super().__init__(path=path, padding_mode=padding_mode)
        if mode not in RASTERIO_OPEN_MODES:
            raise ValueError(f'Mode must be one of {RASTERIO_OPEN_MODES}')
        if mode.startswith('w') and not profile:
            raise ValueError(f'Profile must be specified for mode={mode}')
        self._mode = mode
        self._profile = profile

    def open(self):
        if self._mode.startswith('w'):
            self._descriptor = rasterio.open(self._path, self._mode, **self._profile)
        else:
            self._descriptor = rasterio.open(self._path, self._mode)
            self._profile = self._descriptor.profile
        self._shape = self._descriptor.count, self._descriptor.shape[0], self._descriptor.shape[1]

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return 3

    def __len__(self):
        return self._shape[0]

    @property
    def profile(self):
        return self._profile

    @property
    def crs(self):
        if not self._descriptor:
            raise ValueError(f'File {self._path} is not opened')
        return self._descriptor.crs

    @property
    def res(self):
        if not self._descriptor:
            raise ValueError(f'File {self._path} is not opened')
        return self._descriptor.res

    @property
    def count(self):
        if not self._descriptor:
            raise ValueError(f'File {self._path} is not opened')
        return self._descriptor.count

    @property
    def dtype(self):
        if not self._descriptor:
            raise ValueError(f'File {self._path} is not opened')
        return self._descriptor.profile['dtype']

    def fetch(self, item):
        channels, y, x = item
        return self._descriptor.read([ch+1 for ch in channels],
                                     window=((y.start, y.stop),
                                             (x.start, x.stop)))

    def write(self, item, data):
        channels, y, x = item
        self._descriptor.write(data, [ch+1 for ch in channels],
                               window=((y.start, y.stop),
                                       (x.start, x.stop)))
