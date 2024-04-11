from .imageadapter import ImageWriter, ImageReader
from .filemixin import FileMixin
import numpy as np
import rasterio


class RasterioReader(FileMixin, ImageReader):
    """Provides numpy array-like interface to geotiff file via rasterio"""

    def __init__(self, path, padding_mode: str = 'constant', **kwargs):
        super().__init__(path=path, padding_mode=padding_mode)

    def open(self):
        self._descriptor = rasterio.open(self._path)
        self._shape = self._descriptor.count, self._descriptor.shape[0], self._descriptor.shape[1]

    def fetch(self, item):
        channels, y, x = item
        res = self._descriptor.read([ch+1 for ch in channels],
                                    window=((y.start, y.stop),
                                            (x.start, x.stop)),
                                    boundless=True).astype(np.uint8)
        return res

    @property
    def profile(self):
        if not self._descriptor:
            raise ValueError(f'File {self._path} is not opened')
        return self._descriptor.profile

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


class RasterioWriter(ImageWriter, RasterioReader):
    def __init__(self, path, profile, padding_mode: str = 'constant', **kwargs):
        super().__init__(path=path, padding_mode=padding_mode)
        self.write_profile = profile

    def open(self):
        self._descriptor = rasterio.open(self._path, 'w+', **self.write_profile)
        self._shape = self._descriptor.count, self._descriptor.shape[0], self._descriptor.shape[1]

    def write(self, item, data):
        channels, y, x = item
        self._descriptor.write(data, [ch+1 for ch in channels],
                               window=((y.start, y.stop),
                                       (x.start, x.stop)))
