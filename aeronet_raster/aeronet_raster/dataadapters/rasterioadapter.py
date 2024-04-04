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


class RasterioWriter(FileMixin, ImageWriter):
    def __init__(self, path, **profile):
        super().__init__(path=path)
        self.profile = profile

    def open(self):
        self._descriptor = rasterio.open(self._path, 'w', **self.profile)
        self._shape = self._descriptor.count, self._descriptor.shape[0], self._descriptor.shape[1]

    def write(self, item, data):
        channels, y, x = item
        self._descriptor.write(data, [ch+1 for ch in channels],
                               window=((y.start, y.stop),
                                       (x.start, x.stop)))
