from .abstractadapter import AbstractReader, AbstractWriter, PaddedReaderMixin, PaddedWriterMixin
import numpy as np
import rasterio


class RasterioReader(PaddedReaderMixin, AbstractReader):
    """Provides numpy array-like interface to geotiff file via rasterio"""

    def __init__(self, path, verbose: bool = False, padding_mode: str = 'reflect', **kwargs):
        super().__init__(padding_mode)
        self._path = path
        self._data = rasterio.open(path)
        self.verbose = verbose
        self._shape = np.array((self._data.count, self._data.shape[0], self._data.shape[1]))

    def fetch(self, item):
        res = self._data.read([ch+1 for ch in item[0]],
                              window=((item[1].start, item[1].stop),
                                      (item[2].start, item[2].stop)),
                              boundless=True).astype(np.uint8)
        return res

    def parse_item(self, item):
        item = super().parse_item(item)
        assert len(item) == 3, f"Rasterio geotif must be indexed with 3 axes, got {item}"
        if isinstance(item[0], slice):
            item[0] = list(range(item[0].start, item[0].stop, item[0].step))
        assert isinstance(item[1], slice) and isinstance(item[2], slice),\
            f"Rasterio geotif spatial dimensions (1, 2) must be indexed with slices, got {item}"

        return item


class RasterioWriter(AbstractWriter):
    def __init__(self, path, **profile):
        self._path = path
        self._data = rasterio.open(path, 'w', **profile)
        self._shape = np.array((self._data.count, self._data.shape[0], self._data.shape[1]))

    def write(self, item, data):
        self._data.write(data, [ch+1 for ch in item[0]],
                         window=((item[1].start, item[1].stop),
                                 (item[2].start, item[2].stop)))

    def parse_item(self, item):
        item = super().parse_item(item)
        assert len(item) == 3, f"Rasterio geotif must be indexed with 3 axes, got {item}"
        if isinstance(item[0], slice):
            item[0] = list(range(item[0].start, item[0].stop, item[0].step))
        assert isinstance(item[1], slice) and isinstance(item[2], slice),\
            f"Rasterio geotif spatial dimensions (1, 2) must be indexed with slices, got {item}"

        return item
