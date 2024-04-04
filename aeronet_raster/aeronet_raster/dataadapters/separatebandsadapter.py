from .imageadapter import ImageReader, ImageWriter
from .abstractadapter import AbstractReader, AbstractWriter
from .filemixin import FileMixin
from .rasterioadapter import RasterioReader, RasterioWriter
import numpy as np
from typing import Sequence


class SeparateBandsReader(ImageReader):
    """Provides numpy array-like interface to separate data sources (image bands)"""

    def __init__(self, bands: Sequence[ImageReader], padding_mode='constant'):
        super().__init__(padding_mode=padding_mode)
        self._data = bands
        for b in self._data:
            b.padding_mode = padding_mode
        self._shape = None

    def open(self):
        for d in self._data:
            if hasattr(d, 'open'):
                d.open()

        self._channels = [(0, i) for i in range(self._data[0].shape[0])]
        if len(self._data) > 1:
            for i, b in enumerate(self._data[1:]):
                self._channels.extend([(i + 1, j) for j in range(b.shape[0])])
                if np.any(b.shape[1:] != self._data[0].shape[1:]):
                    raise ValueError(f'Band {i} shape = {b.shape[1:]} != Band 0 shape = {self._data[0].shape[1:]}')
        self._shape = len(self._channels), self._data[0].shape[1], self._data[0].shape[2]

    def close(self):
        for d in self._data:
            if hasattr(d, 'close'):
                d.close()
        self._shape = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        self.close()

    def fetch(self, item):
        res = list()
        for ch in item[0]:
            idx, src_ch = self._channels[ch]
            res.append(self._data[idx][src_ch, item[1], item[2]])
        return np.concatenate(res, 0)

    def parse_item(self, item):
        item = super().parse_item(item)
        if not len(item) == 3:
            raise ValueError(f"Image must be indexed with 3 axes, got {item}")
        if isinstance(item[0], slice):
            item[0] = list(range(item[0].start, item[0].stop, item[0].step))
        assert isinstance(item[1], slice) and isinstance(item[2], slice), \
            f"PIL Image spatial axes (1 and 2) must be indexed with slices, got {item}"
        return item


class SeparateBandsWriter(ImageWriter):
    """Provides numpy array-like interface to separate adapters, representing image bands"""

    def __init__(self, bands: Sequence[ImageWriter]):
        self._data = bands
        self._channels = [(0, i) for i in range(bands[0].shape[0])]
        if len(bands) > 1:
            for i, b in enumerate(bands[1:]):
                self._channels.extend([(i+1, j) for j in range(b.shape[0])])
                if np.any(b.shape[1:] != bands[0].shape[1:]):
                    raise ValueError(f'Band {i} shape = {b.shape[1:]} != Band 0 shape = {bands[0].shape[1:]}')
        self._shape = (len(self._channels), self._data[0].shape[1], self._data[0].shape[2])

    def write(self, item, data):
        assert len(data) == len(item[0])
        for data_ch, ch in enumerate(item[0]):
            idx, i = self._channels[ch]
            self._data[idx][i, item[1], item[2]] = np.expand_dims(data[data_ch], 0)


def get_reader_from_rasterio_bands(bands, padding_mode='constant'):
    return SeparateBandsReader([RasterioReader(b, padding_mode) for b in bands])


def get_writer_to_rasterio_bands(bands):
    return SeparateBandsWriter([RasterioWriter(b) for b in bands])
