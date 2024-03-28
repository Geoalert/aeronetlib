from .abstractadapter import AbstractReader, AbstractWriter
import numpy as np
from typing import Sequence


class SeparateBandsReader(AbstractReader):
    """Provides numpy array-like interface to separate data sources (image bands)"""

    def __init__(self, bands: Sequence[AbstractReader], verbose: bool = False, **kwargs):
        self._data = bands
        self._channels = [(0, i) for i in range(bands[0].shape[0])]
        if len(bands) > 1:
            for i, b in enumerate(bands[1:]):
                self._channels.extend([(i+1, j) for j in range(b.shape[0])])
                if np.any(b.shape[1:] != bands[0].shape[1:]):
                    raise ValueError(f'Band {i} shape = {b.shape[1:]} != Band 0 shape = {bands[0].shape[1:]}')
        self._shape = (len(self._channels), self._data[0].shape[1], self._data[0].shape[2])

    def fetch(self, item):
        res = list()
        for ch in item[0]:
            idx, src_ch = self._channels[ch]
            res.append(self._data[idx][src_ch, item[1], item[2]])
        return np.concatenate(res, 0)

    def parse_item(self, item):
        item = super().parse_item(item)
        assert len(item) == 3, f"Rasterio geotif must be indexed with 3 axes, got {item}"
        if isinstance(item[0], slice):
            item[0] = list(range(item[0].start, item[0].stop, item[0].step))
        assert isinstance(item[1], slice) and isinstance(item[2], slice),\
            f"Rasterio geotif spatial dimensions (1, 2) must be indexed with slices, got {item}"
        return item


class SeparateBandsWriter(AbstractWriter):
    """Provides numpy array-like interface to separate adapters, representing image bands"""

    def __init__(self, bands: Sequence[AbstractWriter], **kwargs):
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

    def parse_item(self, item):
        item = super().parse_item(item)
        assert len(item) == 3, f"Rasterio geotif must be indexed with 3 axes, got {item}"
        if isinstance(item[0], slice):
            item[0] = list(range(item[0].start, item[0].stop, item[0].step))
        assert isinstance(item[1], slice) and isinstance(item[2], slice),\
            f"Rasterio geotif spatial dimensions (1, 2) must be indexed with slices, got {item}"
        return item