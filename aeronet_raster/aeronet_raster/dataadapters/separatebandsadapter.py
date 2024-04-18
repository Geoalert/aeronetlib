from .imageadapter import ImageAdapter
from .rasterioadapter import RasterioAdapter
import numpy as np
from typing import Sequence


class SeparateBandsAdapter(ImageAdapter):
    """Provides numpy array-like interface to separate data sources (image bands)"""
    def __init__(self, bands: Sequence[ImageAdapter], padding_mode: str = 'constant', **kwargs):
        super().__init__(padding_mode=padding_mode, **kwargs)
        self._data = bands
        self._channels = list()  # stores mapping channel_idx -> (band_idx, channel_in_band_idx)

        for i, b in enumerate(bands):
            if hasattr(b, 'open'):
                b.open()
            self._channels.extend([(i, j) for j in range(b.shape[0])])
            if i == 0:
                spatial_shape = b.shape[1:]
                self._dtype = b.dtype
            else:
                if np.any(b.shape[1:] != spatial_shape):
                    raise ValueError(f'Band {i} shape = {b.shape[1:]} != Band 0 shape = {spatial_shape}')
                if b.dtype != self._dtype:
                    raise ValueError(f'Band {i} dtype = {b.dtype} != Band 0 dtype = {self._dtype}')
            if hasattr(b, 'close'):
                b.close()
        self._shape = (len(self._channels), spatial_shape[0], spatial_shape[1])

    def open(self):
        for d in self._data:
            if hasattr(d, 'open'):
                d.open()

    def close(self):
        for d in self._data:
            if hasattr(d, 'close'):
                d.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        self.close()

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return 3

    def __len__(self):
        return self._shape[0]

    @property
    def dtype(self):
        return self._dtype

    def fetch(self, item):
        res = list()
        for ch in item[0]:
            idx, src_ch = self._channels[ch]
            res.append(self._data[idx][src_ch, item[1], item[2]])
        return np.concatenate(res, 0)

    def write(self, item, data):
        assert len(data) == len(item[0])
        for data_ch, ch in enumerate(item[0]):
            idx, i = self._channels[ch]
            self._data[idx][i, item[1], item[2]] = np.expand_dims(data[data_ch], 0)


def from_rasterio_bands(bands, mode='r', profile=None, padding_mode='constant'):
    return SeparateBandsAdapter([RasterioAdapter(b,
                                                 mode=mode,
                                                 profile=profile,
                                                 padding_mode=padding_mode) for b in bands])
