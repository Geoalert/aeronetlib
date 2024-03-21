from .abstractadapter import AbstractReader, AbstractWriter
import numpy as np
from typing import Sequence


class SeparateBandsReader(AbstractReader):
    """Provides numpy array-like interface to separate data sources (image bands)"""

    def __init__(self, bands: Sequence[AbstractReader], verbose: bool = False, **kwargs):
        self._data = bands
        channels = bands[0].shape[0]
        if len(bands) > 1:
            for i, b in enumerate(bands[1:]):
                channels += b.shape[0]
                if b.shape[1:] != bands[0].shape[1:]:
                    raise ValueError(f'Band {i} shape = {b.shape[1:]} != Band 0 shape = {bands[0].shape[1:]}')
        self._shape = (channels, self._data[0].shape[1], self._data[0].shape[2])

    def __getitem__(self, item):
        return np.concatenate([b[item] for b in self._data], 0)


class SeparateBandsWriter(AbstractWriter):
    """Provides numpy array-like interface to separate adapters, representing image bands"""
    __slots__ = ('_channels',)

    def __init__(self, bands: Sequence[AbstractWriter], **kwargs):
        self._data = bands
        channels = bands[0].shape[0]
        if len(bands) > 1:
            for i, b in enumerate(bands[1:]):
                self._channels += b.shape[0]
                if b.shape[1:] != bands[0].shape[1:]:
                    raise ValueError(f'Band {i} shape = {b.shape[1:]} != Band 0 shape = {bands[0].shape[1:]}')
        self._shape = (np.sum(self._channels), self._data[0].shape[1], self._data[0].shape[2])

    def __setitem__(self, item, data):
        current_ch = 0
        for band in self._data:
            band[item] = data[current_ch:current_ch+band.shape[0]]
            current_ch += band.shape[0]
