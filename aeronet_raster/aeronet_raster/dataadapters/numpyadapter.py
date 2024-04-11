from .abstractadapter import AbstractReader, AbstractWriter
from .boundsafemixin import BoundSafeReaderMixin, BoundSafeWriterMixin


class NumpyReader(BoundSafeReaderMixin, AbstractReader):
    """Works with numpy arrays. Useful for testing"""
    def __init__(self, data, padding_mode='constant', **kwargs):
        super().__init__(padding_mode, **kwargs)
        self._data = data

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    def __len__(self):
        return self.shape[0]

    def fetch(self, item):
        if isinstance(item, list):
            item = tuple(item)
        return self._data[item]


class NumpyWriter(BoundSafeWriterMixin, NumpyReader):
    """Works with numpy arrays. Useful for testing"""
    def write(self, item, data):
        if isinstance(item, list):
            item = tuple(item)
        self._data[item] = data
