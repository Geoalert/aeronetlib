from .abstractadapter import AbstractAdapter
from .boundsafemixin import BoundSafeMixin


class NumpyAdapter(BoundSafeMixin, AbstractAdapter):
    """Bound-safe adapter for numpy array"""
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

    def write(self, item, data):
        if isinstance(item, list):
            item = tuple(item)
        self._data[item] = data
