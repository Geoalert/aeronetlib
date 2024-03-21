import numpy as np
from ..utils.utils import validate_coord


class PaddedReaderMixin:
    """
    Redefines __getitem__() so it works even if the coordinates are out of bounds
    """
    def __init__(self, padding_mode: str = 'reflect'):
        self.padding_mode = padding_mode

    def __getitem__(self, item):
        item = self.parse_item(item)

        pads, safe_coords = list(), list()
        for axis, coords in enumerate(item):
            # coords can be either slice or tuple at this point (after parse_item)
            if isinstance(coords, (list, tuple)):  # coords = (coord1, coord2, ...), already sorted
                pads.append((0, 0))  # do nothing since indexing out of bounds makes sense only with slices
                safe_coords.append(coords)
            elif isinstance(coords, slice):  # coords = (min:max:step)
                pads.append((max(-coords.start, 0), max(coords.stop - self.shape[axis], 0)))
                safe_coords.append(slice(coords.start + pads[-1][0], coords.stop - pads[-1][1], coords.step))
            else:
                raise ValueError(f'Can not parse coords={coords} at axis={axis}')

        res = self.fetch(safe_coords)
        return np.pad(res, pads, mode=self.padding_mode)


class PaddedWriterMixin:
    """
    Redefines __setitem__() so it works even if the coordinates are out of bounds
    """
    def __setitem__(self, item, data):
        item = self.parse_item(item)
        assert data.ndim == self.ndim == len(item)
        safe_coords, crops = list(), list()
        for axis, coords in enumerate(item):
            # coords can be either slice or tuple at this point (after parse_item)
            if isinstance(coords, (list, tuple)):  # coords = (coord1, coord2, ...), already sorted
                crops.append((0, 0))  # do nothing since indexing out of bounds makes sense only with slices
                safe_coords.append(coords)
            elif isinstance(coords, slice):  # coords = (min:max:step)
                crops.append((max(-coords.start, 0), max(coords.stop - data.shape[axis], 0)))
                safe_coords.append(slice(coords.start + crops[-1][0], coords.stop - crops[-1][1], coords.step))

        self.write(safe_coords,
                   data[tuple(slice(crops[i][0], data.shape[i]-crops[i][1], 1) for i in range(data.ndim))])


class AbstractReader:
    """Provides numpy array-like interface to arbitrary source of data"""
    def __getattr__(self, item):
        print(item)
        return getattr(self._data, item)

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    def __getitem__(self, item):
        item = self.parse_item(item)
        return self.fetch(item)

    def fetch(self, item):
        """Datasource-specific data fetching, e.g. rasterio.read()"""
        raise NotImplementedError

    def parse_item(self, item):
        """Parse input for __getitem__() to handle arbitrary input
        Possible cases:
           - item is a single value (int) -> turns it into a tuple and adds the slice over the whole axis for every
             missing dimension
           - len(item) < self.ndim -> adds the slice over the whole axis for every missing dimension
           - len(item) > self.ndim -> raises IndexError
           - item contains slices without start or step defined -> defines start=0, step=1
           - item contains negative indexes -> substitute them with (self.shape[axis] - index)
        """
        if isinstance(item, (list, np.ndarray)):
            item = tuple(item)
        if not isinstance(item, tuple):
            item = (item, )
        if len(item) > self.ndim:
            raise IndexError(f"Index={item} has more dimensions than data={self.shape}")
        item = list(item)
        while len(item) < self.ndim:
            item.append(None)

        for axis, coord in enumerate(item):
            item[axis] = validate_coord(coord, self.shape[axis])
        return item


class AbstractWriter(AbstractReader):
    """Provides numpy array-like interface to arbitrary source of data"""
    def __getitem__(self, item):
        raise NotImplementedError

    def fetch(self, item):
        raise NotImplementedError

    def __setitem__(self, item, data):
        item = self.parse_item(item)
        self.write(item, data)

    def write(self, item, data):
        raise NotImplementedError
