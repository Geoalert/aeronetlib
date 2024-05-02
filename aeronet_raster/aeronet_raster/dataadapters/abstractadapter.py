import numpy as np
from ..utils.utils import validate_coord


class AbstractAdapter:
    """Base abstract class for adapters. Provides numpy array-like interface for arbitrary data source"""
    @property
    def shape(self):
        raise NotImplementedError

    @property
    def dtype(self):
        raise NotImplementedError

    @property
    def ndim(self):
        raise NotImplementedError

    def __len__(self):
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

    # Read -------------------------------------------------------------------------------------------------------------
    def __getitem__(self, item):
        item = self.parse_item(item)
        return self.fetch(item)

    def fetch(self, item):
        """Datasource-specific data fetching, e.g. rasterio.read()"""
        raise NotImplementedError

    # Write ------------------------------------------------------------------------------------------------------------
    def __setitem__(self, item, data):
        item = self.parse_item(item)
        self.write(item, data)

    def write(self, item, data):
        raise NotImplementedError




