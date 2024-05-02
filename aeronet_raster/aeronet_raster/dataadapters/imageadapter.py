from .abstractadapter import AbstractAdapter
from .boundsafemixin import BoundSafeMixin


class ImageAdapter(BoundSafeMixin, AbstractAdapter):
    """Abstract class. Redefines parse_item() so that it works with 3-dimensional data (channels, height, width),
    allows indexing channels with Sequence[int], spatial dimensions with slices"""
    def parse_item(self, item):
        item = super().parse_item(item)
        if not len(item) == 3:
            raise ValueError(f"Image must be indexed with 3 axes, got {item}")
        if isinstance(item[0], slice):
            item[0] = list(range(item[0].start, item[0].stop, item[0].step))
        assert isinstance(item[1], slice) and isinstance(item[2], slice),\
            f"Image spatial axes (1 and 2) must be indexed with slices, got {item}"
        return item

    @property
    def ndim(self):
        return 3
