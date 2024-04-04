import numpy as np


class BoundSafeReaderMixin:
    """
    Redefines __getitem__() so it works even if the coordinates are out of bounds
    """
    def __init__(self, padding_mode: str = 'constant', **kwargs):
        super().__init__(**kwargs)
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


class BoundSafeWriterMixin:
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
                crops.append((max(-coords.start, 0), max(coords.stop - self.shape[axis], 0)))
                safe_coords.append(slice(coords.start + crops[-1][0], coords.stop - crops[-1][1], coords.step))

        self.write(safe_coords,
                   data[tuple(slice(crops[i][0], data.shape[i]-crops[i][1], 1) for i in range(data.ndim))])

