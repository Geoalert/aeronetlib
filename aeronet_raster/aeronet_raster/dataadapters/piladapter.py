from .abstractadapter import AbstractReader
import numpy as np
import pkg_resources

if 'PIL' in {pkg.key for pkg in pkg_resources.working_set}:
    from PIL import Image


class PilReader(AbstractReader):
    """Provides numpy array-like interface to PIL-compatible image file"""
    __slots__ = ('_path',)

    def __init__(self, path, verbose: bool = False, **kwargs):
        self._path = path
        self.verbose = verbose
        self._shape = np.array(Image.open(path)).transpose(2, 0, 1).shape

    def __getitem__(self, item):
        channels, y, x = self.parse_item(item)
        res = np.array(Image.open(self._path)).transpose(2, 0, 1)[channels, y.start:y.stop, x.start:x.stop]
        return res
