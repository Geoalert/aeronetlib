import os
import numpy as np
from .band import Band

from .geoobject import GeoObject


class BandCollection(GeoObject):

    def __init__(self, fps):

        super().__init__()

        self._bands = [Band(fp) for fp in fps]
        if not self.is_valid:
            del self
            raise ValueError('Bands are not suitable for collection. '
                             'CRS, transform or shape are different!')

    def __repr__(self):
        names = [b.name for b in self._bands]
        return f'<BandCollection: names={names}'

    @property
    def crs(self):
        return self._bands[0].crs

    @property
    def transform(self):
        return self._bands[0].transform

    @property
    def nodata(self):
        return self._bands[0].nodata

    @property
    def height(self):
        return self._bands[0].height

    @property
    def width(self):
        return self._bands[0].width

    @property
    def count(self):
        return len(self._bands)

    @property
    def shape(self):
        return self.count, self._bands[0].height, self._bands[0].width

    @property
    def res(self):
        return self._bands[0].res

    @property
    def is_valid(self):
        """Check if all bands have the same resolution, shape and coordinate system"""
        if len(self._bands) < 2:
            res = True
        else:
            first = self._bands[0]
            rest = self._bands[1:]
            res = all(first.same(other) for other in rest)

        return res

    def append(self, other):
        """Add band to collection"""
        if all(other.same(band) for band in self._bands):
            self._bands.append(other)
        else:
            raise ValueError('Band is not suitable for collection. '
                             'CRS, transform or shape are different!')
        return

    def sample(self, y, x, height, width):
        """
        Sample memory object from BandCollection
        Args:
            x: int, top left corner `X` offset in pixels
            y: int, top left corner `Y` offset in pixels
            width: int, width of samples in pixels
            height: int, height of samples in pixels

        Returns:
            BandCollectionSample instance
        """
        samples = [band.sample(y, x, height, width) for band in self._bands]
        return BandCollectionSample(samples)

    def generate_samples(self, height, width):
        for x in range(0, self.width, width):
            for y in range(0, self.height, height):
                yield self.sample(y, x, height, width)


class BandCollectionSample(GeoObject):

    def __init__(self, samples):
        super().__init__()
        self._samples = samples

    def __repr__(self):
        names = [b.name for b in self._samples]
        return f'<BandCollectionSample: names={names}>'

    @property
    def crs(self):
        return self._samples[0].crs

    @property
    def transform(self):
        return self._samples[0].transform

    @property
    def nodata(self):
        return self._samples[0].nodata

    @property
    def width(self):
        return self._samples[0].width

    @property
    def height(self):
        return self._samples[0].height

    @property
    def count(self):
        return self._samples[0].count

    @property
    def shape(self):
        return self.count, self.height, self.width

    @property
    def res(self):
        return self._samples[0].res

    @property
    def is_valid(self):
        """Check if all bands have the same resolution, shape and coordinate system"""
        if len(self._samples) < 2:
            res = True
        else:
            first = self._samples[0]
            rest = self._samples[1:]
            res = all(first.same(other) for other in rest)

        return res

    def _get_sample(self, name):
        for s in self._samples:
            if s.name == name:
                return s
        raise NameError(f'No sample with name {name}.')

    def append(self, obj):
        """Add sample to collection"""
        if all(obj.same(band) for band in self._samples):
            self._samples.append(obj)
        else:
            raise ValueError('Band is not suitable for collection. '
                             'CRS, transform or shape are different!')
        return

    def sample(self, y, x, height, width):
        samples = [obj.sample(y, x, height, width) for obj in self._samples]
        return BandCollectionSample(samples)

    def resample(self, dst_res=None, dst_crs=None, dst_shape=None, interpolation='bilinear'):
        resamples = [s.resample(dst_res, dst_crs, dst_shape, interpolation) for s in self._samples]
        return BandCollectionSample(resamples)

    def numpy(self):
        return np.concatenate([x.numpy() for x in self._samples])

    def ordered(self, *names):
        """

        Args:
            *names: order of names

        Returns:
            reordered `BandCollectionSample`
        """
        ordered_bands = [self._get_sample(name) for name in names]
        return BandCollectionSample(ordered_bands)

    def save(self, directory, extension='.tif', **kwargs):
        os.makedirs(directory, exist_ok=True)
        for s in self._samples:
            s.save(directory, extension, **kwargs)
