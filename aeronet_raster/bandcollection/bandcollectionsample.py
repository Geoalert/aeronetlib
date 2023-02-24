from ..geoobject.geoobject import GeoObject
from ..utils.coords import get_utm_zone
import numpy as np
import os


class BandCollectionSample(GeoObject):
    """
    A collection of :obj:`BandSample` , which are also
    """

    def __init__(self, samples):
        super().__init__()
        self._samples = samples
        if not self.is_valid:
            raise ValueError('Validity check failed!')

    def __repr__(self):
        names = [b.name for b in self._samples]
        return '<BandCollectionSample: {}>'.format(names)

    def __getitem__(self, item):
        return self._samples[item]

    def __len__(self):
        return self.count

    # ======================== PROPERTY BLOCK ========================

    @property
    def crs(self):
        return self._samples[0].crs

    @property
    def transform(self):
        return self._samples[0].transform

    @property
    def res(self):
        return self._samples[0].res

    @property
    def width(self):
        return self._samples[0].width

    @property
    def height(self):
        return self._samples[0].height

    @property
    def count(self):
        """ Number of bands (layers) in the collection"""
        return len(self._samples)

    @property
    def shape(self):
        return self.count, self.height, self.width

    @property
    def nodata(self):
        return self._samples[0].nodata

    @property
    def bounds(self):
        return self._samples[0].bounds

    @property
    def is_valid(self):
        """Check if all bands have the same resolution, shape and coordinate system"""
        if len(self._samples) == 0:
            return False
        if len(self._samples) == 1:
            return self._samples[0].is_valid
        return all(self._samples[0].same(other) for other in self._samples[1:]) and\
               all(b.is_valid for b in self._samples)

    # ======================== PRIVATE METHODS ========================

    def _get_sample(self, name):
        if isinstance(name, int):
            return self._samples[name]
        for s in self._samples:
            if s.name == name:
                return s
            elif len(s.name) > len(name):  # legacy datasets support
                if s.name.endswith('_{name}'.format(name=name)):
                    return s
            # in all other cases raise error
        raise NameError('No sample with name {name}.'.format(name=name))

    # ======================== PUBLIC METHODS  ========================

    def append(self, obj):
        """
        Add sample to collection, checking it to be compatible by shape, transform and crs
        """
        if all(obj.same(band) for band in self._samples):
            self._samples.append(obj)
        else:
            raise ValueError('Band is not suitable for collection. '
                             'CRS, transform or shape are different!')
        return

    def sample(self, y, x, height, width):
        samples = [obj.sample(y, x, height, width) for obj in self._samples]
        return BandCollectionSample(samples)

    def reproject(self, dst_crs, interpolation='nearest'):
        """
        Reprojects every BandSample of the collection, see :meth:`BandSample.reproject`
        and returns a new reprojected BandCollectionSample
        """
        reprojected_samples = [s.reproject(dst_crs, interpolation) for s in self._samples]
        return BandCollectionSample(reprojected_samples)

    def reproject_to_utm(self, interpolation='nearest'):
        """
        Alias of `reproject` method with automatic utm zone determining
        """
        dst_crs = get_utm_zone(self.crs, self.transform, (self.height, self.width))
        return self.reproject(dst_crs, interpolation=interpolation)

    def resample(self, dst_res=None, dst_shape=None, interpolation='nearest'):
        """
        Reprojects every BandSample of the collection, see :meth:`BandSample.reproject`
        and returns a new reprojected BandCollectionSample
        """
        resamples = [s.resample(dst_res, dst_shape, interpolation) for s in self._samples]
        return BandCollectionSample(resamples)

    def numpy(self):
        return np.stack([x.numpy() for x in self._samples], 0)

    def ordered(self, *names):
        """
        Creates a new object, containing the specified bands in the specific order.

        Args:
            *names: order of names

        Returns:
            reordered `BandCollectionSample`
        """
        ordered_bands = [self._get_sample(name) for name in names]
        return BandCollectionSample(ordered_bands)

    def save(self, directory, extension='.tif', **kwargs):
        """
        Saves every BandSample in the specified deirectory, see :meth:`BandSample.save`
        """
        os.makedirs(directory, exist_ok=True)
        for s in self._samples:
            s.save(directory, extension, **kwargs)

    def generate_samples(self, height, width):
        """
        A generator for sequential sampling of the BandCollectionSample similar to BandCollection,
        used for the windowed processing of the raster data.

        Args:
            width (int): dimension of sample in pixels and step along `X` axis
            height (int): dimension of sample in pixels and step along `Y` axis

        Yields:
            BandCollectionSample: sequential samples of the specified dimensions
        """
        for x in range(0, self.width, width):
            for y in range(0, self.height, height):
                yield self.sample(y, x, height, width)
