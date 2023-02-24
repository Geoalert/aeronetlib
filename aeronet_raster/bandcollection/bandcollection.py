import os
import numpy as np
from ..band.band import Band
from ..geoobject.geoobject import GeoObject
from ..utils.coords import get_utm_zone
from ..utils.visualization import add_mask
from .bandcollectionsample import BandCollectionSample
from ..utils.utils import to_tuple, to_np_2, to_np_2_2


class BandCollection(GeoObject):
    """
    A collection of :obj:`Band` s with the same crs, transform and shape.

    It is intended to contain a multi-band image
    and associated raster masks. All the bands must have the same resolution, so if the initial images have
    different resolution, they must be resampled previously.
    Every band must be stored in a separate file; if the initial data is in one file,
    use :meth:`aeronet_raster.converters.split.split` to split it to separate files

    Args:
        bands: list of `Band` or list of file paths
    """

    def __init__(self, bands):

        super().__init__()

        self._bands = [band if isinstance(band, Band) else Band(band) for band in bands]
        if not self.is_valid:
            del self
            raise ValueError('Bands are not suitable for collection. '
                             'CRS, transform or shape are different!')

    def __repr__(self):
        names = [b.name for b in self._bands]
        return '<BandCollection: {}>'.format(names)

    def __getitem__(self, item):
        return self._bands[item]

    def __len__(self):
        return self.count

    # ======================== PROPERTY BLOCK ========================

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
    def bounds(self):
        return self._bands[0].bounds

    @property
    def shape(self):
        return self._bands[0].width, self._bands[0].height

    @property
    def res(self):
        return self._bands[0].res

    @property
    def bands(self):
        return self._bands

    @property
    def is_valid(self):
        """Check if all bands have the same resolution, shape and coordinate system"""
        if len(self._bands) == 0:
            return False
        if len(self._bands) == 1:
            return self._bands[0].is_valid
        return all(self._bands[0].same(other) for other in self._bands[1:]) and all(b.is_valid for b in self._bands)

    # ======================== PRIVATE METHODS ========================

    def _get_band(self, name):
        if isinstance(name, int):
            return self._bands[name]
        for b in self._bands:
            if b.name == name:
                return b
            elif len(b.name) > len(name):  # legacy datasets support
                if b.name.endswith('_{name}'.format(name=name)):
                    return b
            # in all other cases raise error
        raise NameError('No sample with name {name}.'.format(name=name))

    # ======================== PUBLIC METHODS  ========================

    def append(self, other):
        """Add a band to collection, checking it to be compatible by shape, transform and crs"""
        if all(other.same(band) for band in self._bands):
            self._bands.append(other)
        else:
            raise ValueError('Band is not suitable for collection. '
                             'CRS, transform or shape are different!')
        return

    def sample(self, y, x, height, width):
        """
        Sample memory object from BandCollection.

        Args:
            x: int, top left corner `X` offset in pixels
            y: int, top left corner `Y` offset in pixels
            width: int, width of samples in pixels
            height: int, height of samples in pixels

        Returns:
            a new :obj:`BandCollectionSample` containing the specified spatial subset of the BandCollection
        """
        samples = [band.sample(y, x, height, width) for band in self._bands]
        return BandCollectionSample(samples)

    def ordered(self, *names):
        """
        Creates a new object, containing the specified bands in the specific order.

        Args:
            *names: order of names

        Returns:
            reordered `BandCollection`
        """
        ordered_bands = [self._get_band(name) for name in names]
        return BandCollection(ordered_bands)

    def reproject(self, dst_crs, dst_res=None, directory=None, interpolation='nearest'):
        """
        Reprojects every Band of the collection, see :meth:`Band.reproject` and returns a new reprojected BandCollection
        """
        if directory:
            os.makedirs(directory, exist_ok=True)
        r_bands = []
        for band in self:
            fp = os.path.join(directory, band.name + '.tif') if directory else None
            r_band = band.reproject(dst_crs, dst_res=dst_res, fp=fp, interpolation=interpolation)
            r_bands.append(r_band)
        return BandCollection(r_bands)

    def reproject_to_utm(self, dst_res=None, directory=None, interpolation='nearest'):
        """
        Alias of `reproject` method with automatic utm zone determining
        """
        dst_crs = get_utm_zone(self.crs, self.transform, (self.height, self.width))
        return self.reproject(dst_crs, dst_res=dst_res, directory=directory, interpolation=interpolation)

    def resample(self, dst_res, directory=None, interpolation='nearest'):
        """
        Resamples every Band of the collection, see :meth:`Band.resample` and returns a new reprojected BandCollection
        """

        if directory:
            os.makedirs(directory, exist_ok=True)
        r_bands = []
        for band in self:
            fp = os.path.join(directory, band.name + '.tif') if directory else None
            r_band = band.resample(dst_res, fp=fp, interpolation=interpolation)
            r_bands.append(r_band)
        return BandCollection(r_bands)

    def generate_samples(self, height, width):
        """
        A generator for sequential sampling of the whole BandCollection, used for the windowed reading of the raster.
        It allows to handle and process large files without reading them at once in the memory.

        Args:
            width (int): dimension of sample in pixels and step along `X` axis
            height (int): dimension of sample in pixels and step along `Y` axis

        Yields:
            BandCollectionSample: sequential samples of the specified dimensions
        """

        for x in range(0, self.width, width):
            for y in range(0, self.height, height):
                yield self.sample(y, x, height, width)

    def numpy(self, box=None, ch_axis=-1):
        if box is None:
            box = np.array((0, 0), (self.width, self.height))
        if not isinstance(box, np.ndarray):
            box = to_np_2_2(box)
        return np.stack([band.numpy(box) for band in self._bands], axis=ch_axis)

    def show(self, box=None, undersampling=1,
             channels=(0, 1, 2), labels=None, **kwargs):
        if box is None:
            box = self.shape
        box = to_np_2_2(box)
        channels = to_tuple(channels)
        labels = to_tuple(labels)
        img_shape = ((box[1] - box[0]) // undersampling).astype(int)
        assert self.count > max(channels+labels), f"Specified channels to show = {channels+labels} while" \
                                                  f" num_channels={self.count}"

        img = list()
        for ch in channels:
            img.append(self._bands[ch].numpy(box)[::undersampling, ::undersampling])

        while len(img) < 3:
            img.append(np.zeros(img_shape).astype(np.uint8))
        img = np.clip(np.stack(img, axis=-1), 0, 255).astype(np.uint8)

        if labels:
            mask = list()
            for ch in labels:
                mask.append(self._bands[ch].numpy(box)[::undersampling, ::undersampling])
            mask = np.clip(np.stack(mask, axis=-1), 0, 1).astype(np.uint8)
            img = add_mask(img, mask, **kwargs)

        return img
