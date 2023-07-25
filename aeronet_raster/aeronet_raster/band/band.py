import os
import warnings
import rasterio
import numpy as np
from typing import Optional, Union
from rasterio import Affine
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling

from .bandsample import BandSample
from ..utils.utils import TMP_DIR, random_name
from ..utils.coords import get_utm_zone
from ..geoobject import GeoObject


class Band(GeoObject):
    """Filesystem object `Band` - Rasterio DatasetReader wrapper.

    The `Band` provides access to a georeferenced raster file placed in the filesystem.
    On creation the Band opens the file for reading and
    stores all the necessary metadata and allows to read the raster data on request.

    The majority of properties are inherited from
    rasterio `DatasetReader
    <https://rasterio.readthedocs.io/en/latest/api/rasterio.io.html#rasterio.io.DatasetReader>`_.

    Any file format supported by GDAL drivers can be read.

    Args:
        fp: full path to the raster file

    """
    def __init__(self, fp: str):
        super().__init__()
        self._band = rasterio.open(fp)
        self._tmp_file = False

    def __del__(self):
        fp = self._band.name
        self._band.close()
        if self._tmp_file:
            os.remove(fp)

    # ======================== PROPERTY BLOCK ========================
    @property
    def crs(self):
        """
            :obj:`CRS` - coordindate reference system of the band;  in the file
        """
        return self._band.crs

    @property
    def transform(self):
        """
        Transform matrix as the `affine.Affine
        <https://github.com/sgillies/affine>`_  object.
        This transform maps pixel row/column coordinates to coordinates in the dataset’s coordinate reference system.

        affine.identity is returned if if the file does not contain transform
        """
        return self._band.transform

    @property
    def nodata(self):
        """
        Band nodata value, type depends on the image dtype; None if the nodata value is not specified
        """
        return self._band.nodata

    @property
    def res(self) -> tuple:
        """
        Spatial resolution (x_res, y_res) of the Band in X and Y directions of the georeferenced coordinate system,
        derived from tranaform. Normally is equal to (transform.a, - transform.e)
        """
        return self._band.res

    @property
    def width(self) -> int:
        return self._band.width

    @property
    def height(self) -> int:
        return self._band.height

    @property
    def count(self) -> int:
        """
        By design of the aeronetlib, should be always 1. A Band can be created from image of any channel count,
        but only the first band can be read. If you need to work with multi-channel image, use
        :meth:`aeronet_raster.converters.split.split` to get the one-channel images.

        Returns:
            (int) number of the bands in the image.
        """
        return self._band.count

    @property
    def shape(self) -> tuple:
        """
        The raster dimension as a Tuple (height, width)
        """
        return self.height, self.width,

    @property
    def name(self) -> str:
        """
        Name of the file associated with the Band, without extension and the directory path
        """
        return os.path.basename(self._band.name).split('.')[0]

    @property
    def bounds(self) -> tuple:
        """
        Georeferenced bounds - bounding box in the CRS of the image, based on transform and shape

        Returns:
            `BoundingBox object
            <https://rasterio.readthedocs.io/en/latest/api/rasterio.coords.html#rasterio.coords.BoundingBox>`_:
            (left, bottom, right, top)
        """
        return self._band.bounds

    @property
    def meta(self):
        """
        The basic metadata of the associated rasterio DatasetReader
        """
        return self._band.meta

    @property
    def dtype(self) -> np.dtype:
        """
        Numerical type of the data stored in raster, according to numpy.dtype
        """
        return self._band.dtypes[0]

    @property
    def is_valid(self) -> bool:
        if self._band.count == 1:
            return True
        return False

    # ======================== METHODS BLOCK ========================

    def numpy(self) -> np.ndarray:
        """
        Return numpy representation of the sample
        """
        return self.sample(0, 0, self.height, self.width).numpy()

    def same(self, other: GeoObject) -> bool:
        """Compare if samples have same resolution, crs and shape.

        This means that the samples represent the same territory (like different spectral channels of the same image)
        and can be processed together as collection.

        Args:
            other: GeoObject to compare with

        Returns:
            True if the objects match in shape, crs, transform, False otherwise
        """
        res = True
        res = res and (self.crs == other.crs)
        res = res and (self.transform == other.transform)
        res = res and (self.height == other.height)
        res = res and (self.width == other.width)
        return res

    def _same_extent(self, other: GeoObject) -> bool:
        """
        Compares the spatial extent of the current and other Bands based on their CRSes and transforms.
        The extent is treated as 'same' if the boundaries differ not more than half of the biggest pixel
        Args:
            other: GeoObject to compare the extent to
        Returns:
            bool: True if the rasters are compatible, False otherwise
        """

        # explicitly calculate the other Band's dimensions and resolution in the current crs
        other_bounds = rasterio.warp.transform_bounds(other.crs, self.crs, *other.bounds)
        other_res = [abs(other_bounds[0] - other_bounds[2]) / other.width,
                     abs(other_bounds[1] - other_bounds[3]) / other.height]
        max_pixel = [max(self.res[0], other_res[0]), max(self.res[1], other_res[1])]

        # check every bound to be different not more than half of the bigger pixel
        if abs(other_bounds[0] - self.bounds[0]) > 0.5*max_pixel[0] or \
           abs(other_bounds[1] - self.bounds[1]) > 0.5 * max_pixel[1] or \
           abs(other_bounds[2] - self.bounds[2]) > 0.5 * max_pixel[0] or \
           abs(other_bounds[3] - self.bounds[3]) > 0.5 * max_pixel[1]:
            return False
        else:
            return True

    def sample(self, y: int, x: int, height: int, width: int, **kwargs) -> BandSample:
        """ Read sample of the Band to memory.

        The sample is defined by its size and position in the raster, without respect to the georeference.
        In case if the sample coordinates spread out of the image boundaries, the image is padded with nodata value.

        Args:
            x: pixel horizontal coordinate of left top corner of the sample
            y: pixel vertical coordinate of left top corner of the sample
            width: spatial dimension of sample in pixels
            height: spatial dimension of sample in pixels

        Returns:
             a new :obj:`BandSample` containing the specified spatial subset of the band
        """

        coord_x = self.transform.c + x * self.transform.a
        coord_y = self.transform.f + y * self.transform.e

        dst_crs = self.crs
        dst_name = os.path.basename(self.name)
        dst_nodata = self.nodata if self.nodata is not None else 0
        dst_transform = Affine(self.transform.a, self.transform.b, coord_x,
                               self.transform.d, self.transform.e, coord_y)

        dst_raster = self._band.read(window=((y, y + height), (x, x + width)),
                                     boundless=True, fill_value=dst_nodata)

        sample = BandSample(dst_name, dst_raster, dst_crs, dst_transform, dst_nodata)

        return sample

    def resample(self, dst_res: tuple, fp: Optional[str] = None, interpolation: str = 'nearest') -> GeoObject:
        """ Change spatial resolution of the band. It does not alter the existing file,
        and creates a new file either in the specified location or a temporary file

        It is based on `rasterio.warp.reproject
        <https://rasterio.readthedocs.io/en/latest/api/rasterio.warp.html#rasterio.warp.reproject>`_,
        see for more variants of interpolation.

        Args:
            dst_res (Tuple[float, float]): new resoluton, georeferenced pixel size for the new band
            fp (str): a filename for the new resampled band. If none, a temporary file is created
            interpolation: interpolation type as in rasterio,  `nearest`, `bilinear`, `cubic`, `lanzsos` or others

        Returns:
            a new resampled Band.
        """
        # get temporary filepath if such is not provided
        tmp_file = False if fp is not None else True
        if fp is None:
            fp = '{tmp}/resampled/{directory}/{name}.tif'.format(
                tmp=TMP_DIR, directory=random_name(), name=self.name)

        os.makedirs(os.path.dirname(fp), exist_ok=True)

        transform = Affine(dst_res[0], self.transform.b, self.transform.c,
                           self.transform.d, - dst_res[1], self.transform.f)
        width = round(self.width / (dst_res[0]/self.res[0]))
        height = round(self.height / (dst_res[1]/self.res[1]))

        kwargs = self.meta.copy()
        kwargs.update({
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(fp, 'w', **kwargs) as dst:
            reproject(source=rasterio.band(self._band, 1),
                      destination=rasterio.band(dst, 1),
                      src_transform=self.transform,
                      src_crs=self.crs,
                      dst_transform=transform,
                      dst_crs=self.crs,
                      resampling=getattr(Resampling, interpolation))

        # new band
        band = Band(fp)
        band._tmp_file = tmp_file  # file will be automatically removed when `Band` instance will be deleted

        return band

    def reproject(self, dst_crs: str, dst_res: tuple = None,
                  fp: Optional[str] = None, interpolation: str = 'nearest') -> GeoObject:
        """ Change coordinate system (projection) of the band.
        It does not alter the existing file, and creates a new file either in the specified location or a
         temporary file.

        The band ground sampling distance is not changed, however the resolution may change due to the new coordinate
         system
        It is based on `rasterio.warp.reproject
        <https://rasterio.readthedocs.io/en/latest/api/rasterio.warp.html#rasterio.warp.reproject>`_,
        see for more variants of interpolation.

        Args:
            dst_crs: new CRS, may be in any form acceptable by rasterio, for example as EPSG code, string, CRS object;
             if dst_crs == `utm`, the appropriate UTM zone is used according to the center of the image
            fp (str): a filename for the new resampled band. If none, a temporary file is created
            interpolation: interpolation type as in rasterio,  `nearest`, `bilinear`, `cubic`, `lanzsos` or others
            dst_res (Tuple[float, float]): new resoluton, georeferenced pixel size for the new band

        Returns:
            a new reprojected and resampled Band
        """
        if dst_crs == 'utm':
            dst_crs = get_utm_zone(self.crs, self.transform, (self.height, self.width))
        else:
            dst_crs = dst_crs if isinstance(dst_crs, CRS) else CRS.from_user_input(dst_crs)

        # Old rasterio compatibility: a separate check for validity
        if not dst_crs.is_valid:
            raise rasterio.errors.CRSError('Invalid CRS {} given'.format(dst_crs))

        # get temporary filepath if such is not provided
        tmp_file = False if fp is not None else True
        if fp is None:
            fp = '{tmp}/reprojected_{crs}/{directory}/{name}.tif'.format(
                tmp=TMP_DIR, crs=dst_crs, directory=random_name(), name=self.name)
        os.makedirs(os.path.dirname(fp), exist_ok=True)

        # calculate params of new reprojected Band
        transform, width, height = calculate_default_transform(
            self.crs, dst_crs, self.width, self.height, resolution=dst_res, *self.bounds)
        
        kwargs = self.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        # reproject
        with rasterio.open(fp, 'w', **kwargs) as dst:
            reproject(source=rasterio.band(self._band, 1),
                      destination=rasterio.band(dst, 1),
                      src_transform=self.transform,
                      src_crs=self.crs,
                      dst_transform=transform,
                      dst_crs=dst_crs,
                      resampling=getattr(Resampling, interpolation))

        # new band
        band = Band(fp)
        band._tmp_file = tmp_file  # file will be automatically removed when `Band` instance will be deleted
        return band

    def reproject_to(self, other: GeoObject, fp: Optional[str] = None, interpolation: str = 'nearest') -> GeoObject:
        """
        Reprojects and resamples the band to match exactly the `other`.

        This function ensures that the raster size, crs and transform will be the same,
        allowing them to be merged into one BandCollection. If the intial raster
        exceeds the other in coverage, it will be cut, and if it is insufficient or displaced, it will be zero-padded.

        It aims to overpass the rounding problem which may cause an image to
        be misaligned with itself after a different series of transforms.

        If the images are far from each other, the warning will be shown,
        because the raster may be zero due to severe misalignment.

        Args:
            other(GeoObject): the Band with the parameters to fit to
            fp (str): a filename for the new resampled band. If none, a temporary file is created
            interpolation: interpolation type as in rasterio,  `nearest`, `bilinear`, `cubic`, `lanzsos` or others.
        Returns:
            a new reprojected and resampled Band
        """
        if not self._same_extent(other):
            warnings.warn('You are trying to match two bands that are not even approxiamtely aligned. '
                          'The resulting raster may be empty')

        # get temporary filepath if such is not provided
        tmp_file = False if fp is not None else True
        if fp is None:
            fp = '{tmp}/reprojected_{crs}/{directory}/{name}.tif'.format(
                tmp=TMP_DIR, crs=other.crs, directory=random_name(), name=self.name)
        os.makedirs(os.path.dirname(fp), exist_ok=True)

        kwargs = self.meta.copy()
        kwargs.update({
            'crs': other.crs,
            'transform': other.transform,
            'width': other.width,
            'height': other.height
        })

        # reproject - as in rio.warp --like
        with rasterio.open(fp, 'w', **kwargs) as dst:
            reproject(source=rasterio.band(self._band, 1),
                      destination=rasterio.band(dst, 1),
                      src_transform=self.transform,
                      src_crs=self.crs,
                      dst_transform=other.transform,
                      dst_crs=other.crs,
                      resampling=getattr(Resampling, interpolation))

        # new band
        band = Band(fp)
        band._tmp_file = tmp_file  # file will be automatically removed when `Band` instance will be deleted
        return band

    def reproject_to_utm(self, fp: Optional[str] = None, interpolation: str = 'nearest') -> GeoObject:
        """
        Alias of :obj:`Band.reproject` method with automatic Band utm zone determining
        """
        dst_crs = get_utm_zone(self.crs, self.transform, (self.height, self.width))
        return self.reproject(dst_crs, fp=fp, interpolation=interpolation)

    def generate_samples(self, width: int, height: int) -> BandSample:
        """
        A generator for sequential sampling of the whole band, used for the windowed reading of the raster.
        It allows to handle and process large files without reading them at once in the memory.

        Args:
            width (int): dimension of sample in pixels and step along `X` axis
            height (int): dimension of sample in pixels and step along `Y` axis

        Yields:
            BandSample: sequential samples of the specified dimensions
        """
        for x in range(0, self.width, width):
            for y in range(0, self.height, height):
                yield self.sample(y, x, height, width)



