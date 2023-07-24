import os
import numpy as np
import rasterio
from typing import Union
from rasterio import Affine
from rasterio.crs import CRS
from rasterio.coords import BoundingBox
from rasterio.warp import calculate_default_transform, reproject, Resampling

from ..utils.utils import band_shape_guard
from ..utils.coords import get_utm_zone
from ..geoobject import GeoObject


class BandSample(GeoObject):
    """ A wrapper over numpy array representing an in-memory georeferenced raster image.

    It implements all the interfaces of the GeoObject, and stores the raster data in memory

    Args:
        name (str): a name of the sample, which is used as a defaule name for saving to file
        raster (np.array): the raster data
        crs: geographical coordinate reference system, as :obj:`CRS` or string representation
        transform (Affine): affine transform
        nodata: the pixels with this value in raster should be ignored
    """

    def __init__(self, name: str, raster: np.ndarray, crs: Union[str, CRS],
                 transform: Union[dict, Affine], nodata: int = 0):
        """

        """

        super().__init__()

        self._name = name
        self._raster = band_shape_guard(raster)
        self._nodata = nodata
        self._transform = Affine(*transform) if not isinstance(transform, Affine) else transform
        self._crs = CRS.from_user_input(crs) if not isinstance(crs, CRS) else crs

        # Old rasterio compatibility: a separate check for validity
        if not self._crs.is_valid:
            raise rasterio.errors.CRSError('Invalid CRS {} given'.format(crs))

    def __eq__(self, other) -> bool:
        res = np.allclose(self.numpy(), other.numpy())
        res = res and (self.crs == other.crs)
        res = res and np.allclose(np.array(self.transform), np.array(other.transform))
        return res

    def __repr__(self) -> str:
        return '<BandSample: name={}, shape={}, dtype={}>'.format(self.name,
                                                                  self.shape,
                                                                  self.dtype)

    # ======================== PROPERTY BLOCK ========================
    @property
    def width(self) -> int:
        return self._raster.shape[1]

    @property
    def height(self) -> int:
        return self._raster.shape[0]

    @property
    def count(self) -> int:
        return 1

    @property
    def shape(self) -> tuple:
        """
        The raster dimension as a Tuple (height, width)
        """
        return self.height, self.width

    @property
    def dtype(self) -> np.dtype:
        """
        Data type of the associated numpy array
        """
        return self._raster.dtype

    @property
    def res(self) -> tuple:
        return abs(self.transform.a), abs(self.transform.e)

    @property
    def transform(self) -> Affine:
        return self._transform

    @property
    def crs(self) -> CRS:
        return self._crs

    @property
    def nodata(self) -> int:
        return self._nodata

    @property
    def bounds(self) -> BoundingBox:
        """
        Georeferenced bounds - bounding box in the CRS of the image, based on transform and shape

        Returns:
            `BoundingBox object
            <https://rasterio.readthedocs.io/en/latest/api/rasterio.coords.html#rasterio.coords.BoundingBox>`_:
            (left, bottom, right, top)
        """
        left = self.transform.c
        top = self.transform.f
        right = left + self.transform.a * self.width
        bottom = top + self.transform.e * self.height
        return BoundingBox(left, bottom, right, top)

    @property
    def name(self) -> str:
        """
        name of the sample, is used as a base filename when saving to file
        """
        return self._name

    @property
    def is_valid(self) -> bool:
        if self._raster.ndim == 2:
            return True
        return False
    # ======================== METHODS BLOCK ========================

    def same(self, other) -> bool:
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
        res = res and (self.transform == self.transform)
        res = res and (self.height == self.height)
        res = res and (self.width == self.width)
        return res

    def save(self, directory: str, ext: str = '.tif', **kwargs):
        """
        Saves the raster data to a new geotiff file; the filename is derived from this `BandSample` name.
        If file exists, it will be overwritten.

        Args:
            directory: folder to save the file
            ext: file extension; as now only GTiff driver is used, it should match `tif`, `tiff`, `TIF` or `TIFF`.
            kwargs: other keywords arguments to be passed to
             `rasterio.open <https://rasterio.readthedocs.io/en/latest/api/rasterio.html#rasterio.open>`_ .
        """
        fp = os.path.join(directory, self._name + ext)
        with rasterio.open(fp, mode='w', driver='GTiff', width=self.width,
                           height=self.height, count=1, crs=self.crs.to_wkt(),
                           transform=self.transform, nodata=self.nodata,
                           dtype=self.dtype, **kwargs) as dst:
            dst.write(self._raster.squeeze(), 1)

    def sample(self, y: int, x: int, height: int, width: int) -> GeoObject:
        """ Subsample of the Sample with specified dimensions and position within the raster:

        Args:
            x (int): horizontal pixel coordinate of left top corner
            y (int): vertical pixel coordinate of left top corner
            width (int): spatial x-dimension of sample in pixels
            height (int): spatial y-dimension of sample in pixels

        Return:
            a new `BandSample` object
        """

        coord_x = self.transform.c + x * self.transform.a
        coord_y = self.transform.f + y * self.transform.e

        dst_transform = Affine(self.transform.a, self.transform.b, coord_x,
                               self.transform.d, self.transform.e, coord_y)
        dst_raster = self._raster[y:y+height, x:x+width]

        return BandSample(self.name, dst_raster, self.crs, dst_transform, self.nodata)

    def reproject(self, dst_crs: Union[str, CRS], interpolation: str = 'nearest') -> GeoObject:
        """ Change coordinate system (projection) of the band.
        It returns a new BandSample and does not alter the current object

        It is based on `rasterio.warp.reproject
        <https://rasterio.readthedocs.io/en/latest/api/rasterio.warp.html#rasterio.warp.reproject>`_,
        see for more variants of interpolation.

        Args:
            dst_crs: new CRS, may be in any form acceptable by rasterio, for example as EPSG code, string, CRS object;
             if dst_crs == `utm`, the appropriate UTM zone is used according to the center of the image
            interpolation: interpolation type as in rasterio,  `nearest`, `bilinear`, `cubic`, `lanzsos` or others

        Returns:
            BandSample: a new instance with changed CRS.
        """
        if isinstance(dst_crs, str) and dst_crs == 'utm':
            dst_crs = get_utm_zone(self.crs, self.transform, (self.height, self.width))
        else:
            dst_crs = dst_crs if isinstance(dst_crs, CRS) else CRS.from_user_input(dst_crs)

        # Old rasterio compatibility: a separate check for validity
        if not dst_crs.is_valid:
            raise rasterio.errors.CRSError('Invalid CRS {} given'.format(dst_crs))

        dst_transform, dst_width, dst_height = calculate_default_transform(
            self.crs, dst_crs, self.width, self.height, *self.bounds)

        new_raster = np.empty(shape=(1, dst_height, dst_width), dtype=self.dtype)

        reproject(
            self._raster, new_raster,
            src_transform=self.transform,
            dst_transform=dst_transform,
            src_crs=self.crs,
            dst_crs=dst_crs,
            resampling=getattr(Resampling, interpolation))

        return BandSample(self.name, new_raster, dst_crs, dst_transform, self.nodata)

    def reproject_to_utm(self, interpolation: str = 'nearest') -> GeoObject:
        """
        Alias of :obj:`BandSample.reproject` method with automatic Band utm zone determining
        """
        dst_crs = get_utm_zone(self.crs, self.transform, (self.height, self.width))
        return self.reproject(dst_crs, interpolation=interpolation)

    def resample(self, dst_res: tuple = None, dst_shape: tuple = None, interpolation: str = 'nearest') -> GeoObject:
        """ Change spatial resolution of the sample, resizing the raster according to the new resolution.
        dst_res should be specified, otherwise the destination transform will be equal to the source.
        If dst_shape is not specified, it is calculated from dst_res,
        but it can be specified to override it and get the desired output shape

        It is based on `rasterio.warp.reproject
        <https://rasterio.readthedocs.io/en/latest/api/rasterio.warp.html#rasterio.warp.reproject>`_,
        see for more variants of interpolation.

        Args:
            dst_res (Tuple[float, float]): new resoluton, georeferenced pixel size for the new band
            dst_shape: new shape of the resampled raster, can override calculated new shape
            interpolation: interpolation type as in rasterio,  `nearest`, `bilinear`, `cubic`, `lanzsos` or others

        Returns:
            a new resampled BandSample.
        """
        transform = self.transform if dst_res is None else Affine(dst_res[1],
                                                                  self.transform.b,
                                                                  self.transform.c,
                                                                  self.transform.d,
                                                                  - dst_res[0],
                                                                  self.transform.f)

        if dst_res is not None and dst_shape is None:
            target_height = int(self.height * self.res[0] / dst_res[0])
            target_width = int(self.width * self.res[1] / dst_res[1])
        elif dst_shape is not None:
            target_height = dst_shape[1]
            target_width = dst_shape[2]
        else:
            target_height = self.height
            target_width = self.width

        new_raster = np.empty(shape=(1, target_height, target_width), dtype=self.dtype)

        reproject(
            self._raster, new_raster,
            src_transform=self.transform,
            dst_transform=transform,
            src_crs=self.crs,
            dst_crs=self.crs,
            resampling=getattr(Resampling, interpolation))

        return BandSample(self._name, new_raster, self.crs, transform, self.nodata)

    def numpy(self) -> np.ndarray:
        """
        A numpy representation of the raster, without metadata

        Returns: Sample's raster data as a numpy array
        """
        return self._raster

    def generate_samples(self, width: int, height: int) -> GeoObject:
        """
        A generator for sequential sampling of the whole sample, similar to Band,
        used for the windowed processing of the raster data.

        Args:
            width (int): dimension of sample in pixels and step along `X` axis
            height (int): dimension of sample in pixels and step along `Y` axis

        Yields:
            BandSample: sequential samples of the specified dimensions
        """
        for x in range(0, self.width, width):
            for y in range(0, self.height, height):
                yield self.sample(y, x, height, width)