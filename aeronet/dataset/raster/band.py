import os
import numpy as np

import rasterio
from rasterio import Affine
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling


from .geoobject import GeoObject


class Band(GeoObject):
    """
    Hard drive object `Band` - Rasterio Band wrapper
    """
    def __init__(self, fp):
        """
        Args:
            fp: path to GeoTiff file
        """
        super().__init__()
        self._band = rasterio.open(fp)

    @property
    def crs(self):
        return self._band.crs

    @property
    def transform(self):
        return self._band.transform

    @property
    def nodata(self):
        return self._band.nodata

    @property
    def res(self):
        return self._band.res

    @property
    def width(self):
        return self._band.width

    @property
    def height(self):
        return self._band.height

    @property
    def count(self):
        return self._band.count

    @property
    def shape(self):
        return self._band.shape

    @property
    def name(self):
        """Name of file without extension."""
        return os.path.basename(self._band.name).split('.')[0]

    @property
    def dtype(self):
        """Raster type of data."""
        return self._band.dtypes[0]

    def same(self, other):
        """Compare bands by crs, transform, width, height. If all match return True."""
        res = True
        res = res and (self.crs == other.crs)
        res = res and (self.transform == self.transform)
        res = res and (self.height == self.height)
        res = res and (self.width == self.width)
        return res

    def sample(self, y, x, height, width):
        """
        Read sample of of band to memory with specified:
            x, y - pixel coordinates of left top corner
            width, height - spatial dimension of sample in pixels
        Return: `Sample` object
        """

        coord_x = self.transform.c + x * self.transform.a
        coord_y = self.transform.f + y * self.transform.e

        dst_crs = self.crs
        dst_name = os.path.basename(self.name)
        dst_nodata = self.nodata
        dst_transform = Affine(self.transform.a, self.transform.b, coord_x,
                               self.transform.d, self.transform.e, coord_y)
        dst_raster = self._band.read(window=((y, y + height), (x, x + width)))

        sample = BandSample(dst_name, dst_raster, dst_crs, dst_transform, dst_nodata)

        return sample

    def generate_samples(self, width, height):
        """
        Yield `Sample`s with defined grid
        Args:
            width: dimension of sample in pixels and step along `X` axis
            height: dimension of sample in pixels and step along `Y` axis

        Returns:
            Generator object
        """
        for x in range(0, self.width, width):
            for y in range(0, self.height, height):
                yield self.sample(y, x, height, width)


class BandSample(GeoObject):

    INTERPOLATION = {
        'bilinear': Resampling.bilinear,
        'nearest': Resampling.nearest,
        'cubic': Resampling.cubic,
    }

    EXTENSIONS = ['tif', 'tiff']

    def __init__(self, name, raster, crs, transform, nodata=0):

        super().__init__()

        self._name = name
        self._raster = raster
        self._nodata = nodata
        self._transform = Affine(*transform) if not isinstance(transform, Affine) else transform
        self._crs = CRS(init=crs) if not isinstance(crs, CRS) else crs

    def __eq__(self, other):
        res = np.allclose(self._raster, other.raster)
        res = res and (self.crs.get('init') == other.crs.get('init'))
        res = res and np.allclose(np.array(self.transform), np.array(other.transform))
        return res

    def __repr__(self):
        return f'<BandSample: name={self.name}>, shape={self.shape}, dtype={self.dtype}'

    @property
    def width(self):
        return self._raster.shape[2]

    @property
    def height(self):
        return self._raster.shape[1]

    @property
    def count(self):
        return 1

    @property
    def shape(self):
        return 1, self.height, self.width

    @property
    def dtype(self):
        return self._raster.dtype

    @property
    def res(self):
        return abs(self.transform.a), abs(self.transform.e)

    @property
    def transform(self):
        return self._transform

    @property
    def crs(self):
        return self._crs

    @property
    def nodata(self):
        return self._nodata

    @property
    def name(self):
        return self._name

    @classmethod
    def from_file(cls, fp):
        band = Band(fp)
        return band.sample(0, 0, band.width, band.height)

    @classmethod
    def from_json(cls, data):
        crs = data.get('crs')
        transform = data.get('transform')
        nodata = data.get('nodata')
        raster = decode(data.get('raster'), data.get('shape'), data.get('dtype'))
        name = data.get('name')
        band = cls(name, raster, crs, transform, nodata)
        return band

    def same(self, other):
        """
        Compare if samples have same resolution, crs and shape
        Args:
            other:

        Returns:

        """
        res = True
        res = res and (self.crs == other.crs)
        res = res and (self.transform == self.transform)
        res = res and (self.height == self.height)
        res = res and (self.width == self.width)
        return res

    def save(self, directory, ext='.tif', **kwargs):

        fp = os.path.join(directory, self._name + ext)
        with rasterio.open(fp, mode='w', driver='GTiff', width=self.width, height=self.height,
                           count=1, crs=self.crs.get('init'), transform=self.transform, nodata=self.nodata,
                           dtype=self.dtype, **kwargs) as dst:
            dst.write(self._raster.squeeze(), 1)

    def as_json(self):
        data = dict()
        data['raster'] = encode(self._raster)
        data['shape'] = self.shape
        data['crs'] = self.crs
        data['transform'] = self.transform
        data['nodata'] = self.nodata
        data['name'] = self._name
        data['dtype'] = str(self._raster.dtype)
        return data

    def sample(self, y, x, height, width):
        """
        Subsample of Sample with specified:
            x, y - pixel coordinates of left top corner
            width, height - spatial dimension of sample in pixels
        Return: `Sample` object
        """

        coord_x = self.transform.c + x * self.transform.a
        coord_y = self.transform.f + y * self.transform.e

        dst_transform = Affine(self.transform.a, self.transform.b, coord_x,
                               self.transform.d, self.transform.e, coord_y)
        dst_raster = self._raster[:, y:y+height, x:x+width]

        return BandSample(self.name, dst_raster, self.crs, dst_transform, self.nodata)

    def resample(self, dst_res=None, dst_crs=None, dst_shape=None, interpolation='bilinear'):

        interpolation = self.INTERPOLATION[interpolation]
        crs = self.crs if dst_crs is None else dst_crs
        transform = self.transform if dst_res is None else Affine(dst_res[1],
                                                                  self.transform.b,
                                                                  self.transform.c,
                                                                  self.transform.d,
                                                                  - dst_res[0],
                                                                  self.transform.f, )

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
            dst_crs=crs,
            resampling=interpolation)

        return BandSample(self._name, new_raster, crs, transform, self.nodata)

    # def plot(self, step=10, **kwargs):
    #     plt.imshow(self.raster[0, ::step, ::step], **kwargs)

    def numpy(self):
        return self._raster
