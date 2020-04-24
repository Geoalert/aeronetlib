class GeoObject:
    """Geo spatial object base interface.

    Represents a set of metadata for the georeferenced raster object
    """
    def __init__(self):
        pass

    @property
    def crs(self):
        """ Geographic coordinate reference system of the object Returns a  `rasterio.CRS
        <https://rasterio.readthedocs.io/en/latest/api/rasterio.crs.html#rasterio.crs.CRS>`_

        """
        raise NotImplementedError

    @property
    def transform(self):
        """
        Transform matrix as the `affine.Affine
        <https://github.com/sgillies/affine>`_  object.
        This transform maps pixel row/column coordinates to coordinates in the dataset’s coordinate reference system.
        """
        raise NotImplementedError

    @property
    def res(self):
        """
        Resolution (or ground sampling distance) along X and Y axes in units of the CRS.
        Tuple (x_resolution, y_resolution)
        """
        raise NotImplementedError

    @property
    def width(self):
        """
        Width of the raster data object in pixels
        """
        raise NotImplementedError

    @property
    def height(self):
        """
        Height of the raster data object in pixels
        """
        raise NotImplementedError

    @property
    def count(self):
        """
        (int) number of channels/bands.
        """
        raise NotImplementedError

    @property
    def shape(self):
        """
        Dimensions of the raster in pixels, a tuple of int: (count, height, width); or (height, width)
        """
        raise NotImplementedError

    @property
    def nodata(self):
        """ The value that should be interpreted as \'No data\'. May be None or a value within dtype range"""
        raise NotImplementedError

    @property
    def bounds(self):
        """
        Georeferenced bounds - bounding box in the CRS of the image, based on transform and shape
        """
        raise NotImplementedError

    @property
    def profile(self):
        """ A joint representation of the main properties

        Returns:
            Dict: {
            'crs': crs,
            'nodata': nodata,
            'transform': transform
            }

        """
        return {
            'crs': self.crs,
            'nodata': self.nodata,
            'transform': self.transform,
        }

    def sample(self, y, x, height, width):
        raise NotImplementedError

    def reproject(self, dst_crs):
        raise NotImplementedError

    def reproject_to_utm(self):
        raise NotImplementedError

    def resample(self, dst_res):
        raise NotImplementedError