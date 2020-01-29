class GeoObject:
    """Geo spatial object base interface.

    Represents a set of metadata for the georeferenced raster object
    """
    def __init__(self):
        pass

    @property
    def crs(self):
        """ Geographic coordinate reference system of the object

        Returns:  `rasterio.CRS
        <https://rasterio.readthedocs.io/en/latest/api/rasterio.crs.html#rasterio.crs.CRS>`_
        object
        """
        raise NotImplementedError

    @property
    def transform(self):
        """ Affine transform matrix
        This transform maps pixel row/column coordinates to coordinates in the dataset’s coordinate reference system.

        Returns:
            `affine.Affine
            <https://github.com/sgillies/affine>`_
            object
        """
        raise NotImplementedError

    @property
    def res(self):
        """
        Resolution or ground sampling distance along X and Y axes.

        Returns:
             Tuple [int, int]: (x_resolution, y_resolution)
        """
        raise NotImplementedError

    @property
    def width(self):
        """Width of the raster data object

        Returns:
            (int) image width in pixels.
        """
        raise NotImplementedError

    @property
    def height(self):
        """ Height of the raster data object

        Returns:
            (int) image height in pixels.
        """
        raise NotImplementedError

    @property
    def count(self):
        """
        Returns:
            (int) number of channels/bands.
        """
        raise NotImplementedError

    @property
    def shape(self):
        """
        A tuple of int: (count, height, width); or (height, width)
        """
        raise NotImplementedError

    @property
    def nodata(self):
        """ The value that should be interpreted as \'No data\'. May be None or a value within dtype range"""
        raise NotImplementedError

    @property
    def bounds(self):
        """The lower left and upper right bounds of the dataset in the units of its coordinate reference system.

        Returns:
            Tuple (float, float, float, float): (lower left x, lower left y, upper right x, upper right y)
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