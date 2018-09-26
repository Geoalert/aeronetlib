class GeoObject:
    """Geo spatial object base interface"""
    def __init__(self):
        pass

    @property
    def crs(self):
        """
        Coordinate reference system
        Returns: rasterio.CRS object
        """
        raise NotImplementedError

    @property
    def transform(self):
        """
        Affine transform matrix - (x_res,   0,    x,
                                     0    y_res,  y)
        Returns: rasterio.Affine
        """
        raise NotImplementedError

    @property
    def res(self):
        """
        Resolution or ground sampling distance along X and Y axes.
        Returns:
             tuple (x_resolution, y_resolution)
        """
        raise NotImplementedError

    @property
    def width(self):
        """
        Returns: (int) image width in pixels.
        """
        raise NotImplementedError

    @property
    def height(self):
        """
        Returns: (int) image height in pixels.
        """
        raise NotImplementedError

    @property
    def count(self):
        """
        Returns: (int) number of channels/bands.
        """
        raise NotImplementedError

    @property
    def shape(self):
        """
        Returns: (tuple of int) count, height, width
        """
        raise NotImplementedError

    @property
    def nodata(self):
        """
        Returns: (None or int) nodata value
        """
        raise NotImplementedError

    @property
    def bounds(self):
        raise NotImplementedError

    @property
    def profile(self):
        return {
            'crs': self.crs,
            'nodata': self.nodata,
            'transform':self.transform,
        }

    def sample(self, y, x, height, width):
        raise NotImplementedError

    def reproject(self, dst_crs):
        raise NotImplementedError

    def reproject_to_utm(self):
        raise NotImplementedError

    def resample(self, dst_res):
        raise NotImplementedError