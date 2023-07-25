import warnings
from rasterio.crs import CRS
from rasterio.warp import transform_geom
from shapely.ops import orient
from shapely.geometry import Polygon, shape, mapping
from .utils import utm_zone, CRS_LATLON
import shapely
import numpy as np


class Feature:
    """
    Proxy class for shapely geometry, include crs and properties of feature
    """

    def __init__(self, geometry, properties=None, crs=CRS.from_epsg(4326)):
        self.crs = crs
        self._geometry = self._valid(shape(geometry))
        self.properties = properties

    def __repr__(self):
        print('CRS: {}\nProperties: {}'.format(self.crs, self.properties))
        return repr(self._geometry)

    def __getattr__(self, item):
        return getattr(self._geometry, item)
    
    def __setstate__(self, state):
        self.__dict__ = state

    def __getstate__(self):
        return self.__dict__

    def _valid(self, shape):
        if not shape.is_valid:
            shape = shape.buffer(0)
        return shape

    def apply(self, func):
        return Feature(func(self._geometry), properties=self.properties, crs=self.crs)

    @property
    def shape(self):
        return self._geometry

    @property
    def geometry(self):
        return mapping(self._geometry)

    @property
    def centroid(self):
        return list(self._geometry.centroid.coords)[0]

    @property
    def bbox(self):
        bbox = np.array(tuple(shapely.geometry.box(*self.shape.bounds).exterior.coords)[:-1])
        return np.array(((bbox[:, 0].min(), bbox[:, 0].max()), (bbox[:, 1].max(), bbox[:, 1].min())))

    def squared_distance(self, other):
        self_centroid = self.centroid
        other_centroid = other.centroid
        return (self_centroid[0] - other_centroid[0])**2 + (self_centroid[1] - other_centroid[1])**2

    def IoU(self, other):
        return self._geometry.intersection(other._geometry).area / self._geometry.union(other._geometry).area

    def as_geojson(self, hold_crs=False):
        """ Return Feature as GeoJSON formatted dict
        Args:
            hold_crs (bool): serialize with current projection, that could be not ESPG:4326 (which is standards violation)
        Returns:
            GeoJSON formatted dict
        """
        if self.crs != CRS_LATLON and not hold_crs:
            f = self.reproject(CRS_LATLON)
        else:
            f = self

        shape = f.shape
        if shape.is_empty:
            # Empty geometries are not allowed in FeatureCollections,
            # but here it may occur due to reprojection which can eliminate small geiometries
            # This case is processed separately as orient(POLYGON_EMPTY) raises an exception
            # TODO: do not return anything on empty polygon and ignore such features in FeatureCollection.geojson
            shape = Polygon()
        else:
            try:
                shape = orient(shape)
            except Exception as e:
                # Orientation is really not a crucial step, it follows the geojson standard,
                # but not oriented polygons can be read by any instrument. So, in case of any troubles with orientation
                # we just fall back to not-oriented version of the same geometry
                warnings.warn(f'Polygon orientation failed: {str(e)}. Returning initial shape instead',
                              RuntimeWarning)
                shape = f.shape

        f = Feature(shape, properties=f.properties)
        data = {
            'type': 'Feature',
            'geometry': f.geometry,
            'properties': f.properties
        }
        return data

    @property
    def geojson(self):
        return self.as_geojson()

    def reproject(self, dst_crs):
        new_geometry = transform_geom(
            src_crs=self.crs,
            dst_crs=dst_crs,
            geom=self.geometry,
        )
        return Feature(new_geometry, properties=self.properties, crs=dst_crs)
    
    def reproject_to_utm(self):
        lon1, lat1, lon2, lat2 = self.shape.bounds
        # todo: BUG?? handle non-latlon CRS!
        dst_crs = utm_zone((lat1 + lat2)/2, (lon1 + lon2)/2)
        return self.reproject(dst_crs)
