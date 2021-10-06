import json
import rtree
import warnings
import shapely
import shapely.geometry
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import orient

from rasterio.warp import transform_geom

from ..coords import _utm_zone


CRS_LATLON = 'EPSG:4326'


class Feature:
    """
    Proxy class for shapely geometry, include crs and properties of feature
    """

    def __init__(self, geometry, properties=None, crs=CRS_LATLON):
        self.crs = crs
        self._geometry = self._valid(
            shapely.geometry.shape(geometry))
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
        return shapely.geometry.mapping(self._geometry)
    
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
                # but not oriented polygons can be read by any instrument. So, ni case of any troubles with orientation
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
        utm_zone = _utm_zone((lat1 + lat2)/2 , (lon1 + lon2)/2)
        return self.reproject(utm_zone)


class FeatureCollection:
    """A set of Features with the same CRS"""

    def __init__(self, features, crs=CRS_LATLON):
        self.crs = crs
        self.features = self._valid(features)

        # create indexed set for faster processing
        self.index = rtree.index.Index()
        for i, f in enumerate(self.features):
            self.index.add(i, f.bounds, f.shape)

    def __getitem__(self, item):
        return self.features[item]

    def __len__(self):
        return len(self.features)

    def _valid(self, features):
        valid_features = []
        for f in features:
            if not f.geometry.get('coordinates'):  # remove possible empty shapes
                warnings.warn('Empty geometry detected. This geometry have been removed from collection.',
                              RuntimeWarning)
            else:
                valid_features.append(f)
        return valid_features

    def apply(self, func):
        """ Applies a given function to all the Features of this FeatureColletion

        Args:
            func: A function to be applied to the Features. Must take and return shapely.geometry

        Returns:
            A new FeatureCollection with modified Features
        """
        new_features = [f.apply(func) for f in self.features]
        return FeatureCollection(new_features, crs=self.crs)

    def filter(self, func):
        features = [x for x in self if func(x)]
        return FeatureCollection(features, crs=self.crs)

    def extend(self, fc):
        for i, f in enumerate(fc):
            self.index.add(i + len(self), f.bounds)
        self.features.extend(fc.features)

    def append(self, feature):
        self.index.add(len(self) + 1, feature.bounds)
        self.features.append(feature)

    def bounds_intersection(self, feature):
        idx = self.index.intersection(feature.bounds)
        features = [self.features[i] for i in idx]
        return FeatureCollection(features, self.crs)

    def intersection(self, feature):
        proposed_features = self.bounds_intersection(feature)
        features = []
        for pf in proposed_features:
            if pf.intersection(feature).area > 0:
                features.append(pf)
        return FeatureCollection(features, self.crs)

    @classmethod
    def read(cls, fp):
        with open(fp, 'r', encoding='utf-8') as f:
            collection = json.load(f)
        
        crs = collection.get('crs', CRS_LATLON)
        
        features = []
        for i, feature in enumerate(collection['features']):
            try:
                feature_ = Feature(
                    geometry=feature['geometry'], 
                    properties=feature['properties'],
                    crs=crs,
                )
                features.append(feature_)
            except (KeyError, IndexError) as e:
                message = 'Feature #{} have been removed from collection. Error: {}'.format(i, str(e))
                warnings.warn(message, RuntimeWarning)
                
        return cls(features, crs=crs)

    def save(self, fp, indent=None, hold_crs=False):
        """ Saves feature collection as GeoJSON file
        Args:
            fp (str): filepath
            indent (int): JSON block indent
            hold_crs (bool): make GeoJSON with current projection, that could be not ESPG:4326 (which is standards violation)
        Returns:
            None
        """
        with open(fp, 'w') as f:
            json.dump(self.as_geojson(hold_crs), f, indent=indent)
    
    def as_geojson(self, hold_crs=False):
        """ Returns feature collection as GeoJSON string
        Args:
            hold_crs (bool): make GeoJSON with current projection, that could be not ESPG:4326 (which is standards violation)
        Returns:
            GeoJSON string
        """
        if hold_crs:
            data = {
                'type': 'FeatureCollection',
                'crs': self.crs,
                'features': [f.as_geojson(hold_crs=True) for f in self.features]
            }
        else:
            data = {
                'type': 'FeatureCollection',
                'crs': CRS_LATLON,
                'features': [f.as_geojson() for f in self.features]
            }
        return data

    @property
    def geojson(self):
        return self.as_geojson()

    def reproject(self, dst_crs):
        features = [f.reproject(dst_crs) for f in self.features]
        return FeatureCollection(features, dst_crs)

    def reproject_to_utm(self):
        lon1, lat1, lon2, lat2 = self.index.bounds
        utm_zone = _utm_zone((lat1 + lat2)/2 , (lon1 + lon2)/2)
        features = [f.reproject(utm_zone) for f in self.features]
        return FeatureCollection(features, utm_zone)
