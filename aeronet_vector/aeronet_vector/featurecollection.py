import json
import rtree
import warnings
from rasterio.crs import CRS
from rasterio.errors import CRSError
from .feature import Feature
from .utils import utm_zone, CRS_LATLON
from typing import Callable


class FeatureCollection:
    """A set of Features with the same CRS"""

    def __init__(self, features, crs=CRS.from_epsg(4326)):
        self._crs = crs
        self.features = self._valid(features)

        # create indexed set for faster processing
        self.index = rtree.index.Index()
        for i, f in enumerate(self.features):
            self.index.add(i, f.bounds, f.shape)

    @property
    def crs(self):
        return self._crs

    @crs.setter
    def crs(self, value):
        # Not reprojecting, just setting new value
        self._crs = value
        for f in self.features:
            f.crs = value

    def __getitem__(self, item):
        return self.features[item]

    def __len__(self):
        return len(self.features)

    @staticmethod
    def _valid(features):
        valid_features = []
        for f in features:
            if not f.geometry.get('coordinates'):  # remove possible empty shapes
                warnings.warn('Empty geometry detected. This geometry has been removed from collection.',
                              RuntimeWarning)
            else:
                valid_features.append(f)
        return valid_features

    def apply(self,  func: Callable, inplace: bool = True):
        """Applies function to collection geometries
        Args:
            func (Callable): function to apply
            inplace (bool): if True modifies collection inplace, else returns new collection
        Returns:
            new FeatureCollection if inplace, else None
        """
        if inplace:
            for f in self.features:
                f.apply(func, True)
        else:
            return FeatureCollection([f.apply(func) for f in self.features], crs=self.crs)

    def filter(self, func: Callable, inplace: bool = True):
        """Filters collection according to func
        Args:
            func (Callable): filtering function
            inplace (bool): if True modifies collection inplace, else returns new collection
        Returns:
            new FeatureCollection if inplace, else None
        """
        if inplace:
            self.features = list(filter(func, self.features))
        else:
            return FeatureCollection(filter(func, self.features), crs=self.crs)

    def sort(self, key: Callable, reverse: bool = False):
        """Sorts collection inplace
        Args:
            key (Callable): sorting function
            reverse (bool): if True, ascending sorting order, else descending"""
        self.features.sort(key=key, reverse=reverse)
        self.index = rtree.index.Index()
        for i, f in enumerate(self.features):
            self.index.add(i, f.bounds, f.shape)

    def extend(self, fc):
        """Extends collection with another collection (inplace)
        Args:
            fc (FeatureCollection): collection to extend with"""
        for i, f in enumerate(fc):
            self.index.add(i + len(self), f.bounds)
        self.features.extend(fc.features)

    def append(self, feature):
        """Appends feature to the collection (inplace)
        Args:
            feature (Feature): Feature to append"""
        self.index.add(len(self), feature.bounds)
        self.features.append(feature)

    def bounds_intersection(self, feature):
        """Returns subset of collection features, which bounding boxes intersects with given feature bbox
        Args:
            feature (Feature): Feature to check intersection with
        Returns:
            FeatureCollection"""
        idx = self.index.intersection(feature.bounds)
        features = [self.features[i] for i in idx]
        return FeatureCollection(features, self.crs)

    def intersection(self, feature):
        """Returns subset of collection features, which intersects with given feature
        Args:
            feature (Feature): Feature to check intersection with
        Returns:
            FeatureCollection"""
        proposed_features = self.bounds_intersection(feature)
        features = []
        for pf in proposed_features:
            if pf.intersection(feature).area > 0:
                features.append(pf)
        return FeatureCollection(features, self.crs)

    @staticmethod
    def _process_errors(err_msg, ignore_errors):
        if not ignore_errors:
            raise CRSError(err_msg)

        warning_msg = 'Assuming EPSG:4326 (lat-lon). May cause an error in further reprojection ' \
                      'or rasterization if it is not so.'
        message = f'{err_msg} {warning_msg}'
        warnings.warn(message, RuntimeWarning)

        return CRS_LATLON

    @staticmethod
    def _read_crs(collection, ignore_errors=True):
        if 'crs' not in collection.keys():
            err_msg = 'CRS is not in collection.'
            return FeatureCollection._process_errors(err_msg, ignore_errors)

        crs_raw = collection.get('crs')
        crs = CRS()

        try:
            if isinstance(crs_raw, str):
                crs = CRS.from_user_input(crs_raw)
            elif isinstance(crs_raw, dict):
                if 'type' in crs_raw.keys() and 'properties' in crs_raw.keys():
                    if crs_raw['type'] == 'name':
                        crs = CRS.from_user_input(crs_raw['properties']['name'])
                elif 'init' in crs_raw.keys():
                    crs = CRS.from_user_input(crs_raw['init'])
                else:
                    err_msg = f'CRS can not be interpreted in dict {crs_raw}.'
                    return FeatureCollection._process_errors(err_msg, ignore_errors)
            else:
                err_msg = f'CRS can not be interpreted in {crs_raw}.'
                return FeatureCollection._process_errors(err_msg, ignore_errors)

            # Old rasterio compatibility: a separate check for validity
            if not crs.is_valid:
                err_msg = 'CRS is not valid.'
                return FeatureCollection._process_errors(err_msg, ignore_errors)

            return crs
        # Really invalid CRS will throw CRSError
        except CRSError:
            err_msg = 'CRS was not imported correctly.'
            return FeatureCollection._process_errors(err_msg, ignore_errors)

    @classmethod
    def read(cls, fp):
        r"""Reading the FeatureCollection from a geojson file.
            Args:
                fp: file identifier to open and read the data
            Returns:
                new FeatureCollection with all the polygon data in the file
           """
        with open(fp, 'r', encoding='utf-8') as f:
            collection = json.load(f)

        '''
        We want the CRS to be specified in rasterio-compatible way so that we could reproject the collection
        If it is not specified, it is OK and assumed by the geojson standard that it is CRS_LATLON
        Else it must be read from the file, and if it does not meet any known scheme, we have 2 options:
        - either throw an exception (predictable, but can fail unnecessarily)
        - or ignore the CRS data with a warning (will work better normally, but can give unexpected results)
        '''
        crs = cls._read_crs(collection)
        features = []
        for i, feature in enumerate(collection['features']):
            try:
                feature_ = Feature(
                    geometry=feature['geometry'],
                    properties=feature['properties'],
                    crs=crs,
                )
                features.append(feature_)
            except (KeyError, IndexError, AttributeError) as e:
                message = 'Feature #{} has been removed from collection. Error: {}'.format(i, str(e))
                warnings.warn(message, RuntimeWarning)

        return cls(features, crs=crs)

    def save(self, fp, indent=None, hold_crs=False):
        """ Saves feature collection as GeoJSON file
        Args:
            fp (str): filepath
            indent (int): JSON block indent
            hold_crs (bool): make GeoJSON with current projection, that could be not ESPG:4326
            (which is standards violation)
        Returns:
            None
        """
        with open(fp, 'w') as f:
            json.dump(self.as_geojson(hold_crs), f, indent=indent)

    def as_geojson(self, hold_crs=False):
        """ Returns feature collection as GeoJSON string
        Args:
            hold_crs (bool): make GeoJSON with current projection, that could be not ESPG:4326
            (which is standards violation)
        Returns:
            GeoJSON string
        """
        if hold_crs:
            data = {
                'type': 'FeatureCollection',
                'crs': self.crs.to_dict(),
                'features': [f.as_geojson(hold_crs=True) for f in self.features]
            }
        else:
            data = {
                'type': 'FeatureCollection',
                'crs': CRS_LATLON.to_dict(),
                'features': [f.as_geojson() for f in self.features]
            }
        return data

    @property
    def geojson(self):
        return self.as_geojson()

    def reproject(self, dst_crs):
        """
        Reprojects all the features to the new crs
        Args:
            dst_crs: rasterio.CRS or any acceptable by your rasterio version input (str, dict, epsg code), ot 'utm'

        Returns:
            new reprojected FeatureCollection
        """
        if isinstance(dst_crs, str) and dst_crs == 'utm':
            lon1, lat1, lon2, lat2 = self.index.bounds
            dst_crs = utm_zone((lat1 + lat2) / 2, (lon1 + lon2) / 2, self.crs)
        else:
            dst_crs = dst_crs if isinstance(dst_crs, CRS) else CRS.from_user_input(dst_crs)

        # Old rasterio compatibility: a separate check for validity
        if not dst_crs.is_valid:
            raise CRSError('Invalid CRS {} given'.format(dst_crs))

        features = [f.reproject(dst_crs) for f in self.features]
        return FeatureCollection(features, dst_crs)

    def reproject_to_utm(self):
        """
        Alias of `reproject` method with automatic Band utm zone determining
        The utm zone is determined according to the center of the bounding box of the collection.
        Does not suit to large area geometry, that would not fit into one zone (about 6 dergees in longitude)
        Returns:
            new reprojected FeatureCollection
        """
        return self.reproject(dst_crs='utm')

    def copy(self):
        """Returns a copy of collection"""
        return FeatureCollection((f.copy() for f in self.features), crs=self.crs)

    def simplify(self, tolerance: float, inplace: bool = True):
        """Simplifies geometries with Douglas-Pecker
        Args:
            tolerance (float): simplification tolerance
            inplace (bool): if True modifies Feature inplace, else returns new Feature
        Returns:
            FeatureCollection if inplace, else None"""
        if inplace:
            for f in self.features:
                f.simplify(tolerance, inplace=True)
        else:
            return self.copy().simplify(tolerance, inplace=True)

    def cast_property_to(self, key: str, new_type: type, inplace: bool = True):
        """Casts property to new type (e.g. str to int)
        Args:
            key (str): key of modified property
            new_type (bool): type to cast to
            inplace (bool): if True modifies Feature inplace, else returns new Feature
        Returns:
            FeatureCollection if inplace, else None"""
        if inplace:
            for f in self.features:
                f.cast_property_to(key, new_type, inplace=True)
        else:
            return self.copy().cast_property_to(key, new_type, inplace=True)

    def index_of(self, condition):
        """Returns indexes of features where condition == True
        Args:
            condition (Callable): if condition(feature)==True, index of that feature will be returned
        Raises:
            ValueError when no features found
        Returns:
            (int) index of first occurrence"""
        for i, f in enumerate(self.features):
            if condition(f):
                return i
        raise ValueError('No features found')
