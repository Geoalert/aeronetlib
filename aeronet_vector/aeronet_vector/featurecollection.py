import json
import rtree
import warnings
from rasterio.crs import CRS
from rasterio.errors import CRSError
from .feature import Feature
from .utils import utm_zone, CRS_LATLON


class FeatureCollection:
    """A set of Features with the same CRS"""

    def __init__(self, features, crs=CRS.from_epsg(4326)):
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
        return FeatureCollection([f.apply(func) for f in self.features], crs=self.crs)

    def filter(self, func):
        return FeatureCollection(filter(func, self.features), crs=self.crs)

    def sort(self, key, reverse=False):
        self.features.sort(key=key, reverse=reverse)

    def extend(self, fc):
        for i, f in enumerate(fc):
            self.index.add(i + len(self), f.bounds)
        self.features.extend(fc.features)

    def append(self, feature):
        self.index.add(len(self), feature.bounds)
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
                message = 'Feature #{} have been removed from collection. Error: {}'.format(i, str(e))
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
            # todo: BUG?? handle non-latlon CRS!
            dst_crs = utm_zone((lat1 + lat2) / 2, (lon1 + lon2) / 2)
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
        """
        return self.reproject(dst_crs='utm')