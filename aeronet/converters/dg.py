import os
import shapely

from gbdxtools import CatalogImage

from ..dataset import Feature, FeatureCollection
from .split import split as split_raster


GBDX_DIR = '/tmp/gbdx/'


def get_meta(gbdx_image):
    metadata = gbdx_image.metadata['image']

    geometry_wkt = metadata['imageBoundsWGS84']
    geometry = shapely.wkt.loads(geometry_wkt)

    properties = {
        'sat_azimuth': metadata['satAzimuth'],
        'sat_elevation': metadata['satElevation'],
        'sun_azimuth': metadata['sunAzimuth'],
        'sun_elevation': metadata['sunElevation'],
        'off_nadir': metadata['offNadirAngle'],
        'datetime': metadata['acquisitionDate'],
        'gsd': metadata['groundSampleDistanceMeters'],
        'image_id': metadata['imageId'],
        'satellite': metadata['sensorPlatformName'],
    }

    crs = gbdx_image.metadata['georef']['spatialReferenceSystemCode']
    f = Feature(geometry, properties=properties, crs=crs)
    fc = FeatureCollection([f], crs=f.crs)

    return fc


class DGImage:

    def __init__(self, image_id):
        self.image_id = image_id

        self.image = None
        self.image_path = os.path.join(GBDX_DIR, '{}.tif'.format(random_word(10)))
        self.bc = None

    def __getattr__(self, item):
        return getattr(self.image, item)

    def fetch(self, *args, **kwargs):
        self.image = CatalogImage(self.image_id, *args, **kwargs)

    def load(self, *args, **kwargs):
        if self.image is not None:
            self.image.geotiff(self.image_path, *args, **kwargs)
        else:
            raise AttributeError('Fetch image before loading')

    def transform(self, dst_path, channels):
        if os.path.exists(self.image_path):

            # split raster to separate bands
            split_raster(self.image, dst_path, channels)

            # extract meta information and save
            meta_fc = get_meta(self.image)
            meta_fc.save(os.path.join(dst_path, 'meta.geojson'))
            
        else:
            raise FileExistsError('Load GBDX Image before transform')
