import numpy as np
import os
import rasterio
from rasterio.features import geometry_mask
from .bandcollection import BandCollection
from .band import BandSample


def rasterize(feature_collection, transform, out_shape, name='mask'):
    """Transform vector geometries to raster form, return band sample where
       raster is np.array of bool dtype (`True` value correspond to objects area)

    Args:
        feature_collection: `FeatureCollection` object
        transform: Affine transformation object
            Transformation from pixel coordinates of `source` to the
            coordinate system of the input `shapes`. See the `transform`
            property of dataset objects.
        out_shape: tuple or list
            Shape of output numpy ndarray.
        name: output sample name, default `mask`

    Returns:
        `BandSample` object
    """
    if len(feature_collection) > 0:
        geometries = (f.geometry for f in feature_collection)
        mask = geometry_mask(geometries, out_shape=out_shape, transform=transform, invert=True).astype('uint8')
    else:
        mask = np.zeros(out_shape, dtype='uint8')

    return BandSample(name, mask, feature_collection.crs, transform)


def split(src_fp, dst_fp, channels, exist_ok=True):
    """Split multi-band tiff to separate bands

    This is necessary to prepare the source multi-band data for use with the BandCollection

    Args:
        src_fp: file path to multi-band tiff
        dst_fp: destination path to band collections
        channels: names for bands
        exist_ok:

    Returns:
        BandCollection

    """
    # create directory for new band collection
    os.makedirs(dst_fp, exist_ok=exist_ok)

    # parse extension of bands
    ext = src_fp.split('.')[-1]

    # open existing GeoTiff
    with rasterio.open(src_fp) as src:
        assert len(channels) == src.count
        profile = src.profile
        profile.update({'count': 1})
        dst_pathes = []
        for n in range(src.count):

            dst_band_path = os.path.join(dst_fp, channels[n] + '.{}'.format(ext))
            with rasterio.open(dst_band_path, 'w', **profile) as dst:
                dst.write(src.read(n+1), 1)

            dst_pathes.append(dst_band_path)

    return BandCollection(dst_pathes)
