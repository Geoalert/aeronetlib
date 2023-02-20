import os
import re
import glob
from warnings import warn
import numpy as np
from typing import Union, Optional, Final
import rasterio
from rasterio.features import geometry_mask
from .bandcollection import BandCollection
from .band import BandSample

COLORS: Final[tuple] = ((255, 0, 0),
                        (0, 255, 0),
                        (0, 0, 255),
                        (255, 255, 0),
                        (255, 0, 255),
                        (0, 255, 255))


def _random_color():
    return tuple(np.random.randint(0, 256, 3))


def parse_directory(directory: str, names: tuple[str], extensions: tuple[str] = ('tif', 'tiff', 'TIF', 'TIFF')):
    """
    Extract necessary filenames
    Args:
        directory: str
        names: tuple of str, band or file names, e.g. ['RED', '101']
        extensions: tuple of str, allowable file extensions

    Returns:
        list of matched paths
    """
    paths = glob.glob(os.path.join(directory, '*'))
    extensions = '|'.join(extensions)
    res = []
    for name in names:
        # the channel name must be either full filename (that is, ./RED.tif) or a part after '_' (./dse_channel_RED.tif)
        sep = os.sep if os.sep != '\\' else '\\\\'
        pattern = '.*(^|{}|_)({})\.({})$'.format(sep, name, extensions)
        band_path = [path for path in paths if re.match(pattern, path) is not None]

        # Normally with our datasets it will never be the case, and may indicate wrong file naming
        if len(band_path) > 1:
            warn(RuntimeWarning(
                "There are multiple files matching the channel {}. "
                "It can cause ambiguous behavior later.".format(name)))
        res += band_path

    return res


def add_mask(image: np.ndarray,
             mask: np.ndarray,
             colormap: Optional[list, tuple] = None,
             intensity: float = 0.5):
    """
    Put a mask on the image
    Args:
        image: Image as ndarray (width, height, channels=3),
        mask: Mask as ndarray (width, height) or (width, height, channels),
        colormap: Color for each mask channel, a list of (R, G, B) tuples
        intensity: Mask intensity within [0, 1]:
    """
    assert image.ndim == 3 and image.shape[2] == 3, "Image as ndarray (width, height, channels=3) expected"
    if mask.ndim < 3:
        mask = np.expand_dims(mask, 2)
    assert mask.ndim == 3, "Mask as ndarray (width, height) or (width, height, channels) expected"
    assert image.shape[:2] == mask.shape[:2], 'Shapes mismatch'

    if not colormap:
        colormap = list(COLORS)
    while len(colormap) < mask.shape[2]:
        colormap.append(_random_color())

    rgb_mask = np.zeros((*mask.shape[:2], 3)).astype(np.int16)
    for ch in range(mask.shape[2]):
        rgb_mask += np.stack((mask[:, :, ch]*colormap[ch][0],
                              mask[:, :, ch]*colormap[ch][1],
                              mask[:, :, ch]*colormap[ch][2]), axis=-1)
    image += np.clip(rgb_mask*intensity, 0, 256).astype(np.uint8)
    return np.clip(image, 0, 256)


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
