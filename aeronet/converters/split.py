import os
import rasterio

from ..dataset import BandCollection


def split(src_fp, dst_fp, channels, exist_ok=True):
    '''
    Split multi-band tiff to separate bands

    Args:
        src_fp: file path to multi-band tiff
        dst_fp: destination path to band collections
        channels: names for bands
        exist_ok:

    Returns:
        BandCollection

    '''

    # create directory for new band collection
    os.makedirs(dst_fp, exist_ok=exist_ok)

    # parse extension of bands
    ext = src_fp.split('.')[-1]

    # open existing GeoTiff
    with rasterio.open(src_fp) as src:

        assert len(channels) == src.count

        profile = src.profile
        profile.update({
            'count': 1,
        })

        # write each band separately
        dst_pathes = []
        for n in range(src.count):

            dst_band_path = os.path.join(dst_fp, channels[n] + '.{}'.format(ext))
            with rasterio.open(dst_band_path, 'w', **profile) as dst:
                dst.write(src.read(n+1), 1)

            dst_pathes.append(dst_band_path)

    return BandCollection(dst_pathes)