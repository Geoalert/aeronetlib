import os
import rasterio
import numpy as np
from tqdm import tqdm
from typing import List
from rasterio.windows import Window
from rasterio.enums import MaskFlags, ColorInterp
from .bandcollection.bandcollection import BandCollection


def _check_channels_num(src, channels: List[str],
                        dst_channels: int, allow_singleband: bool) -> bool:
    src_channels = src.count
    singleband = allow_singleband and \
                 (src_channels == 1 or (src_channels == 2 and src.colorinterp[1] == ColorInterp.alpha))
    # we can handle real singleband images and with alpha channel

    if src_channels < dst_channels:
        raise ValueError(f"Input raster expected to have {dst_channels} ({channels}) channles,"
                         f" but got {src_channels} channels")

    return singleband


def _create_profile(src_profile: dict) -> dict:
    # Of course, image geometry stays the same
    copy_keys = ['width', 'height', 'transform', 'crs', 'dtype']
    profile = {k: src_profile[k] for k in copy_keys}
    # We make image non-compressed, and tiled for faster windowed read-write
    profile.update({
        'count': 1,
        'blockxsize': 256,
        'blockysize': 256,
        'tiled': True
    })
    return profile


def _get_nodata(src: rasterio.DatasetReader, band: int = 0) -> tuple:
    """
    band: zero-based band num
    returns: (nodataValue, readMask)

    # Nodata is based on the following rules:
    # - If there is NODATA value in th einput image, we use it
    # - Else, if there is internal NODATA bitmask, we apply it to the image (assigning the masked pixels to zero)
    # and make NODATA=0
    # - Else, we look at colorinterp: if there is Alpha channel, we use it as a nodata mask for the image and, again,
    # make NODATA=0
    # - Finally, if none of the options are present, we assume that there is no mask and make NODATA=None

    """
    assert band < src.count
    if MaskFlags.nodata in src.mask_flag_enums[band]:
        # we do not need to read the mask as we know it already is contained inside the band
        return src.nodata, False
    elif MaskFlags.alpha in src.mask_flag_enums[band]:
        alpha = [b for b in range(src.count) if src.colorinterp[b] == ColorInterp.alpha]
        if len(alpha) == 1:
            return 0, True
        return None, False
    elif MaskFlags.per_dataset in src.mask_flag_enums[band]:
        # there should be most probably a bitmask which is not an alpha-channel
        # So, we must read the mask and transform it into nodata mask
        # change nodata mask to internal dataset bitmasks all library-wise
        return 0, True
    else:
        # All_valid
        return None, False


def generate_windows(dataset_height: int, dataset_width: int, window_height: int, window_width: int):
    for y in range(0, dataset_height, window_height):
        for x in range(0, dataset_width, window_width):
            yield (Window(col_off=x, row_off=y,
                          width=min(window_width, dataset_width-x),
                          height=min(window_height, dataset_height-y)))


def split(src_fp: str,
          dst_fp: str,
          channels: List[str],
          exist_ok: bool = True,
          allow_singleband: bool = True,
          window_size: int = 10000):
    """Split multi-band tiff to separate bands
    This is necessary to prepare the source multi-band data for use with the BandCollection
    Args:
        src_fp: file path to multi-band tiff
        dst_fp: destination path to band collections
        channels: names for bands
        exist_ok: set True to not rise error, if output path aleready exists
        allow_singleband: allow to copy singleband image to produce multiband output
        window_size: size of square image window that is read at once, to limit the memory consumption
    Returns:
        BandCollection
    """

    # create directory for new band collection
    os.makedirs(dst_fp, exist_ok=exist_ok)
    # open existing GeoTiff
    with rasterio.open(src_fp) as src:
        dst_channels = len(channels)
        singleband = _check_channels_num(src=src,
                                         dst_channels=dst_channels,
                                         channels=channels,
                                         allow_singleband=allow_singleband)

        profile = _create_profile(src.profile)
        # a doubtful thing: we use the read_mask flag per-dataset, whearas masks may be per-channel
        # however, this is not the case in most situations, and also nodata value is per-dataset
        nodata, read_mask = _get_nodata(src)
        profile.update(nodata=nodata)
        # write each band separately
        dst_pathes = []
        for out_band in tqdm(range(dst_channels)):
            dst_band_path = os.path.join(dst_fp, channels[out_band] + '.tif')
            with rasterio.open(dst_band_path, 'w', **profile) as dst:
                if not window_size:
                    window_size = max(src.height, src.width)
                for window in generate_windows(dataset_height=src.height, dataset_width=src.width,
                                               window_height=window_size, window_width=window_size):
                    if singleband:
                        in_band = 1
                    else:
                        in_band = out_band+1

                    data = src.read(in_band, window=window)
                    if read_mask:
                        mask = src.read_masks(in_band, window=window)
                        data = np.where(mask != 0, data, nodata)
                    dst.write(data, 1, window=window)
            dst_pathes.append(dst_band_path)

    return BandCollection(dst_pathes)
