import rasterio
import numpy as np


def generate_array(width, height, count, dtype, mode='ones'):
    if mode == 'zeros':
        return np.zeros(shape=(count, height, width), dtype=dtype)
    elif mode == 'ones':
        return np.ones(shape=(count, height, width), dtype=dtype)
    elif mode == 'gradient':
        values = np.linspace(0, np.iinfo(dtype).max, count * width * height, dtype=dtype)
        return values.reshape((count, height, width))


def create_tiff_file(filename, width, height, mode='ones', **kwargs):
    profile = {'width': width,
               'height': height,
               'dtype': 'uint8',
               'count': 3,
               'driver': 'GTiff',
               'transform': rasterio.Affine(10.0, 0, 100000, 0, -10.0, 20000),
               'crs': 'EPSG:3857',
               'nodata': 0}
    # we can add\modify any other creation option
    profile.update(**kwargs)
    data = generate_array(width, height, profile['count'], profile['dtype'], mode=mode)

    with rasterio.open(filename, 'w', **profile) as dst:
        dst.write(data)
