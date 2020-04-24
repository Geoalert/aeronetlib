import random
import string


def band_shape_guard(raster):
    raster = raster.squeeze()
    if raster.ndim != 2:
        raise ValueError('Raster file has wrong shape {}. '.format(raster.shape) +
                         'Should be 2 dim np.array')
    return raster


def random_name(length=10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))