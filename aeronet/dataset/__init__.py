from logging import getLogger

logger = getLogger("aeronet")

# todo: rewrite with importlib.util.find_spec to skip imports and warnings when subpackages are not installed

try:
    from aeronet_vector import (Feature, FeatureCollection)
except ImportError:
    logger.warning("aeronet-vector is not installed! Install as `pip install aeronet[vector]`")

try:
    from aeronet_raster import (Band,
                                BandSample,
                                BandCollection,
                                BandCollectionSample,
                                CollectionProcessor as Predictor)
except ImportError:
    logger.warning("aeronet-raster is not installed! Install as `pip install aeronet[raster]`")

try:
    from aeronet_convert import (rasterize, polygonize)
except ImportError:
    logger.warning("aeronet-convert is not installed! Install as `pip install aeronet[convert]`")