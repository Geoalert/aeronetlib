from logging import getLogger

logger = getLogger("aeronet")

try:
    from aeronet_raster.utils.coords import CRS_LATLON, _utm_zone, get_utm_zone
except ImportError:
    logger.warning("aeronet-raster is not installed! Install as `pip install aeronet[raster]`")
