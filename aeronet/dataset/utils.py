from logging import getLogger

logger = getLogger("aeronet")

try:
    from aeronet_raster import parse_directory
except ImportError:
    logger.warning("aeronet-raster is not installed! Install as `pip install aeronet[raster]`")
