import math
from rasterio.crs import CRS


CRS_LATLON = CRS.from_epsg(4326)


def utm_zone(lat: float, lon: float) -> CRS:
    """
    Calculates UTM zone for latitude and longitude
    :param lat:
    :param lon:
    :return: UTM zone in format 'EPSG:32XYZ'
    """
    zone = (math.floor((lon + 180) / 6) % 60) + 1
    str_zone = str(zone).zfill(2)

    if lat > 0:
        return CRS.from_string('EPSG:326' + str_zone)
    else:
        return CRS.from_string('EPSG:325' + str_zone)
