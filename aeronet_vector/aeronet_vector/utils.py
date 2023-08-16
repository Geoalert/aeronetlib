import math
from rasterio.crs import CRS
from rasterio.warp import transform

CRS_LATLON = CRS.from_epsg(4326)


def utm_zone(y: float,
             x: float,
             crs=CRS_LATLON) -> CRS:
    """
    Calculates UTM zone for latitude and longitude
    :param y: y-coordinate of the point; latitude for lat-lon
    :param x: x-coordinate of the point; longitude for lat-lon
    :param crs: coordinate system in any appropriate for rasterio form
    :return: UTM zone in format CRS.from_epsg('32XYZ')
    """
    crs = CRS.from_user_input(crs)
    if crs == CRS_LATLON:
        lat = y
        lon = x
    else:
        res = transform(src_crs=crs,
                        dst_crs=CRS_LATLON,
                        xs=[x],
                        ys=[y])
        lon = res[0][0]
        lat = res[1][0]

    zone = (math.floor((lon + 180) / 6) % 60) + 1
    str_zone = str(zone).zfill(2)

    if lat > 0:
        return CRS.from_string('EPSG:326' + str_zone)
    else:
        return CRS.from_string('EPSG:327' + str_zone)
