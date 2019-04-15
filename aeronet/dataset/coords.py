import math
from rasterio.transform import xy
from rasterio import warp


CRS_LATLON = 'EPSG:4326'


def _utm_zone(lat, lon):
    """
    Calculates UTM zone for latitude and longitude
    :param lat:
    :param lon:
    :return: UTM zone in format 'EPSG:32XYZ'
    """
    zone = (math.floor((lon + 180)/6) % 60) + 1
    str_zone = str(zone).zfill(2)
    
    if lat > 0:
        return 'EPSG:326' + str_zone
    else:
        return 'EPSG:325' + str_zone

def get_utm_zone(crs, transform, shape):
    """
    Calculates the UTM zone for the image center
    :param crs: image crs
    :param transform: image transform
    :param size: image size [height, width]
    :return: UTM zone in format 'EPSG:32XYZ'
    """
    #find image extents
    #todo: check for image size!
    #if it is more than 600 km longitude - recommend not to transform to utm at once
    #find image center
    center_xy = xy(transform, shape[0] / 2, shape[1] / 2)
    center_latlon = warp.transform(crs, CRS_LATLON, [center_xy[0]], [center_xy[1]])
    #calc zone
    return _utm_zone(center_latlon[1][0], center_latlon[0][0])
