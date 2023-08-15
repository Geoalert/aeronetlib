from rasterio.crs import CRS
from aeronet_vector.aeronet_vector.utils import utm_zone


def test_utm_zone_from_latlon():
    assert utm_zone(1.0, 1.0) == CRS.from_epsg(32631)
    assert utm_zone(1.0, 7.0) == CRS.from_epsg(32632)
    assert utm_zone(1.0, 145.0) == CRS.from_epsg(32655)
    assert utm_zone(-1.0, 1.0) == CRS.from_epsg(32731)
    assert utm_zone(1.0, -1.0) == CRS.from_epsg(32630)
    assert utm_zone(-1.0, -1.0) == CRS.from_epsg(32730)


def test_utm_zone_takes_user_input():
    assert utm_zone(1.0, 145.0, "EPSG:4326") == CRS.from_epsg(32655)
    assert utm_zone(10.0, 16100000.0, {"init": "EPSG:3857"}) == CRS.from_epsg(32655)


def test_utm_zone_from_projected():
    assert utm_zone(10.0, 10.0, CRS.from_epsg(3857)) == CRS.from_epsg(32631)
    assert utm_zone(10.0, 1200000.0, CRS.from_epsg(3857)) == CRS.from_epsg(32632)
    assert utm_zone(10.0, 16100000.0, CRS.from_epsg(3857)) == CRS.from_epsg(32655)
    assert utm_zone(-10.0, 10.0, CRS.from_epsg(3857)) == CRS.from_epsg(32731)
    assert utm_zone(10.0, -10.0, CRS.from_epsg(3857)) == CRS.from_epsg(32630)
    assert utm_zone(-10.0, -10.0, CRS.from_epsg(3857)) == CRS.from_epsg(32730)


