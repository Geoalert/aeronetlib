import cv2
import warnings
import numpy as np
from collections import defaultdict
from rasterio.transform import IDENTITY, xy
from shapely.geometry import shape

from ..vector import Feature, FeatureCollection
from ._vectorize_exact import vectorize_exact
from ._vectorize_opencv import vectorize_opencv
methods = {'opencv': vectorize,
           'exact': vectorize_exact}


def polygonize(sample, method='opencv', properties={}, **kwargs):

    """ Transform the raster mask to vector polygons.
    The pixels in the raster mask are treated as belonging to the object if their value is non-zero, and zero values are background.
    All the objects are transformed to the vector form (polygons).

    The default method is from OpenCV
    `cv2.findContours
    <https://docs.opencv.org/3.1.0/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a>`_

    or alternative based on rasterio.features.shapes

    This method is used as it does process the hierarchy of inlayed contours correctly.
    It also makes polygon simplification, which produces more smooth and lightweight polygons, but they do not match
    the raster mask exactly, which should be taken into account.

    Args:
        sample: BandSample to be vectorized
        method: `opencv` for opencv-based vectorization with approximation. `exact` for rasterio-based method with
            exact correspondence of the polygons to the mask pixel boundaries.
        epsilon: the epsilon parameter for the cv2.approxPolyDP, which specifies the approximation accuracy.
        This is the maximum distance between the original curve and its approximation
        properties: (dict) Properties to be added to the resulting FeatureCollection

    Returns:
        FeatureCollection:
            Polygons in the CRS of the sample, that represent non-black objects in the image
    """
    if method == 'opencv':
        fc = _vectorize_opencv(sample.numpy(), transform=sample.transform, **kwargs)
        # remove all the geometries except for polygons
        polys = _extract_polygons(geoms)
        features = ([Feature(geometry, properties=properties, crs=sample.crs)
                     for geometry in polys])
        fc = FeatureCollection(features, crs=sample.crs)
    elif method == 'exact':
        fc = vectorize_exact(sample, properties, **kwargs)
        if epsilon > 0:
            warnings.warn('Param epsilon is ignored in exact vectorization mode')
    else:
        raise ValueError('Unknown vectorization method, use `opencv` or `exact`')

    return fc


def _extract_polygons(geometries):
    """
    Makes the consistent polygon-only geometry list of valid polygons
    It ignores all other features like linestrings, points etc. that could have been generated during vectorization
    Returns:
        a list of shapely Polygons
    """
    shapes = []
    for geom in geometries:
        if not geom['type'] in ['Polygon', 'MultiPolygon']:
            continue
        else:
            new_shape = shape(geom).buffer(0)
            if isinstance(new_shape, Polygon):
                shapes.append(new_shape)
            elif isinstance(new_shape, MultiPolygon):
                shapes += new_shape
            elif isinstance(new_shape, GeometryCollection):
                for sh in new_shape:
                    if isinstance(sh, Polygon):
                        shapes.append(sh)
                    elif isinstance(sh, MultiPolygon):
                        shapes += sh
    return shapes