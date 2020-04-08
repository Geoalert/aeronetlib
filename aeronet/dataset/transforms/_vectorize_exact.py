from shapely.geometry import Polygon
from rasterio.features import shapes
from rasterio.transform import IDENTITY

from ..vector import Feature, FeatureCollection
from ._hier_feature import HierFeature


"""
    Here we assume that the input contours are not intersecting apart from the cases when one is fully inside another.
    The contours may be inlied in one another, which means that the inner contours are holes in the current one,
    and the inner contour of a hole is a new contour of a class.

    We will represent it as separate polygons, when every new one within a hole is a new Feature.

    The implications are:
    - if two contours intersect, one of them lies within the other
    - the intersecting contour of the smallest area that is greater than the area of current contour is the direct parent
    - we establish a two-layer hierarchy when the contour is either an outer boundary or an inner hole.
"""


def vectorize_exact(binary_image, min_area=0, transform=IDENTITY):
    """
    Makes an exact vectorization, where the contours match the mask's pixels outer boundaries.

    :param binary_image: numpy array, everything with value > 0 is treated as a positive class
    :param min_area:
    :param transform:
    :return:
    """

    all_contours = shapes((binary_image > 0).astype('uint8'), transform=transform)
    all_contours = _get_contours_hierarchy(all_contours, min_area)
    valid_contours = _remove_outer_holes(all_contours)

    polygons = _get_polygons(valid_contours)
    return polygons


def _get_contours_hierarchy(all_contours, min_area=0):
    """
    Finds the whole hierarchy from a list of contours. Every contour is wrapped into HierFeature
    with the associated parent, children and is_hole flag.
    :param all_contours: contours from rasterio.features.shapes
    :return: FeatureCollection of HierFeatures
    """

    all_contours = FeatureCollection([HierFeature(geometry=Polygon(contour[0]['coordinates'][0]),
                                                  is_hole=(contour[1] == 0))
                                      for contour in all_contours])
    if min_area:
        all_contours = FeatureCollection.filter(lambda x: x.area > min_area)
    for contour in all_contours:
        contour.find_parent(all_contours)

    add_children(all_contours)
    return all_contours


def add_children(fc):
    """
    Parents must be already found
     others:
    :return:
    """
    for contour in fc:
        if contour.parent:
            contour.parent.children.append(contour)


def _remove_outer_holes(fc: FeatureCollection):
    """
    The outer-level contours which are holes should be plainly removed from the collection, we do not need them
    :param fc: FeatureCollection to be cleaned from the holes at the outer hierarchy level. This FC is altered
    :return: new feature collection
    """
    for poly in fc:
        if poly.parent is None and poly.is_hole:
            for ch in poly.children:
                ch.parent = None
    fc = FeatureCollection([feat for feat in fc if not (feat.parent is None and feat.is_hole)])
    return fc


def _get_polygons(fc):
    polygons = []
    used = []
    while len(used) < len(fc):
        tmp_fc = FeatureCollection([feat for feat in fc if feat not in used])
        for feat in fc:
            if not feat.is_hole:
                # Now we get rid of hierarchy information and properties as we add the holes into the geometry
                poly = Polygon(shell=feat.shape.exterior.coords,
                               holes=[c.shape.exterior.coords for c in feat.children])

                polygons.append(mapping(poly))
                # we mark the contours as 'already used for polygon creation'
                # the exterior of the current polygon
                used.append(feat)
                # we don't need the interiors of the polygon anymore, but for the non-hole interiors
                # (there may be such cases, ant they must be added as a sepate polygon)
                used += [f for f in feat.children if f.is_hole]

    return polygons


