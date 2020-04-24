from shapely.geometry import Polygon, mapping
from rasterio.features import shapes
from rasterio.transform import IDENTITY

from ..vector import Feature, FeatureCollection
import numpy as np


class HierFeature(Feature):
    """
    We can operate it just like a Feature, with the collections and all,
    and have the hierarchy information of the parent and child contours

    Every contour may contain others or be inside another one, but they should not partially intersect.
    Patial intersections will be ignored when the hierarchy is established, but it can cause problems in the future
    It is either no intersection, or full intersection, otherwise it will cause problems with hierarchy

    We can assume it if these features originate from rasterio.features.shapes data

    Also the hierarchical features are one-contour (no holes), and the contour may be marked as a hole itself.
    """

    def __init__(self, geometry, parent=None, children=None, is_hole=False,
                 properties=None, crs='EPSG:4326'):

        super().__init__(geometry, properties, crs)
        self.parent = parent
        if children is None:
            self.children = []
        else:
            self.children = children
        self.is_hole = is_hole

    @classmethod
    def from_feature(cls, feature, parent=None, children=None, is_hole=None):
        if feature.shape.interiors:
            raise ValueError('Hierarchical feature must not have interior contours (holes)')

        if is_hole is None:
            # The intended use is for hierarchy, so if we create them from features, we need
            # initial information about the contours - either the value of the pixels within,
            # or directly is in a hole or not
            try:
                is_hole = (feature.properties['value'] == 0)
            except KeyError as e:
                print('Input feature must have a `value` property, or is_hole value must be specified')

        return HierFeature(feature.shape,
                           parent,
                           children,
                           is_hole,
                           properties=feature.properties, crs=feature.crs)

    def find_parent(self, others: FeatureCollection):
        """
        Parent is a minimum area contour that contains the current one
        """
        # Check for the bounds_intersection turns out faster than intersection()
        all_parents = others.bounds_intersection(self)
        all_parents = [other for other in all_parents if other.contains(self) and other != self]

        areas = [other.area for other in all_parents]
        if len(areas) == 0:
            self.parent = None
        else:
            argmin_area = np.argmin(areas)
            self.parent = all_parents[argmin_area]


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
    Args:
        binary_image:
        min_area:
        transform:

    Returns:

    """

    all_contours = shapes((binary_image > 0).astype('uint8'), transform=transform)
    all_contours = _get_contours_hierarchy(all_contours, min_area)
    valid_contours = _remove_outer_holes(all_contours)

    polygons = _add_holes(valid_contours)
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
        all_contours = all_contours.filter(lambda x: x.area > min_area)
    for contour in all_contours:
        contour.find_parent(all_contours)

    _add_children(all_contours)
    return all_contours


def _add_children(fc):
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


def _add_holes(fc):
    """
    Add all holes to the corresponding outer contours, making the hierarchical feature collection where each feature
    has a single contour into a normal list of features consisting of polygons with outer contours and holes,


    Args:
        fc:
    Returns:
         A list of polygons with holes
    """
    polygons = []
    used = []
    while len(used) < len(fc):
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


