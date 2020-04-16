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