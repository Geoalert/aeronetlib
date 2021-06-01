from shapely.geometry import Polygon
from rasterio.features import shapes

from .rasterize import rasterize
from ..raster import BandSample
from ..vector import Feature, FeatureCollection

import numpy as np


class HierFeature(Feature):
    """
    We can operate it just like a Feature, with the collections and all,
    and have the hierarchy information of the child contours

    This is necessary in the case when the polygon is invalid
    and we must deal with its exterior and interiors separately

    Hence, the hierarchical features are one-contour (no holes), and the contour may be marked as a hole itself.
    """

    def __init__(self, geometry, children=None, is_hole=False, **feature_args):

        super().__init__(geometry, **feature_args)

        if children is None:
            self.children = []
        else:
            self.children = children
        self.is_hole = is_hole

        # If the input geometry happen to have the interiors
        # (which may occur on super().__init__ due to _valid() preprocessing)
        # we must take them to separate features
        if self.shape.interiors:
            self._geometry = Polygon(self.shape.exterior)
            for hole in self.shape.interiors:
                new_feat = Feature(Polygon(hole))
                self.children.append(new_feat, is_hole=not self.is_hole)

    def is_parent(self, other: Feature):
        return other.contains(self) and other != self

    def to_plain_feature(self):
        # Returns a feature with all the child features as holes in the parent one
        return Feature(Polygon(shell=self.shape.exterior.coords,
                               holes=[hole.shape.exterior.coords for hole in self.children]),
                       crs=self.crs, properties=self.properties)


def _construct_hierarchy(shells, holes):
    """
    Assignes holes to the corresponding shells
    Args:
        shells: FeatureCollection or list of HierFeatures. Represent outer contours. They become parent HierFeatures.
        holes: FeatureCollection or list of Features. Represent inner contours to be subtracted from the shells.
        They become child contours.
    Returns:
        FeatureCollection of HierFeatures
    """

    for shell in shells:
        candidate_holes = holes.bounds_intersection(shell)
        for hole in candidate_holes:
            if shell.contains(hole):
                shell.children.append(hole)
    return shells


def _to_features(fc):
    """
    Add all holes to the corresponding outer contours, making the hierarchical feature collection where each feature
    has a single contour into a normal list of features consisting of polygons with outer contours and holes,

    Args:
        fc:
    Returns:
         A list of polygons with holes
    """
    features = []
    for parent in fc:
        if not isinstance(parent, Feature):
            raise ValueError('Input fc must be a FeatureCollection')
        elif not isinstance(parent, HierFeature):
            features.append(parent)
        else:  # HierFeature
            if not parent.is_hole:
                features.append(parent.to_plain_feature())

    return FeatureCollection(features, crs=fc.crs)


def vectorize_exact(sample, properties, **kwargs):
    """
    Args:
        sample (BandSample): input image, is treated as binary mask
        properties: key-value pairs to be added to each feature
    Returns:

    """
    binary_image = (sample.numpy()>0)
    all_contours = list(shapes(binary_image.astype('uint8'), transform=sample.transform))

    # First of all, select the contours that evaluate to valid shapely polygons and can be added to output directly
    # This is fast operation and leaves less work for time-consuming procedures later
    # contour[1] is value of the raster, so we select only the shapes that correspond to non-zero pixels
    polys = [Polygon(shell=contour[0]['coordinates'][0],
                      holes=contour[0]['coordinates'][1:]) for contour in all_contours if contour[1] != 0]

    valid_fc = FeatureCollection([Feature(poly,
                                          crs=sample.crs,
                                          properties=properties) for poly in polys if poly.is_valid],
                                 crs=sample.crs)

    # Now we rasterize the valid contours ans subtract them from the initial binary image to get the raster residuals
    # that need separate processing. We cannot just take all invalid polygons because we also need the holes
    # And if we take all the holes from initial contours, there will be holes that are already counted in valid polygons
    # so leaving them behind will speed up the processing
    rasterized_valid = rasterize(valid_fc, sample.transform, binary_image.shape)
    residual = (sample.numpy() != rasterized_valid.numpy())
    residual_contours = list(shapes(residual.astype('uint8'), transform=sample.transform))

    shells = []
    holes = []
    for contour in residual_contours:
        if contour[1] != 0:
            shells.append(Polygon(shell=contour[0]['coordinates'][0]))
        else:
            holes.append(Polygon(shell=contour[0]['coordinates'][0]))

    # We actually need these FCs to use their indexing and bound_intersection function
    # otherwise we could use plain shapely geometries
    shells = FeatureCollection([HierFeature(geometry=Polygon(shell), properties=properties,
                                            crs=sample.crs,
                                            is_hole=False) for shell in shells],
                               crs=sample.crs)

    holes = FeatureCollection([HierFeature(geometry=poly,
                                           crs=sample.crs,
                                           is_hole=True) for poly in holes],
                                  crs=sample.crs)

    validated_contours = _to_features(_construct_hierarchy(shells, holes))

    valid_fc.extend(validated_contours)

    return valid_fc
