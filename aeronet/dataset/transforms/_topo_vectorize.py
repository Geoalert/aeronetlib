from numpy import unique

from ..vector import Feature, FeatureCollection
from ._vectorize import _vectorize, _extract_polygons
from ..raster import BandSample


def topo_polygonize(sample: BandSample, epsilon=0.1, properties=None, labels=None, label_name='class_id'):
    """
    makes polygonization of a multi-valued mask, where each value represents a separate class of the features


    Args:
        sample: BandSample for vectorization. It may contain multiple values, and each unique value will be vectorized
        as a separate set of features. It is intended for a multi-coloured mask, where each value corresponds to the class

        epsilon:
        properties: common properties for all the features
        labels: dict(value:label) , the properties unique for each value.
        If the labels are specified, only the corresponding values are vectorized.
        If labels is None (default) the labels are generated for each value present in the raster
        label_name: the name (key) of the property that will contain class label
    """

    raster = sample.numpy()
    values = unique(raster)

    if properties is None:
        properties = {}
    if labels is None:
        # make proxy labels
        labels = {val: str(val) for val in values}

    features = []
    for value, label in labels.items():
        geoms = _vectorize((raster == value), epsilon=epsilon, transform=sample.transform)
        # remove all the geometries except for polygons
        polys = _extract_polygons(geoms)
        layer_props = properties
        layer_props[label_name] = label
        features += [Feature(geometry, properties=layer_props, crs=sample.crs)
                     for geometry in polys]

    return FeatureCollection(features, crs=sample.crs)

