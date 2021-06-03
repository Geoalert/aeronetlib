from shapely.geometry import shape, Polygon, MultiPolygon, GeometryCollection
from numpy import unique

from ..raster import Band, BandSample, BandCollection, BandCollectionSample
from ._topo_simplify import topo_simplify
from ..vector import Feature, FeatureCollection
from ._vectorize_exact import vectorize_exact
from ._vectorize_opencv import vectorize_opencv, _extract_polygons


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
        geoms = vectorize_opencv(sample.numpy(), transform=sample.transform, **kwargs)
        # remove all the geometries except for polygons
        polys = _extract_polygons(geoms)
        features = ([Feature(geometry, properties=properties, crs=sample.crs)
                     for geometry in polys])
        fc = FeatureCollection(features, crs=sample.crs)
    elif method == 'exact':
        fc = vectorize_exact(sample, properties, **kwargs)
    else:
        raise ValueError('Unknown vectorization method, use `opencv` or `exact`')

    return fc


def polygonize_topo(sample, epsilon=0.1, properties=None, labels=None, label_name='class_id'):
    """
    makes polygonization of a multi-valued mask, where each value represents a separate class of the features
    Args:
        sample: BandSample for vectorization. It may contain multiple values, and each unique value will be vectorized
        as a separate set of features. It is intended for a multi-coloured mask, where each value corresponds to the class
        epsilon:
        properties: common properties for all the features
        labels: dict(value:label) , the properties unique for each value.
        If the labels are specified, only the corresponding values (or layers of Collection) are vectorized.
        If labels is None (default) the labels are generated for each value present in the raster
        label_name: the name (key) of the property that will contain class label. If label_name==False, the labels
        are not written to output features
    """
    if isinstance(sample, (BandSample, Band)):
        # We treat one band as a multi-value mask in which each value means separate class
        raster = sample.numpy()
        values = unique(raster)
        if labels is None:
            rasters = {str(value): BandSample(str(value),
                                              (raster == value).astype('uint8'),
                                              sample.crs,
                                              sample.transform) for value in values}
        else:
            rasters = {label: BandSample(label,
                                         (raster == value).astype('uint8'),
                                         sample.crs,
                                         sample.transform) for label, value in labels.items() if value in values}
        print('band', rasters)
    elif isinstance(sample, (BandCollectionSample, BandCollection)):
        # BandCollection is treated as a list of binary masks
        if labels is None:
            rasters = {str(i): band for i, band in enumerate(sample)}
        else:
            rasters = {label: sample[value] for label, value in labels.items() if value < sample.count}
        print('bc',rasters)
    else:
        raise ValueError(f'sample must be BandSample or BandCollectionSample, got {type(sample)} instead')
    
    if properties is None:
        properties = {}

    fc = FeatureCollection([], crs=sample.crs)
    for label, raster in rasters.items():
        layer_props = properties.copy()
        if label_name:
            # We have an option to disable saving of the layer name to the
            layer_props[label_name] = label
        print(layer_props)
        print(sample.shape)
        fc_ = vectorize_exact(raster, layer_props)
        print('fc_', len(fc_))
        fc.extend(fc_)
        print('fc', len(fc))

    if epsilon > 0:
        print('simplify')
        fc = topo_simplify(fc, epsilon * sample.res[0])

    return fc


