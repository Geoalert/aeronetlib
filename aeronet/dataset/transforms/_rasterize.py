from rasterio.features import geometry_mask
from ..raster import BandSample


def rasterize(feature_collection, transform, out_shape, name='mask'):
    """
    Transform vector geometries to raster form, return band sample where
        raster is np.array of bool dtype (`True` value correspond to objects area)
    Args:
        feature_collection: `FeatureCollection` object
        transform: Affine transformation object
            Transformation from pixel coordinates of `source` to the
            coordinate system of the input `shapes`. See the `transform`
            property of dataset objects.
        out_shape: tuple or list
            Shape of output numpy ndarray.
        name: output sample name, default `mask`

    Returns:
        `BandSample` object

    """
    geometries = (f.geometry for f in feature_collection)
    mask = geometry_mask(geometries, out_shape=out_shape, transform=transform, invert=True).astype('uint8')

    return BandSample(name, mask, feature_collection.crs, transform)

