Data preparation
~~~~~~~~~~~~~~~~
Assume that we have a georeferenced RGB image and a map,
and want to create a dataset for image segmentation. We will use :func:`~aeronet.converters.split.split` function
to save each channel of the image as a separate file, that is in compatible way.
Then we use :func:`~aeronet.dataset.transforms.rasterize` to transform the map to the segmentation mask

.. code:: python

    from aeronet.dataset import BandCollection, FeatureCollection, rasterize
    from aeronet.converters.split import split

    # configuration
    IMAGE_FILE = '/path/to/image.tif'
    MASK_FILE = '/path/to/mask.geojson'
    OUT_PATH = '/path/to/dataset/'

    channels = ['RED', 'GRN', 'BLU']
    label = '100'

    # split the multi-channel image to the separate files for each band
    # it saves the files to the filesystem and returns the BandCollection handle to them
    bc = split(IMAGE_FILE, OUT_PATH, channels)

    # Read the vector data
    fc = FeatureCollection.read(MASK_FILE)

    # The files can have different coordinate systems, so we will need to reproject vector data
    # crs is coordinate system feature for a georeferenced object
    fc = fc.reproject(dst_crs=bc.crs)

    # Now we can rasterize the map data, the mask will be saved in the same folder
    # the result is a BandSample of the size out_shape, with the given georeference and name

    mask_sample = rasterize(feature_collection=fc,
              transform=bc.transform,
              out_shape=bc[0].shape,
              name=label)

    # Now save it, the path is a folder to save, and the filename is derived from the BandSample name
    mask_sample.save(OUT_PATH)
