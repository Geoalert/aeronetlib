Aeronet_convert
~~~~~~~~~~
Python library to convert geospatial vector maps to raster masks and vice versa

List of content
~~~~~~~~~~~~~~~
- Aim and scope
- Modules
- Quickstart example
- Requirements and installation
- License

**Aim and scope**

aeronet-convert is a bridge which connects aeronet-raster and aeronet-vector.
It allows to transform raster masks to vector polygons (`polygonize`) and create raster masks from geometries (`rasterize`)

It is designed for easy use of map information to train the deep learning models, and to get the maps from image processing.

Rasterization relies on `rasterio`.

Vectorization uses `opencv` functions.

**Quickstart example**

Rasterizing map given in geojson format.
We take the georeference from an existing raster file, and the created raster mask has the same georeference, resolution and extent.

.. code:: python
    from aeronet_raster import Band
    from aeronet_vector import FeatureCollection
    from aeronet_convert import rasterize

    # rasterize dataset
    band = Band('red.tif')
    vector_mask = FeatureCollection.read('manual_markup.geojson')
    raster_mask = rasterize(vector_mask)

    raster_mask.save()
    bc = BandCollection(['RED.tif', 'GRN.tif', 'BLU.tif', 'MASK.tif'])


Vectorizing the result of image processing

.. code:: python
    from aeronet_raster import Band
    from aeronet_vector import FeatureCollection
    from aeronet_convert import polygonize, rasterize

    # rasterize dataset
    mask_band = Band('predicted_mask.tif')
    fc = polygonize(mask_band)

    fc.save('result.geojson')

**Requirements and installation**

1. python 3.6+
2. aeronet-raster
3. aeronet-vector
4. opencv>=4.0.0

Pypi package:
.. code:: bash

    $ pip install aeronet-raster

**License**

Project is distributed under `MIT License`_.

.. _`requirements.txt`: https://github.com/aeronetlab/aeronetlib/blob/master/requirements.txt
.. _`project wiki`: https://github.com/aeronetlab/aeronetlib/wiki
.. _`MIT License`: https://github.com/aeronetlab/aeronetlib/blob/master/LICENSE

