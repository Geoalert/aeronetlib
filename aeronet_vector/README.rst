Aeronet_vector
~~~~~~~~~~
Python library to work with geospatial vector data

List of content
~~~~~~~~~~~~~~~
- Aim and scope
- Modules
- Quickstart example
- Requirements and installation
- License

**Aim and scope**

As a part of Aeronetlib, which is designed to make it easier for the deep learning researchers to handle
the remote sensing data, Aeronet_vector provides an interface to handle geojson vector data.

It is based on Shapely for geometry object processing, with spatial indexing using
r-tree.

**Modules and classes**
 - .feature
    - `Feature`
 - .featurecollection
    - `FeatureCollection`

Every `Feature` represents a shapely object with Coordinate Reference System (CRS) and a dict of properties.

`FeatureCollection` is a set o `Features` with the same CRS, indexed with R-Tree algorithm,
which allows very fast search for intersecting objects.

Contrary to builtin shapely indexing, aeronet-vector `FeatureCollection` allows index modification
(adding and deleting of geometries), though frequent changes of the index may slow down the execution.

The FeatureCollection has builtin interface to read/write data from/to `GeoJSON` file.

**Quickstart example**

.. code:: python
    from aeronet_vector import Feature, FeatureCollection

    # you can construct Feature either from shapely object, or from geojson-like mapping
    feature = Feature({"type":"Polygon",
                       "coordinates": [[[0.0,0.0],
                                        [0.0,1.0],
                                        [1.0,1.0],
                                        [1.0,0.0],
                                        [0.0,0.0]]})
    # FeatureCollection can be created from an Iterable of Features, or read from file
    fc = FeatureCollection.read("./input.geojson")

    # this is why we need the RTree
    fc = fc.intersection(feature)

    # use
    fc.save("./output.geojson")

**Requirements and installation**

1. python 3.6+
2. rasterio >= 1.0.0
3. shapely >= 1.7.1 < 2.0
4. tqdm

Pypi package:
.. code:: bash

    $ pip install aeronet-vector

**License**

Project is distributed under `MIT License`_.

.. _`requirements.txt`: https://github.com/aeronetlab/aeronetlib/blob/master/requirements.txt
.. _`project wiki`: https://github.com/aeronetlab/aeronetlib/wiki
.. _`MIT License`: https://github.com/aeronetlab/aeronetlib/blob/master/LICENSE

