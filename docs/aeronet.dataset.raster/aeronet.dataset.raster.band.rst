Band
======================
This module contains the classes to handle bands - that is, single layer raster images, with georeference.

The `Band` class represents an object in the filesystem,
where the raster is not read and only the handle and metadata are stored.
The `BandSample` contains the in-memory raster data that can be operated.

Band
-------

.. autoclass:: aeronet.dataset.raster.Band
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

BandSample
----------
.. autoclass:: aeronet.dataset.raster.BandSample
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
