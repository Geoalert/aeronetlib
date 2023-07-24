Aeronet
~~~~~~~~~~
Python library to work with geospatial data

List of content
~~~~~~~~~~~~~~~
- Aim and scope
- Modules
- Quickstart example
- Requirements and installation
- Documentation and wiki
- Citing
- License

**Aim and scope**

As a part of Aeronetlib, which is designed to make it easier for the deep learning researchers to handle
the remote sensing data, Aeronet_raster provides an interface to handle geotiff raster images.

**Modules and classes**
 - .raster
    - `Band` | `BandCollection`
    - `BandSample` | `BandSampleCollection`
 - .collectionprocessor
    - `CollectionProcessor`
    - `SampleWindowWriter`
    - `SampleCollectionWindowWriter`
 - .visualization
    - `add_mask`

**Quickstart example**

.. code:: python
    from aeronet_raster import BandCollection, split
    from matplotlib import pyplot as plt

    # Split multiband image to single-bands
    IMAGE_FILE = '/path/to/image.tif'
    OUT_PATH = '/path/to/dataset/'
    channels = ['RED', 'GRN', 'BLU']

    bc = split(IMAGE_FILE, OUT_PATH, channels)
    print(f"num_bands is {bc.count}, shape is {bc.shape}")

    # bc.show() returns RGB image as numpy array
    # undersampling specifies resize coefficient
    plt.imshow(bc.show(undersampling=32))

    #############################################################################

    # Load several singleband files
    path_to_folder = 'data/handlabeled'
    channels = ['RED', 'GRN', 'BLU']
    labels = ['101', '901', '902']  # mask channels, each one in separate file
    extension = '.tif'

    paths = [os.path.join(path_to_folder, ch + extension) for ch in channels+labels]
    bc =BandCollection(path)

    # Inspect area at pixel coordinates (xmin = 4000, xmax=5000, ymin=4000, ymax=5000)
    # labels = numbers of bands with masks to show over RGB
    plt.imshow(bc.show((4000, 4000, 5000, 5000), labels = (3,4,5)))

    # Get specified area as numpy array
    # ch_axis - channel axis in resulting array
    sample = bc.numpy((4000, 4000, 5000, 5000), ch_axis=0)
    print(sample.shape) # returns (6, 1000, 1000)

    # Get specified area as BandSampleCollection
    sample = bc.sample(y, x, height, width)

    # Process BandCollection

    from aeronetlib_raster.aeronet_raster import CollectionProcessor

    OUT_CHANNELS = ['RED', 'GRN', 'BLU']

    def processing_fn(sample):
        # Processing function, makes image brighter
        # Here can be segmentation model prediction, etc.
        return np.clip(sample+50, 0, 255).astype(np.uint8)

    # Wrap the function into Predictor
    # `bound` means the width of samples overlap
    predictor = CollectionProcessor(input_channels = (0, 1, 2),
                                    output_labels=OUT_CHANNELS,
                                    processing_fn=processing_fn,
                                    sample_size=(2048,2048),
                                    bound=512)

    # Open the imagery and process it
    result_bc = predictor.process(bc, 'result')
    plt.imshow(result_bc.show(undersampling=32))

**Requirements and installation**

1. python 3
2. rasterio >= 1.0.0
3. shapely >= 1.7.1
4. rtree>=0.8.3,<1.0.0
5. opencv-python>=4.0.0
6. tqdm >=4.36.1

Pypi package:
.. code:: bash

    $ pip install aeronet [all]

for partial install:

Raster-only
.. code:: bash

    $ pip install aeronet [raster]

Vector-only
.. code:: bash

    $ pip install aeronet [vector]

Source code:
.. code:: bash

    $ pip install git+https://github.com/aeronetlab/aeronetlib


**Contributing**
We accept pull-requests and bug reports at github page

You can use ```make build``` to build the libraries and ```make upload``` to update them at pypi (authorization required).


**Documentation and wiki**

The `project wiki`_  contains some insights about the background of the remote sensing data storage
and processing and useful links to the external resources.
Latest **documentation** is available at `Read the docs <https://aeronetlib.readthedocs.io/en/latest/>`__

**Citing**

.. code:: bibtex

    @misc{Yakubovskiy:2019,
      Author = {Pavel Yakubovskiy, Alexey Trekin},
      Title = {Aeronetlib},
      Year = {2019},
      Publisher = {GitHub},
      Journal = {GitHub repository},
      Howpublished = {\url{https://github.com/aeronetlab/aeronetlib}}
    }


**License**

Project is distributed under `MIT License`_.

.. _`requirements.txt`: https://github.com/aeronetlab/aeronetlib/blob/master/requirements.txt
.. _`project wiki`: https://github.com/aeronetlab/aeronetlib/wiki
.. _`MIT License`: https://github.com/aeronetlab/aeronetlib/blob/master/LICENSE

