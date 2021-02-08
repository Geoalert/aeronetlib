Aeronetlib
~~~~~~~~~~
Python library to work with geospatial raster and vector data for deep learning

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

Aeronetlib is designed to make it easier for the deep learning researchers to handle the remote sensing data.

The main functionality of the library:

- handling of the geospatial rasters, including big images;
- random sampling of the image patches from a dataset for the model training;
- sequential sampling of the patches from an image for inference;
- transformation between raster mask and vector polygons in both directions.

**Modules and classes**
 - .raster
    - `Band` | `BandCollection`
    - `BandSample` | `BandSampleCollection`
 - .vector
    - `Feature` | `FeatureCollection`
 - .transforms
    - `polygonize`
    - `rasterize`
 - .io
    - `Predictor`
    - `WindowReader`
    - `SampleWindowWriter`
    - `SampleCollectionWindowWriter`
 - .visualization
    - `add_mask`

**Quickstart example**

Assume that we have a georeferenced RGB image and a map,
and want to create a dataset for image segmentation. We will need to split the image and save each channel separately,
and also to rasterize the map to get a segmentation mask

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

Dataset exploration and training data sampling.
Now we can again open the dataset and create a RandomSampler to genereate the training samples
=======
Aeronet
=======

Python library to work with geospatial raster and vector data.

Modules
~~~~~~~

.backend
^^^^^^^^

.. code:: python

    import matpoltib.pyplpot as plt

    from aeronet.dataset import BandCollection
    from aeronet.dataset import RandomDataset

    from aeronet.dataset.utils import parse_directory
    from aeronet.dataset.visualization import add_mask

    # configuration
    SRC_DIR = '/path/to/elements/'
    channels = ['RED', 'GRN', 'BLU']
    labels = ['100']

    # directories of dataset elements
    dirs = [os.path.join(SRC_DIR, x) for x in os.listdir(SRC_DIR)]
    print('Found collections: ', len(dirs), end='\n\n')

    # parse channels in directories
    band_paths = [parse_direcotry(x, channels + labels) for x in dirs]
    print('BandCollection 0 paths:\n', band_paths[0], end='\n\n')

    # convert to `BandCollection` objects
    band_collections = [BandCollection(fps) for fps in band_paths]
    print('BandCollection 0 object:\n', repr(band_collections[0]))


    # create random dataset sampler
    dataset = RandomDataset(band_collections,
                            sample_size=(512, 512),
                            input_channels=channels,
                            output_labels=labels,
                            transform=None) # pre-processing function

    # get random sample
    generated_sample = dataset[0]
    image = generated_sample['image']
    mask = generated_sample['mask']

    #visualize
    masked_image = add_mask(image, mask)

    plt.figure(figsize=(10,10))
    plt.imshow(masked_image)
    plt.show()

Having a trained model, we can now process the new data.
The main feature here is that the processing is carried out by
sequential sampling of the image patches as we cannot read the whole image at once.
The pathches overlap each other to avoid the boundary effects as possible.

.. code:: python

    from keras.models import load_model
    from aeronet.dataset import Predictor

    # configuration
    INPUT_BC = '/path/to/test/element/'
    channels = ['RED', 'GRN', 'BLU']
    labels = ['100']

    # Load the model. Keras is for example, you can use any
    model = load_model('path/to/model/file.h5', compile=False)

    # Make a prediction function that processes a BandSample
    def processing_fn(sample):
        # Extracting the data from BandSample
        x = sample.numpy().astype(np.float32)

        # Transform the data to fit the model
        x = x.transpose(1,2,0)
        x = np.expand_dims(x, 0)

        # prediction
        y = model.predict(x)

        # Thresholding the output to get a mask
        if threshold is not None:
            y = (y > 0.5).astype(np.uint8)
        return y.squeeze(0).transpose(2,0,1)

    # Wrap the function into Predictor
    # `bound` means the width of samples overlap
    predictor = Predictor(channels,
                      labels,
                      processing_fn=processing_fn,
                      sample_size=(2048,2048),
                      bound=512
                      ))

    # Open the imagery and process it
    bc = BandCollection(parse_direcotry(INPUT_BC, channels))
    bc.process(bc, '/path/to/output/')

    # Make polygons
    vector_data = polygonize(mask2[0], properties={'class': '100'}

**Requirements and installation**

1. python 3
2. rasterio >= 1.0.0
3. shapely >= 1.7a1
4. opencv-python >= 4.0.0
5. rtree
6. tqdm

Pypi package:
.. code:: bash

    $ pip install aeronet

Source code:
.. code:: bash

    $ pip install git+https://github.com/aeronetlab/aeronetlib

You can also use the docker image with current version installed:
.. code:: bash

    $ docker pull aeronetlab/dev:latest

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

