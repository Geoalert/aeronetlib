Dataset exploration and model training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Now we can again open the dataset and create a RandomSampler to genereate the training samples

.. code:: python

    import os
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

