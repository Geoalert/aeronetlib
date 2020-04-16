Inference
~~~~~~~~~
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
