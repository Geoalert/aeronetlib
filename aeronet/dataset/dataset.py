import numpy as np


class RandomDataset:

    def __init__(self, band_collections, sample_size, input_channels, output_labels,
                 transform=None, seed=42):
        """
        Produce samples from band collections randomly (uniform dist.) with specified shape.

        Args:
            band_collections: list of 'BandCollection' objects
            sample_size: spatial resolution of sample in pixels (height, width)
            input_channels: list of channel names, e.g. ['RED', 'GRN', '100'], case sensitive
            output_labels: list of output label names, e.g. ['100', '101', 'roof'], case sensitive
            transform: function for sample transformation
            seed: `NotImplemented`
        """
        self.band_collections = band_collections
        self.sample_size = sample_size
        self.input_channels = input_channels
        self.output_labels = output_labels
        self.transform = transform
        self.seed = seed #TODO: make dataset with seed

    def __getitem__(self, i):
        bc = self.band_collections[i]

        h, w = self.sample_size

        # random sampler
        x_range = max(1, bc.width - w)
        y_range = max(1, bc.height - h)

        x = np.random.randint(x_range)
        y = np.random.randint(y_range)

        # extract samples
        sample = dict()

        sample['image'] = (bc.sample(y, x, h, w)
                           .ordered(*self.input_channels)
                           .numpy()
                           .transpose(1, 2, 0))

        sample['mask'] = (bc.sample(y, x, h, w)
                          .ordered(*self.output_labels)
                          .numpy()
                          .transpose(1, 2, 0))

        # transform samples, e.g. augmentations or standartization
        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    def __len__(self):
        return len(self.band_collections)
