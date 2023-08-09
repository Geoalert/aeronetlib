import os
import warnings
import numpy as np
import rasterio
from multiprocessing.pool import ThreadPool
from threading import Lock
from tqdm import tqdm
from .band.band import Band
from .bandcollection.bandcollection import BandCollection
from typing import Union, Optional, Callable, List, Tuple


class SequentialSampler:

    def __init__(self,
                 band_collection: BandCollection,
                 channels: List[str],
                 sample_size: Union[int, tuple, list],
                 bound: int = 0):
        """ Iterate over BandCollection sequentially with specified shape (+ bounds)
        Args:
            band_collection: BandCollection instance
            channels: list of str, required channels with required order
            sample_size: (height, width), size of `pure` sample in pixels (bounds not included)
            bound: int, bounds in pixels added to sample
        Return:
            Iterable object (yield SampleCollection instances)
        """

        self.band_collection = band_collection
        if not isinstance(sample_size, (list, tuple)):
            sample_size = (sample_size, sample_size)
        self.sample_size = sample_size
        self.bound = bound
        self.channels = channels
        self.blocks = self._compute_blocks()

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, i: int) -> tuple:
        block = self.blocks[i]
        sample = (self.band_collection
                  .ordered(*self.channels)
                  .sample(block['y'], block['x'], block['height'], block['width']))
        return sample, block

    def _compute_blocks(self) -> List[dict]:

        h, w = self.sample_size
        blocks = []
        height = h + 2 * self.bound
        width = w + 2 * self.bound

        for y in range(- self.bound, self.band_collection.height, h):
            for x in range(- self.bound, self.band_collection.width, w):
                rigth_x_bound = max(self.bound,
                                    x + width - self.band_collection.width)
                bottom_y_bound = max(self.bound,
                                     y + height - self.band_collection.height)

                blocks.append({'x': x,
                               'y': y,
                               'height': height,
                               'width': width,
                               'bounds':
                                   [[self.bound, bottom_y_bound], [self.bound, rigth_x_bound]],
                               })
        return blocks


class SampleWindowWriter:

    def __init__(self, fp: str, shape: tuple, transform, crs, nodata: int, dtype: str = 'uint8'):
        """ Create empty `Band` (rasterio open file) and write blocks sequentially
        Args:
            fp: file path of created Band
            shape: (height, width), size of band in pixels
            transform: rasterio Affine object
            crs: rasterio CRS or epsg core of coordinate system
            nodata: value of pixels without data
            dtype: str, one of rasterio data types
        Returns:
            when closed return `Band`
        Examples:
            ```python
            # create band
            bc = BandCollection(['/path/to/RED.tif', '/path/to/GRN.tif'])
            src = SequentialSampler(bc, channels, (1024, 1024), 512)
            dst = SampleWindowWriter('./test.tif', src.shape, **bc.profile)
            for sample, block in src:
                # read raster
                raster = sample.ordered('RED').numpy()
                # transform raster
                raster += 1
                # write raster
                dst.write(raster, **block)
            # close file when all data precessed
            created_band = dst.close()
            ```
        """
        self.fp = fp
        self.shape = shape
        self.transform = transform
        self.nodata = nodata
        self.crs = crs
        self.dtype = dtype
        self.dst = self.open()

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def width(self) -> int:
        return self.shape[1]

    def open(self):
        return rasterio.open(self.fp, 'w', driver='GTiff', transform=self.transform, crs=self.crs,
                             height=self.height, width=self.width, count=1,
                             dtype=self.dtype, nodata=self.nodata)

    def close(self) -> Band:
        self.dst.close()
        return Band(self.fp)

    def write(self, raster: np.ndarray,
              y: int, x: int, height: int, width: int,
              bounds: Optional[Union[list, tuple]] = None):
        """ Writes the specified raster into a window in dst
        The raster boundaries can be cut by 'bounds' pixels to prevent boundary effects on the algorithm output.
        If width and height are not equal to size of raster (after the bounds are cut), which is not typical,
        the raster is stretched to the window size (width and height)
        Args:
            raster: numpy array to be written into dst
            x: begin position of window
            y: begin position of window
            width: size of window
            height: size of window
            bounds: [[,][,]] - number of pixels to cut off from each side of the raster before writing
        Returns:
        """

        if bounds:
            raster = raster[bounds[0][0]:raster.shape[0] - bounds[0][1], bounds[1][0]:raster.shape[1] - bounds[1][1]]
            x += bounds[1][0]
            y += bounds[0][0]
            width = width - bounds[1][1] - bounds[1][0]
            height = height - bounds[0][1] - bounds[0][0]
        self.dst.write(raster, 1, window=((y, y + height), (x, x + width)))


class SampleCollectionWindowWriter:

    def __init__(self, directory: str, channels: List[str], shape: Tuple[int],
                 transform, crs, nodata: int, dtype: str = 'uint8'):
        """ Create empty `Band` (rasterio open file) and write blocks sequentially
        Args:
            directory: directory path of created BandCollection
            channels: channel names of created BandCollection
            shape: (height, width), size of band in pixels
            transform: rasterio Affine object
            crs: rasterio CRS or epsg core of coordinate system
            nodata: value of pixels without data
            dtype: str, one of rasterio data types
        Returns:
            when closed return `BandCollection`
        Examples:
            ```python
            # create band
            bc = BandCollection(['/path/to/RED.tif', '/path/to/GRN.tif'])
            src = SequentialSampler(bc, channels, (1024, 1024), 512)
            dst = SampleCollectionWindowWriter('./test.tif', src.shape, **bc.profile)
            for sample, block in src:
                # read raster
                raster = sample.numpy()
                # transform raster
                raster += 1
                # write raster
                dst.write(raster, **block)
            # close file when all data precessed
            created_bc = dst.close()
            ```
        """
        if directory:
            os.makedirs(directory, exist_ok=True)

        self.fps = [os.path.join(directory, channel + '.tif') for channel in channels]
        self.channels = channels
        self.shape = shape
        self.transform = transform
        self.nodata = nodata
        self.crs = crs
        self.dtype = dtype
        self.writers = self.open()

    def open(self):
        writers = []
        for fp in self.fps:
            writers.append(
                SampleWindowWriter(fp, self.shape, self.transform,
                                   self.crs, self.nodata, self.dtype)
            )
        return writers

    def write(self, raster: np.ndarray,
              y: int, x: int, height: int, width: int,
              bounds: Optional[Union[list, tuple]] = None):
        for i in range(len(self.channels)):
            self.writers[i].write(raster[i], y, x, height, width, bounds=bounds)

    def write_empty_block(self,
                          y: int, x: int, height: int, width: int,
                          bounds: Optional[Union[list, tuple]] = None ):
        empty_raster = np.full(shape=self.shape, fill_value=self.nodata, dtype=self.dtype)
        for i in range(len(self.channels)):
            self.writers[i].write(empty_raster, y, x, height, width, bounds=bounds)

    def close(self) -> BandCollection:
        bands = [w.close() for w in self.writers]
        return BandCollection(bands)


class CollectionProcessor:

    def __init__(self, input_channels: List[str], output_labels: List[str], processing_fn: Callable,
                 sample_size: Tuple[int] = (1024, 1024), bound: int = 256,
                 src_nodata=None,
                 nodata=None, dst_nodata=0,
                 dtype=None, dst_dtype="uint8",
                 n_workers: int = 1, verbose: bool = True):
        """
        Args:
            input_channels: list of str, names of bands/channels
            output_labels: list of str, names of output classes
            processing_fn: callable, function that take as an input `SampleCollection`
                and return raster with shape (output_labels, H, W)
            sample_size: (height, width), size of `pure` sample in pixels (bounds not included)
            bound: int, bounds in pixels added to sample
            src_nodata: value in source bandCollection, that is not processed
                (if all the pixels in sample have this value, the result is filled with dst_nodata)
            dst_nodata: value to fill nodata pixels in resulting mask
            dst_dtype: data type to write to the mask
            nodata: deprecated arg, previously used to pass directly to SampleCollectionWindowWriter.
                Replaced by dst_nodata, preserved for backwards compatibility
            dtype: deprecated arg, previously used to pass directly to SampleCollectionWindowWriter.
                Replaced by dst_dtype, preserved for backwards compatibility
        Returns:
            processed BandCollection
        """

        self.input_channels = input_channels
        self.output_labels = output_labels
        self.processing_fn = processing_fn
        self.sample_size = sample_size
        self.bound = bound
        self.src_nodata = src_nodata
        if nodata is not None:
            warnings.warn("Parameter dtype is deprecated! Use `dst_dtype` instead", DeprecationWarning)
            self.dst_nodata = nodata
        else:
            self.dst_nodata = dst_nodata
        if dtype is not None:
            warnings.warn("Parameter dtype is deprecated! Use `dst_dtype` instead", DeprecationWarning)
            self.dst_dtype = dtype
        else:
            self.dst_dtype = dst_dtype
        self.n_workers = n_workers
        self.verbose = verbose
        self.lock = Lock()

    def _threaded_processing(self, args):
        self._processing(*args)

    def _processing(self, sample: np.ndarray, block: dict, dst: SampleCollectionWindowWriter):
        if np.all(sample == self.src_nodata):
            with self.lock:
                dst.write_empty_block(**block)
        else:
            raster = self.processing_fn(sample)
            with self.lock:
                dst.write(raster, **block)

    def process(self, bc: BandCollection, output_directory: str) -> BandCollection:
        src = SequentialSampler(bc, self.input_channels, self.sample_size, self.bound)
        dst = SampleCollectionWindowWriter(directory=output_directory,
                                           channels=self.output_labels,
                                           shape=bc.shape[1:],
                                           nodata=self.dst_nodata,
                                           crs=bc.crs,
                                           transform=bc.transform,
                                           dtype=self.dst_dtype)

        args = ((sample, block, dst) for sample, block in src)
        blocks_num = ((bc.shape[1] + self.bound) // self.sample_size[0] + 1) * \
                     ((bc.shape[2] + self.bound) // self.sample_size[1] + 1)

        if self.n_workers > 1:
            with ThreadPool(self.n_workers) as p:
                with tqdm(total=blocks_num, disable=(not self.verbose)) as pbar:
                    for _ in p.imap(self._threaded_processing, args):
                        pbar.update()
        else:
            with tqdm(args, total=blocks_num, disable=(not self.verbose)) as data:
                for sample, block, dst in data:
                    self._processing(sample, block, dst)

        return dst.close()
