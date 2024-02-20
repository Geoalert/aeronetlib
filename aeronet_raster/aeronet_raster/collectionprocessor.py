import os
import warnings
import numpy as np
import rasterio
from multiprocessing.pool import ThreadPool
from threading import Lock
from tqdm import tqdm
import cv2
from typing import Union, Optional, Callable, List, Tuple
from .band.band import Band
from .bandcollection.bandcollection import BandCollection
from .utils.calc_window_weight_mtrx import calc_weight_mtrx, recalc_bound_weight_mtrx


class SequentialSampler:

    def __init__(self,
                 band_collection: BandCollection,
                 channels: List[str],
                 sample_size: Union[int, tuple, list],
                 bound: int = 0,
                 padding: str = 'none',
                 nodata: float = 0,
                 nodata_mask_mode: bool = False):
        """ Iterate over BandCollection sequentially with specified shape (+ bounds)
        Args:
            band_collection: BandCollection instance
            channels: list of str, required channels with required order
            sample_size: (height, width), size of `pure` sample in pixels (bounds not included)
            bound: int, bounds in pixels added to sample
            padding: str, padding method
            nodata: float, nodata value
            nodata_mask_mode: bool, nodata mask mode
        Return:
            Iterable object (yield SampleCollection instances)
        """

        self.band_collection = band_collection
        if not isinstance(sample_size, (list, tuple)):
            sample_size = (sample_size, sample_size)
        self.sample_size = sample_size
        self.bound = bound
        self.channels = channels
        self.padding = padding
        self.nodata = nodata
        self.nodata_mask_mode = nodata_mask_mode
        self.blocks = self._compute_blocks()

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, i: int) -> tuple:
        block = self.blocks[i]
        sample = (self.band_collection
                  .ordered(*self.channels)
                  .sample(block['y'], block['x'], block['height'], block['width'])
                  .numpy())

        if np.all(sample == self.nodata):  # all pixels are nodata
            block['non_pad_bounds'] = None
            block['nodata_mask'] = None

            return sample, block

        non_pad_bounds = None
        nodata_mask = None

        if self.nodata_mask_mode:
            nodata_mask = np.all(sample == self.nodata, axis=0)

        if self.padding == 'mirror':
            if sample.shape[0] in [1, 3]:  # only 1 and 3 channels
                sample, non_pad_bounds = self.pad_mirror(sample, nodata_mask)
            else:
                warnings.warn("For `padding` == 'mirror' only 1 and 3 channels are supported. "
                              "Padding will be ignored.",
                              RuntimeWarning)

        block['non_pad_bounds'] = non_pad_bounds
        block['nodata_mask'] = nodata_mask

        return sample, block

    def pad_mirror(self, sample: np.ndarray, nodata_mask: Optional[np.ndarray] = None) -> tuple:
        """
        Pad the given sample array to create a mirrored image.

        Parameters:
            sample :
            nodata_mask :

        Returns:
            tuple: A tuple containing the padded sample array and the non-black bounds as (sample, non_pad_bounds).
        """
        non_pad_bounds = None
        if nodata_mask is None:
            valid_mask = np.logical_not(np.all(sample == self.nodata, axis=0))
        else:
            valid_mask = np.logical_not(nodata_mask)

        y_inds, x_inds = np.nonzero(valid_mask)
        if len(y_inds) >= 2 and len(x_inds) >= 2:
            y_max = max(y_inds)
            y_min = min(y_inds)
            x_max = max(x_inds)
            x_min = min(x_inds)
            non_pad_bounds = (y_min, y_max, x_min, x_max)

            non_pad_sample = sample[:, y_min:y_max + 1, x_min:x_max + 1]

            top = y_min
            left = x_min
            bottom = sample.shape[1] - non_pad_sample.shape[1] - top
            right = sample.shape[2] - non_pad_sample.shape[2] - left

            # CxHxW -> HxWxC
            non_pad_sample = non_pad_sample.transpose(1, 2, 0)
            if non_pad_sample.shape[2] == 1:
                non_pad_sample = non_pad_sample[:, :, 0]
                border_sample = cv2.copyMakeBorder(non_pad_sample, top, bottom, left, right, 4)
                # HxW -> 1xHxW
                sample = np.expand_dims(border_sample, 0)
            elif non_pad_sample.shape[2] == 3:
                border_sample = cv2.copyMakeBorder(non_pad_sample, top, bottom, left, right, 4)
                # HxWx3 -> 3xHxW
                sample = border_sample.transpose(2, 0, 1)

        return sample, non_pad_bounds

    def _compute_blocks(self) -> List[dict]:

        h, w = self.sample_size
        blocks = []
        height = h + 2 * self.bound
        width = w + 2 * self.bound

        for y in range(- self.bound, self.band_collection.height - self.bound, h):
            for x in range(- self.bound, self.band_collection.width - self.bound, w):
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

    def __init__(self, fp: str, shape: tuple, transform, crs, nodata: int, dtype: str = 'uint8',
                 weight_mtrx: Optional[np.ndarray] = None):
        """ Create empty `Band` (rasterio open file) and write blocks sequentially
        Args:
            fp: file path of created Band
            shape: (height, width), size of band in pixels
            transform: rasterio Affine object
            crs: rasterio CRS or epsg core of coordinate system
            nodata: value of pixels without data
            dtype: str, one of rasterio data types
            weight_mtrx: weight matrix
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
        self.weight_mtrx = weight_mtrx
        self.open_mode = 'w'
        if self.weight_mtrx is not None:
            self.open_mode = 'w+'  # read+write
        self.dst = self.open()

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def width(self) -> int:
        return self.shape[1]

    def open(self):
        return rasterio.open(self.fp, self.open_mode, driver='GTiff', transform=self.transform, crs=self.crs,
                             height=self.height, width=self.width, count=1,
                             dtype=self.dtype, nodata=self.nodata)

    def close(self) -> Band:
        self.dst.close()
        return Band(self.fp)

    def write(self, raster: np.ndarray,
              y: int, x: int, height: int, width: int,
              bounds: Optional[Union[list, tuple]] = None,
              non_pad_bounds: Optional[tuple] = None,
              nodata_mask: Optional[np.ndarray] = None):
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
            non_pad_bounds: after 'mirror' padding it is necessary to fill raster with nodata
            to avoid artifacts where there were nodata in the original raster
            nodata_mask: mask of nodata
        Returns:
        """

        if non_pad_bounds is not None:
            y_min, y_max, x_min, x_max = non_pad_bounds
            raster[:y_min].fill(self.nodata)
            raster[y_max + 1:].fill(self.nodata)
            raster[:, :x_min].fill(self.nodata)
            raster[:, x_max + 1:].fill(self.nodata)

        if nodata_mask is not None:
            raster[nodata_mask] = self.nodata

        if bounds:
            if self.weight_mtrx is not None:
                up_bound = bounds[0][0]
                bottom_bound = bounds[0][1]
                left_bound = bounds[1][0]
                right_bound = bounds[1][1]
                sample_size = (self.weight_mtrx.shape[0] - 2 * up_bound,
                               self.weight_mtrx.shape[1] - 2 * left_bound)
                # recalculate weight matrix if window is near to bound
                window_weight_mtrx = recalc_bound_weight_mtrx(y, x, up_bound, left_bound, sample_size, self.weight_mtrx,
                                                              self.height, self.width)
                weighted_raster = raster * window_weight_mtrx

                # cut raster if window is near to bound
                if (y + sample_size[0] + up_bound) >= self.height:
                    weighted_raster = weighted_raster[:-bottom_bound]

                if (x + sample_size[1] + left_bound) >= self.width:
                    weighted_raster = weighted_raster[:, :-right_bound]

                if y < 0:
                    y += up_bound
                    weighted_raster = weighted_raster[up_bound:]

                if x < 0:
                    x += left_bound
                    weighted_raster = weighted_raster[:, left_bound:]

                height = weighted_raster.shape[0]
                width = weighted_raster.shape[1]

                # part of window can be out of raster
                if y + height > self.height:
                    height = self.height - y
                    weighted_raster = weighted_raster[:height]
                if x + width > self.width:
                    width = self.width - x
                    weighted_raster = weighted_raster[:, :width]

                # read current values
                current_raster = self.dst.read(1, window=((y, y + height), (x, x + width)))
                # sum weighted values
                raster = weighted_raster + current_raster
                # round and clip values to avoid in case 'uint8' for example 256 -> 0
                if self.dtype not in ['float32', 'float64']:
                    raster = np.around(raster).clip(0, np.iinfo(self.dtype).max).astype(self.dtype)
            else:
                raster = raster[bounds[0][0]:raster.shape[0] - bounds[0][1],
                                bounds[1][0]:raster.shape[1] - bounds[1][1]]
                x += bounds[1][0]
                y += bounds[0][0]
                width = width - bounds[1][1] - bounds[1][0]
                height = height - bounds[0][1] - bounds[0][0]

        self.dst.write(raster, 1, window=((y, y + height), (x, x + width)))


class SampleCollectionWindowWriter:

    def __init__(self, directory: str, channels: List[str], shape: Tuple[int],
                 transform, crs, nodata: int, dtype: str = 'uint8',
                 weight_mtrx: Optional[np.ndarray] = None):
        """ Create empty `Band` (rasterio open file) and write blocks sequentially
        Args:
            directory: directory path of created BandCollection
            channels: channel names of created BandCollection
            shape: (height, width), size of band in pixels
            transform: rasterio Affine object
            crs: rasterio CRS or epsg core of coordinate system
            nodata: value of pixels without data
            dtype: str, one of rasterio data types
            weight_mtrx: weight matrix
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
        self.weight_mtrx = weight_mtrx
        self.writers = self.open()

    def open(self):
        writers = []
        for fp in self.fps:
            writers.append(
                SampleWindowWriter(fp, self.shape, self.transform,
                                   self.crs, self.nodata, self.dtype, self.weight_mtrx)
            )
        return writers

    def write(self, raster: np.ndarray,
              y: int, x: int, height: int, width: int,
              bounds: Optional[Union[list, tuple]] = None,
              non_pad_bounds: Optional[tuple] = None,
              nodata_mask: Optional[np.ndarray] = None
              ):

        for i in range(len(self.channels)):
            self.writers[i].write(raster[i], y, x, height, width, bounds=bounds, non_pad_bounds=non_pad_bounds,
                                  nodata_mask=nodata_mask)

    def write_empty_block(self,
                          y: int, x: int, height: int, width: int,
                          bounds: Optional[Union[list, tuple]] = None,
                          non_pad_bounds: Optional[tuple] = None,
                          nodata_mask: Optional[np.ndarray] = None):
        empty_raster = np.full(shape=(height, width), fill_value=self.nodata, dtype=self.dtype)
        for i in range(len(self.channels)):
            self.writers[i].write(empty_raster, y, x, height, width, bounds=bounds, non_pad_bounds=None,
                                  nodata_mask=None)

    def close(self) -> BandCollection:
        bands = [w.close() for w in self.writers]
        return BandCollection(bands)


class CollectionProcessor:

    def __init__(self,
                 input_channels: List[str],
                 output_labels: List[str],
                 processing_fn: Callable,
                 sample_size: Tuple[int] = (1024, 1024),
                 bound: int = 256,
                 src_nodata=0,
                 nodata=None, dst_nodata=0,
                 dtype=None, dst_dtype="uint8",
                 n_workers: int = 1,
                 verbose: bool = True,
                 bound_mode: str = 'drop',
                 padding: str = 'none',
                 nodata_mask_mode: bool = False,
                 processing_fn_use_block: bool = False,
                 write_output_to_dst: bool = True,
                ):
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
            n_workers: int, number of workers
            verbose: bool, whether to print progress
            bound_mode: str, 'drop' or 'weight', default 'drop', how to handle boundaries:
                'drop' - drop boundaries, 'weight' - weight boundaries
            padding: str, 'none' or 'mirror', default 'none':
                'none' - no padding, 'mirror' - mirror padding of nodata areas
            nodata_mask_mode: bool, whether to fill by dst_nodata where nodata mask is True
            processing_fn_use_block: bool, whether to pass 'block' argument to processing_fn
            write_output_to_dst: bool, whether to write output of processing_fn to dst 
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

        if bound_mode == 'weight':
            weight_mtrx = calc_weight_mtrx(sample_size, bound)
            if self.dst_dtype not in ['float32', 'float64']:
                warnings.warn("For `bound_mode` == 'weight' `dst_dtype` is recommended to be 'float32' or 'float64'",
                              RuntimeWarning)
        elif bound_mode == 'drop':
            weight_mtrx = None
        else:
            raise ValueError(f"Unknown `bound_mode`: {bound_mode}, should be 'drop' or 'weight'")
        self.weight_mtrx = weight_mtrx

        if padding not in ['none', 'mirror']:
            raise ValueError(f"Unknown `padding`: {padding}, should be 'none' or 'mirror'")
        self.padding = padding

        self.nodata_mask_mode = nodata_mask_mode
        self.processing_fn_use_block = processing_fn_use_block
        self.write_output_to_dst = write_output_to_dst
        self.n_workers = n_workers
        self.verbose = verbose
        self.lock = Lock()

    def _threaded_processing(self, args):
        self._processing(*args)

    def _processing(self, sample: np.ndarray, block: dict, dst: SampleCollectionWindowWriter):
        if np.all(sample == self.src_nodata):
            if self.write_output_to_dst:
                with self.lock:
                    dst.write_empty_block(**block)
        else:
            if self.processing_fn_use_block:
                raster = self.processing_fn(sample, block)
            else:
                raster = self.processing_fn(sample)
            
            if self.write_output_to_dst:
                with self.lock:
                    dst.write(raster, **block)

    def process(self, bc: BandCollection, output_directory: str) -> BandCollection:
        self.src_nodata = self.src_nodata if bc.nodata is None else bc.nodata
        src = SequentialSampler(bc, self.input_channels, self.sample_size, self.bound, self.padding, self.src_nodata,
                                self.nodata_mask_mode)
        dst = SampleCollectionWindowWriter(directory=output_directory,
                                           channels=self.output_labels,
                                           shape=bc.shape[1:],
                                           nodata=self.dst_nodata,
                                           crs=bc.crs,
                                           transform=bc.transform,
                                           dtype=self.dst_dtype,
                                           weight_mtrx=self.weight_mtrx)

        args = ((sample, block, dst) for sample, block in src)
        blocks_num = (bc.shape[1] // self.sample_size[0] + int((bc.shape[1] % self.sample_size[0]) != 0)) * \
                     (bc.shape[2] // self.sample_size[1] + int((bc.shape[2] % self.sample_size[1]) != 0))

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
