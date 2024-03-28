import logging

from .utils.samplers.gridsampler import GridSampler, make_grid, get_safe_shape
from .utils.utils import to_np_2
from .dataadapters.rasterioadapter import RasterioReader, RasterioWriter
from .dataadapters.abstractadapter import AbstractReader, AbstractWriter
from typing import Sequence, Callable, Optional, Iterable, Union
import numpy as np


def process(src: np.array,
            src_sampler: GridSampler,
            src_sample_size: Sequence[int],
            processor: Callable,
            dst: np.array,
            dst_sampler: GridSampler,
            dst_sample_size: Sequence[int],
            verbose: bool = False):
    """
    Processes array-like data with predictor in windowed mode. Writes to dst inplace
    Args:
        src: source data array-like
        src_sampler: must yield coordinate tuples
        src_sample_size: window size for src
        processor: processing function
        dst: destination, array-like
        dst_sampler: must yield coordinate tuples
        dst_sample_size: window size for dst
        verbose: verbose
    """

    # grid shape must be the same
    assert len(src_sampler.grid) == len(dst_sampler.grid), f'{src_sampler.grid.shape} != {dst_sampler.grid.shape}'

    for src_coords, dst_coords in zip(src_sampler, dst_sampler):
        if verbose:
            logging.info(f'Reading from {src_coords}:{src_coords+src_sample_size} \n'
                         f'Writing into {dst_coords}:{dst_coords+dst_sample_size}')

        sample = src[tuple(slice(src_coords[i],
                                  src_coords[i]+src_sample_size[i],
                                  1) for i in range(len(src_coords)))]
        res = processor(sample)
        dst[tuple(slice(dst_coords[i],
                        dst_coords[i]+dst_sample_size[i],
                        1) for i in range(len(dst_coords)))] = res


def get_auto_cropped_processor(processor: Callable, margin: Sequence[int]) -> Callable:
    """Wraps processor, crops its output by margin"""
    def inner(x):
        return processor(x)[tuple(slice(margin[i], x.shape[i]-margin[i], 1) for i in range(len(margin)))]
    return inner


def process_image(src: AbstractReader,
                  src_sample_size: Union[int, Sequence[int]],
                  src_margin: Union[int, Sequence[int]],
                  processor: Callable,
                  dst: AbstractWriter,
                  dst_sample_size: Union[int, Sequence[int], None] = None,
                  dst_margin: Union[int, Sequence[int], None] = None,
                  verbose: bool = False):
    """
    Args:
        src: reader
        src_sample_size: size of the window including margins (processor input), so stride = sample_size - 2 * margin
        src_margin: size of windows overlap along each axis
        processor: processing function
        dst: writer
        dst_sample_size: size of the window including margins (processor output), so stride = sample_size - 2 * margin
        dst_margin: processor output crop along each axis
        verbose: verbose
    """
    def build_sampler(shape, sample_size, margin):
        stride = sample_size - 2 * margin
        assert np.all(stride > 0)
        safe_shape = get_safe_shape(shape, stride)
        return GridSampler(make_grid([(-margin[i],
                                       safe_shape[i] - margin[i]) for i in range(len(safe_shape))], stride))

    def add_ch_ndim(size, n_ch):
        """Adds extra dim"""
        if isinstance(size, int):
            return np.array((n_ch, size, size))
        elif isinstance(size, (tuple, list, np.ndarray)) and len(size) == 2:
            return np.array((n_ch, size[0], size[1]))
        else:
            raise ValueError(f'Expecting size to be int or Sequence[int, int], got {size}')

    dst_margin = add_ch_ndim(dst_margin or src_margin, 0)
    dst_sample_size = add_ch_ndim(dst_sample_size or src_sample_size, dst.shape[0])

    src_sample_size = add_ch_ndim(src_sample_size, src.shape[0])
    src_margin = add_ch_ndim(src_margin, 0)

    processor = get_auto_cropped_processor(processor, dst_margin)

    src_sampler = build_sampler(src.shape, src_sample_size, src_margin)
    if verbose:
        logging.info(f'Src sampler grid: {src_sampler.grid}')

    dst_sample_size = dst_sample_size-2*dst_margin  # exclude margin from dst sample size since we crop it in processor
    dst_sampler = build_sampler(dst.shape, dst_sample_size, np.array((0, 0, 0)))  # zero margin
    if verbose:
        logging.info(f'Dst sampler grid: {dst_sampler.grid}')

    process(src, src_sampler, src_sample_size, processor,
            dst, dst_sampler, dst_sample_size, verbose)
