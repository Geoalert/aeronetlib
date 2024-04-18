import logging
from .utils.samplers.gridsampler import GridSampler, make_grid, get_safe_shape
from .dataadapters.abstractadapter import AbstractAdapter
from .dataadapters.imageadapter import ImageAdapter
from typing import Sequence, Callable, Union, Final, Tuple, Optional
import numpy as np

ArrayLike = Union[np.array, AbstractAdapter]

DST_MARGIN_MODES: Final[Tuple] = ('crop', 'crossfade')


def process(src: ArrayLike,
            src_sampler: GridSampler,
            src_sample_size: Sequence[int],
            processor: Callable,
            dst: ArrayLike,
            dst_sampler: GridSampler,
            dst_sample_size: Sequence[int],
            mode: str = 'crop',
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
        mode: 'crop' or 'crossfade'
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
        if mode == 'crop':
            dst[tuple(slice(dst_coords[i],
                            dst_coords[i]+dst_sample_size[i],
                            1) for i in range(len(dst_coords)))] = res
        elif mode == 'crossfade':
            readen = dst[tuple(slice(dst_coords[i],
                            dst_coords[i] + dst_sample_size[i],
                            1) for i in range(len(dst_coords)))]

            dst[tuple(slice(dst_coords[i],
                            dst_coords[i] + dst_sample_size[i],
                            1) for i in range(len(dst_coords)))] = readen + res

def get_blend_mask(shape: Sequence[int], margin: Sequence[int]) -> np.ndarray:
    """
    Returns alpha-blend float mask with values within [0..1] and linear fades on each side
    shape: mask shape
    margin: margin values for every axis
    Returns: np.ndarray
    """
    if len(shape) != len(margin):
        raise ValueError('len(shape) != len(margin). Margin for every axis is required')
    mask = np.ones(shape)
    for axis in range(len(shape)):
        if margin[axis] == 0:
            continue  # skip this axis
        intersection = margin[axis] * 2  # number of pixels of neighbour windows intersection along given axis
        if intersection*2 > shape[axis]:
            raise ValueError(f'margin must be less than shape//4, got {margin[axis]}, {shape[axis]} along axis={axis}')
        min_v = 1 / (intersection + 1)  # min value in the mask
        linear_mask = np.concatenate((np.linspace(min_v, 1 - min_v, intersection),
                                      np.ones(shape[axis] - 2 * intersection),
                                      np.linspace(1 - min_v, min_v, intersection)))

        mask = np.swapaxes(mask, len(shape)-1, axis)
        mask = mask*linear_mask
        mask = np.swapaxes(mask, len(shape)-1, axis)
    return mask


def get_auto_cropped_processor(processor: Callable, margin: Sequence[int], mode: str = 'crop',
                               blend_mask: Optional[np.ndarray] = None) -> Callable:
    """Wraps processor, crops its output by margin
    processor: function to wrap
    margin: margin values for every axis
    mode: 'crop' - crop every axis by margin, 'crossfade' - apply alpha-blend mask.
    blend_mask: mask to use. Must be same size as the sample. If not specified, will be calculated for each sample
    Returns: Callable
    """
    if mode not in DST_MARGIN_MODES:
        raise ValueError(f'mode must be one of {DST_MARGIN_MODES}')

    def inner(x):
        if mode == 'crop':
            return processor(x)[tuple(slice(margin[i], x.shape[i]-margin[i], 1) for i in range(len(margin)))]
        if mode == 'crossfade':
            output = processor(x)
            if blend_mask is None:
                mask = get_blend_mask(output.shape, margin)
            else:
                mask = blend_mask
            return output*mask
    return inner


def build_sampler(shape: np.ndarray, sample_size: np.ndarray, margin: np.ndarray, mode: str = 'crop'):
    stride = sample_size - 2 * margin
    assert np.all(stride > 0)
    safe_shape = get_safe_shape(shape, stride)  # Equal or bigger shape that is divisible by stride
    if mode == 'crop':
        grid_start = [-margin[i] for i in range(len(shape))]
        grid_end = [safe_shape[i] - margin[i] for i in range(len(shape))]
    elif mode == 'crossfade':  # we need bigger margins here because intersection = margin*2
        grid_start = [-2*margin[i] for i in range(len(shape))]
        grid_end = [safe_shape[i] + margin[i]*2 for i in range(len(shape))]
    else:
        raise ValueError(f'mode must be one of {DST_MARGIN_MODES}')
    return GridSampler(make_grid([(grid_start[i], grid_end[i]) for i in range(len(shape))], stride))


def process_image(src: ImageAdapter,
                  src_sample_size: Union[int, Sequence[int]],
                  src_margin: Union[int, Sequence[int]],
                  processor: Callable,
                  dst: ImageAdapter,
                  dst_sample_size: Union[int, Sequence[int], None] = None,
                  dst_margin: Union[int, Sequence[int], None] = None,
                  mode: str = 'crop',
                  verbose: bool = False):
    """
    Helper function that prepares samplers and mimics the behavior of the old collectionprocessor
    Args:
        src: reader
        src_sample_size: size of the window including margins (processor input), so stride = sample_size - 2 * margin
        src_margin: size of windows overlap along each axis
        processor: processing function
        dst: writer
        dst_sample_size: size of the window including margins (processor output), so stride = sample_size - 2 * margin.
                         If None - same as src_sample_size
        dst_margin: processor output crop along each axis. If None - same as src_margin
        mode: 'crop' - crop every axis by margin, 'crossfade' - apply alpha-blend mask.
        verbose: verbose
    """
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

    src_sampler = build_sampler(src.shape, src_sample_size, src_margin, mode)
    if verbose:
        logging.info(f'Src sampler grid: {src_sampler.grid}')

    if mode == 'crop':
        dst_sample_size = dst_sample_size-2*dst_margin  # exclude margin from dst sample size since we crop it in the processor
        processor = get_auto_cropped_processor(processor, dst_margin, mode)
        dst_margin = np.array((0, 0, 0))  # zero margin
    elif mode == 'crossfade':
        if dst.dtype not in ('float', 'float32', 'float16', np.float64):
            logging.warning(f'For crossfade mode it is recommended to set destination dtype to float, got {dst.dtype}')
        mask = get_blend_mask(dst_sample_size, dst_margin)
        processor = get_auto_cropped_processor(processor, dst_margin, mode, mask)

    dst_sampler = build_sampler(dst.shape, dst_sample_size, dst_margin, mode)
    if verbose:
        logging.info(f'Dst sampler grid: {dst_sampler.grid}')

    process(src, src_sampler, src_sample_size, processor,
            dst, dst_sampler, dst_sample_size, mode, verbose)
