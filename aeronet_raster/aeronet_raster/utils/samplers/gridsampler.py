import numpy as np
from typing import Sequence


def get_safe_shape(shape: Sequence[int], stride: Sequence[int]):
    """
    Returns safe shape that is divisible by stride (Equal or bigger than original shape).
    """
    assert len(shape) == len(stride)
    return [shape[i] if not shape[i]%stride[i] else shape[i]//stride[i]*stride[i]+stride[i] for i in range(len(shape))]


def make_grid(boundaries, stride):
    assert len(boundaries) == len(stride)
    return np.stack([x.reshape(-1) for x in np.meshgrid(*tuple(np.arange(boundaries[i][0],
                                                                         boundaries[i][1],
                                                                         stride[i]) for i in range(len(boundaries))),
                                                        indexing='ij')]).transpose(1, 0)


class GridSampler:
    """
    yields from grid

    Args:
        grid: array;
        verbose: print debug
    """
    __slots__ = ('_grid', 'verbose')

    def __init__(self,
                 grid: np.ndarray,
                 verbose: bool = False):
        if verbose:
            print(f"Initializing sampler {type(self)}:\n")
        self.verbose = verbose
        self._grid = grid
        if verbose:
            print(f"grid shape is {self._grid.shape}\n")

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    def __len__(self):
        return self._grid.shape[0]

    def __iter__(self):
        yield from self._grid


# def get_sampler(shape: Sequence[int], stride: Sequence[int], offset: Sequence[int]):
