import numpy as np

EPS = 10e-12


def __channels_flatten(arr):
    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])
    elif arr.ndim ==2:
        arr = arr.flatten()
    return arr


def IoU(gt, pr):
    """

    Args:
        gt: raster, 2D or 3D array with shape (H, W) or (H, W, C)
        pr: raster, 2D or 3D array with shape (H, W) or (H, W, C)

    Returns:
        IoU score for each channel

    """

    if gt.ndim != pr.ndim:
        raise ValueError('Targets have different shapes: {} and {}'.format(gt.ndim, pr.ndim))

    gt = __channels_flatten(gt)
    pr = __channels_flatten(pr)

    intersection = (gt * pr).sum(axis=0)
    union = (gt + pr).sum(axis=0) - intersection

    return (intersection + EPS) / (union + EPS)


def mIoU(gt, pr, class_weights=1.):
    classes_ious = IoU(gt, pr)
    return np.mean(classes_ious * class_weights)