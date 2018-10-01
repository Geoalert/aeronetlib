from functools import wraps

import keras.backend as K
from keras.losses import binary_crossentropy
from keras.losses import categorical_crossentropy
from keras.utils.generic_utils import get_custom_objects

from .metrics import iou_score, custom_iou_score
from .metrics import f1_score, custom_f_score


# ============================== Jaccard Losses ==============================

def jaccard_loss(gt, pr):
    return 1 - iou_score(gt, pr)


def _bce_jaccard_loss(gt, pr, bce_weight=1, smooth=1):

    iou_score = custom_iou_score(smooth=smooth)

    bce = K.mean(binary_crossentropy(gt, pr))
    jaccard_loss = 1 - iou_score(gt, pr)

    return bce_weight * bce + jaccard_loss


def _cce_jaccard_loss(gt, pr, bce_weight=1, class_weights=1, smooth=1):

    iou_score = custom_iou_score(class_weights=class_weights, smooth=smooth)

    cce = categorical_crossentropy(gt, pr) * class_weights
    cce = K.mean(cce)
    jaccard_loss = 1 - iou_score(gt, pr)
    return bce_weight * cce + jaccard_loss


def bce_jaccard_loss(gt, pr):
    return _bce_jaccard_loss(gt, pr, bce_weight=1, smooth=1)


def cce_jaccard_loss(gt, pr):
    return _cce_jaccard_loss(gt, pr, bce_weight=1, class_weights=1, smooth=1)


### Custom jaccard losses

def custom_jaccard_loss(class_weights=1, smooth=1):
    metric = custom_iou_score(class_weights=class_weights, smooth=smooth)
    def loss(gt, pr):
        return 1 - metric(gt, pr)
    return loss

def custom_bce_jaccard_loss(bce_weight=1, smooth=1):
    def loss(gt, pr):
        return _bce_jaccard_loss(gt, pr, bce_weight=bce_weight, smooth=smooth)
    return loss


def custom_cce_jaccard_loss(bce_weight=1, class_weights=1, smooth=1):
    def loss(gt, pr):
        return _cce_jaccard_loss(gt, pr, bce_weight=bce_weight, class_weights=class_weights, smooth=smooth)
    return loss


# Update custom objects
get_custom_objects().update({
    'jaccard_loss': jaccard_loss,
    'bce_jaccard_loss': bce_jaccard_loss,
    'cce_jaccard_loss': cce_jaccard_loss,
})


