from functools import wraps
import keras.backend as K
from keras.utils.generic_utils import get_custom_objects


# ============================ Jaccard score ============================

def _iou_score(gt, pr, class_weights=1., smooth=1e-12):
    axes = [1, 2]

    intersection = K.sum(gt * pr, axis=axes)
    union = K.sum(gt + pr, axis=axes) - intersection
    iou = (intersection + smooth) / (union + smooth)

    # mean per image
    iou = K.mean(iou, axis=0)

    # weighted mean per class
    iou = K.mean(iou * class_weights)

    return iou


def iou_score(gt, pr):
    return _iou_score(gt, pr, class_weights=1., smooth=1e-12)


def custom_iou_score(class_weights=1, smooth=1):
    def score(gt, pr):
        return _iou_score(gt, pr, class_weights=class_weights, smooth=smooth)
    return score


# Update custom objects
get_custom_objects().update({
    'iou_score': iou_score,
})


# ============================== F-score ==============================

def _f_score(gt, pr, class_weights=1, beta=1, smooth=1e-12):
    axes = [1, 2]

    tp = K.sum(gt * pr, axis=axes)
    fp = K.sum(pr, axis=axes) - tp
    fn = K.sum(gt, axis=axes) - tp

    f_score = ((1 + beta**2) * tp + smooth) \
              / ((1 + beta**2) * tp + beta**2 * fn + fp + smooth)

    # mean per image
    f_score = K.mean(f_score, axis=0)

    # weighted mean per class
    f_score = K.mean(f_score * class_weights)

    return f_score


def f1_score(gt, pr):
    return _f_score(gt, pr, class_weights=1, beta=1, smooth=1e-12)


def custom_f_score(class_weights=1, beta=1, smooth=1):
    def score(gt, pr):
        return _f_score(gt, pr, class_weights=class_weights, beta=beta, smooth=smooth)
    return score


# Update custom objects
get_custom_objects().update({
    'f1_score': f1_score,
})
