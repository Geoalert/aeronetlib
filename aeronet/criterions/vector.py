import numpy as np

EPS = 10e-12


# ============================= Object-wise metrics ==================================

def iou(gt_feature, pr_feature):
    intersection = gt_feature.intersection(pr_feature).area
    union = gt_feature.union(pr_feature).area
    return (intersection + 10e-12) / (union + 10e-12)


def confusion_matrix(gt_fc, pr_fc, iou_threshold=0.5):
    tp = 0
    fn = 0

    for feature in gt_fc:

        proposed_features = pr_fc.intersection(feature)
        iou_scores = [iou(feature, f) for f in proposed_features]

        if len(iou_scores) == 0:
            fn += 1
        elif max(iou_scores) >= iou_threshold:
            tp += 1
        else:
            fn += 1
    fp = len(pr_fc) - tp
    return tp, fp, fn


def mAP(gt_fc, pr_fc, iou_threshold=0.5, beta=1):
    tp, fp, fn = confusion_matrix(gt_fc, pr_fc, iou_threshold=iou_threshold)
    score = ((1 + beta ** 2) * tp + EPS) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + EPS)
    return score


def mAP50(gt_fc, pr_fc, beta=1):
    return mAP(gt_fc, pr_fc, beta=beta, iou_threshold=0.5)


def mAP75(gt_fc, pr_fc, beta=1):
    return mAP(gt_fc, pr_fc, beta=beta, iou_threshold=0.75)


def mAPxx(gt_fc, pr_fc, thresholds, beta=1):
    mAPs = [mAP(gt_fc, pr_fc, beta=beta, iou_threshold=t) for t in thresholds]
    return np.mean(mAPs)


def mAP5095(gt_fc, pr_fc, beta=1):
    thresholds = np.linspace(0.5, 0.95, 10)
    return mAPxx(gt_fc, pr_fc, thresholds, beta=beta)


# ============================= Area metrics ==================================

def _union(fc):
    a = fc[0]
    for f in fc:
        a = a.union(f)
    return a


def area_iou(gt_fc, pr_fc):
    gt = _union(gt_fc)
    pr = _union(pr_fc)

    intersection = gt.intersection(pr).area
    union = gt.union(pr).area

    return (intersection + EPS) / (union + EPS)
