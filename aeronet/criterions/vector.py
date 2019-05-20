import numpy as np

EPS = 10e-12


# ============================= Object-wise metrics ==================================

def iou(gt_feature, pr_feature):
    intersection = gt_feature.intersection(pr_feature).area
    union = gt_feature.union(pr_feature).area
    return (intersection + 10e-12) / (union + 10e-12)


def collection_iou(gt_fc, pr_fc) -> np.array:
    
    scores = []
    
    for feature in gt_fc:

        proposed_features = pr_fc.intersection(feature)
        feature_scores = [iou(feature, f) for f in proposed_features]
        if feature_scores:
            max_iou_score = max([iou(feature, f) for f in proposed_features])
        else:
            max_iou_score = 0
        scores.append(max_iou_score)
    
    return np.array(scores)
        

def confusion_matrix(gt_fc, pr_fc, iou_threshold=0.5):
    
    scores = collection_iou(gt_fc, pr_fc)
    tp = sum(scores > iou_threshold)
    fp = len(pr_fc) - tp
    fn = len(gt_fc) - tp
    
    return tp, fp, fn


def mAPxx(gt_fc, pr_fc, thresholds, beta=1) -> float:
    
    scores = collection_iou(gt_fc, pr_fc)
    mAPs = []
    
    for threshold in thresholds:
        tp = sum(scores > threshold)
        fp = len(pr_fc) - tp
        fn = len(gt_fc) - tp
        
        mAP_ = ((1 + beta ** 2) * tp + EPS) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + EPS)
        mAPs.append(mAP_)
        
    return np.mean(mAPs)


def mAP(gt_fc, pr_fc, iou_threshold=0.5, beta=1) -> float:
    return mAPxx(gt_fc, pr_fc, thresholds=[iou_threshold], beta=beta)


def mAP50(gt_fc, pr_fc, beta=1) -> float:
    return mAPxx(gt_fc, pr_fc, beta=beta, thresholds=[0.5])


def mAP75(gt_fc, pr_fc, beta=1) -> float:
    return mAPxx(gt_fc, pr_fc, beta=beta, thresholds=[0.75])

def mAP5095(gt_fc, pr_fc, beta=1) -> float:
    thresholds = np.linspace(0.5, 0.95, 10)
    return mAPxx(gt_fc, pr_fc, beta=beta, thresholds=thresholds)


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
