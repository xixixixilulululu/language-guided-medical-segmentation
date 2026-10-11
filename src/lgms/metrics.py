"""Common per-case metrics of the final evaluator (Task 0.5).

All three methods are scored with these functions, in the common 224x224 space:

* the prediction is the foreground probability map; it is binarised with `prob > 0.5`
  (for the two-class softmax models TransUNet and nnU-Net this is identical to argmax);
* Dice  = 2|P∩G| / (|P| + |G|)
* IoU   = |P∩G| / |P∪G| of the lesion class; "mIoU" is the mean of this per-case IoU over cases,
  as in LViT (test_model.py: sklearn jaccard_score of the binary lesion mask, averaged over images);
* HD95 and ASSD from medpy (medpy.metric.binary.hd95 / assd), voxel spacing 1, i.e. in pixels of
  the 224x224 image, surface connectivity 1 (medpy default). In medpy 0.5.2 ASSD is the mean of the
  pooled surface distances P->G and G->P (sum of all distances / number of surface pixels of both).

Empty-mask rules (distances are undefined when one of the masks is empty):

* prediction empty, ground truth not empty: Dice = IoU = 0, HD95 = ASSD = MAX_DIST;
* ground truth empty, prediction not empty: Dice = IoU = 0, HD95 = ASSD = MAX_DIST;
* both empty: Dice = IoU = 1, HD95 = ASSD = 0 (perfect agreement).
  (No Val/Test mask is empty in either dataset; the rule only matters for completeness.)

MAX_DIST is the image diagonal 224*sqrt(2) = 316.78 px, the largest possible distance.
"""
import math

import numpy as np
from medpy.metric import binary as mbin

IMG_SIZE = 224
THRESHOLD = 0.5
MAX_DIST = IMG_SIZE * math.sqrt(2.0)  # 316.78 px


def binarize(prob):
    """Foreground probability (or 0/1 mask) -> bool mask with the common threshold."""
    return np.asarray(prob) > THRESHOLD


def dice_score(pred, gt):
    """Dice of two bool masks (empty rules as in the module docstring)."""
    ps, gs = pred.sum(), gt.sum()
    if ps == 0 and gs == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / (ps + gs))


def case_metrics(pred, gt):
    """All four metrics for one case. `pred`, `gt`: bool arrays of the same shape (224x224)."""
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    assert pred.shape == gt.shape, (pred.shape, gt.shape)
    ps, gs = int(pred.sum()), int(gt.sum())
    if ps == 0 and gs == 0:
        return {"dice": 1.0, "miou": 1.0, "hd95": 0.0, "assd": 0.0}
    if ps == 0 or gs == 0:
        return {"dice": 0.0, "miou": 0.0, "hd95": MAX_DIST, "assd": MAX_DIST}
    inter = int(np.logical_and(pred, gt).sum())
    union = ps + gs - inter
    return {
        "dice": 2.0 * inter / (ps + gs),
        "miou": inter / union,
        "hd95": float(mbin.hd95(pred, gt)),
        "assd": float(mbin.assd(pred, gt)),
    }


def dice_many(probs, gts):
    """Per-case Dice of N cases at once (used for the per-epoch validation Dice during training).

    probs: (N, H, W) foreground probabilities; gts: (N, H, W) {0, 1}. Same rules as dice_score.
    """
    p = binarize(probs)
    g = np.asarray(gts) > 0
    inter = np.logical_and(p, g).sum(axis=(1, 2))
    den = p.sum(axis=(1, 2)) + g.sum(axis=(1, 2))
    return np.where(den == 0, 1.0, 2.0 * inter / np.maximum(den, 1))
