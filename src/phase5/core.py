"""Phase 5 core: Soft Dice Confidence, Fixed Average, aggregation and the mixed-reliability crossover.

Definitions: configs/phase5/spec.json (committed before any data were read). Pure functions, tested on synthetic
data by src/phase5/test_phase5.py.
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from lgms import metrics  # noqa: E402

NAME = {"lvit": "LViT-T", "reclmis": "RecLMIS", "ddcmd": "DD-CMD"}
METHODS = ["lvit", "reclmis", "ddcmd"]
SEEDS = [1001, 1002, 1003]
CONDS = ["C0", "C2", "C4", "C5", "C6"]
WRONG = ["C4", "C5", "C6"]
KEYS = ["dataset", "method", "seed", "sample_id", "condition", "variant_id", "donor_sample_id"]
METRICS = ["dice", "miou", "hd95", "assd", "empty"]


def sdc(p):
    """Soft Dice Confidence of a foreground probability map: 2 sum(P B) / (sum P + sum B), B = P > 0.5; 0 if 0/0."""
    p = np.asarray(p, dtype=np.float64)
    b = (p > metrics.THRESHOLD).astype(np.float64)
    den = p.sum() + b.sum()
    return 0.0 if den == 0 else float(2.0 * (p * b).sum() / den)


def fixed_average(p_lang, p_img):
    """(P_avg, B_avg): 0.5 / 0.5 average of the stored float32 maps; B_avg = P_avg > 0.5 (strict)."""
    p = 0.5 * np.asarray(p_lang, np.float32) + 0.5 * np.asarray(p_img, np.float32)
    return p, metrics.binarize(p)


def score_mask(mask, gt):
    """Common evaluator metrics of a binary mask (src/evaluate.py _score) plus empty."""
    m = np.asarray(mask, bool)
    return {**metrics.case_metrics(m, np.asarray(gt) > 0), "empty": float(not m.any())}


def tau_from_t(t):
    """SDC threshold reported in SDC units: tau = -t (t = +inf -> tau = -inf: never fall back)."""
    return -float(t)


def sdc_switch(s, tau):
    """Fall back to IMG if SDC < tau."""
    return np.asarray(s, float) < tau


def group_weighted_mean(S, value="dice"):
    """Per (method, condition): mean over groups of [mean over images of [mean over seeds of [mean over donors]]]."""
    k = ["method", "condition"]
    v = S.groupby(k + ["group", "sample_id", "seed"])[value].mean()           # donors within image and seed
    v = v.groupby(k + ["group", "sample_id"]).mean()                           # seeds within image
    v = v.groupby(k + ["group"]).mean()                                        # images within group
    return v.groupby(k).mean()                                                 # groups equally


def crossover(delta_clean, delta_wrong, tol=0.0):
    """Exact p* of delta_clean + p (delta_wrong - delta_clean) = 0 in [0, 1], else None."""
    slope = delta_wrong - delta_clean
    if abs(slope) <= tol:
        return None
    p = -delta_clean / slope
    return float(p) if 0.0 <= p <= 1.0 else None


def better_interval(delta_clean, delta_wrong):
    """(lo, hi) closed-open description of {p in [0, 1]: delta_clean + p (delta_wrong - delta_clean) > 0}, or None."""
    a, b = delta_clean, delta_wrong - delta_clean
    if b == 0:
        return (0.0, 1.0) if a > 0 else None
    p = -a / b
    lo, hi = (max(0.0, p), 1.0) if b > 0 else (0.0, min(1.0, p))
    if lo > hi or (lo == hi and not (a + b * lo > 0)):
        return None
    return (lo, hi)


def intersect(i1, i2):
    if i1 is None or i2 is None:
        return None
    lo, hi = max(i1[0], i2[0]), min(i1[1], i2[1])
    return (lo, hi) if lo < hi else None


def expected(d_clean, d_wrong, p):
    return (1 - p) * d_clean + p * d_wrong
