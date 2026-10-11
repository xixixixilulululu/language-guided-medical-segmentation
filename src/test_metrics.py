"""Check the common evaluator (src/lgms/metrics.py) on hand-made 224x224 masks.

Every case is compared with an expected value that does not come from medpy:
  * an analytic value where one exists (identical, disjoint, half-overlapping, empty masks);
  * otherwise an independent brute-force implementation of the surface distances
    (4-neighbour surface extraction + all-pairs Euclidean distances), which follows the medpy
    definitions: surface = mask minus its erosion with the 4-connected cross;
    HD95 = 95th percentile of the pooled distances P->G and G->P; ASSD = mean of the same pooled
    distances (medpy >= 0.5 definition; medpy 0.4 averaged the two directed means instead).

Usage:  python src/test_metrics.py   (exits with status 1 if any check fails)
"""
import os
import sys

import numpy as np
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lgms import metrics  # noqa: E402

S = metrics.IMG_SIZE


def box(r0, r1, c0, c1):
    """Bool 224x224 mask with the rectangle rows r0..r1-1, cols c0..c1-1 set."""
    m = np.zeros((S, S), bool)
    m[r0:r1, c0:c1] = True
    return m


def surface(m):
    """Surface pixels: foreground pixels with at least one 4-neighbour in the background."""
    p = np.pad(m, 1)
    interior = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return np.argwhere(m & ~interior)


def brute_force(pred, gt):
    """Reference HD95 / ASSD without medpy."""
    d = cdist(surface(pred), surface(gt))
    d_pg, d_gp = d.min(axis=1), d.min(axis=0)
    pooled = np.hstack([d_pg, d_gp])
    return float(np.percentile(pooled, 95)), float(pooled.mean())


def random_blobs(rng, n):
    """Union of n random discs (a non-trivial shape for the brute-force comparison)."""
    yy, xx = np.mgrid[:S, :S]
    m = np.zeros((S, S), bool)
    for _ in range(n):
        cy, cx, r = rng.integers(30, 194), rng.integers(30, 194), rng.integers(4, 25)
        m |= (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
    return m


def main():
    D = metrics.MAX_DIST
    sq = box(80, 120, 80, 120)                         # 40x40 square
    cases = [
        # name, pred, gt, expected {metric: value} (None = brute-force reference)
        ("identical (40x40 square)", sq, sq, {"dice": 1, "miou": 1, "hd95": 0, "assd": 0}),
        ("disjoint single pixels 10 px apart", box(50, 51, 50, 51), box(50, 51, 60, 61),
         {"dice": 0, "miou": 0, "hd95": 10, "assd": 10}),
        ("disjoint squares (no overlap, 30 px gap)", box(20, 30, 20, 30), box(20, 30, 60, 70),
         {"dice": 0, "miou": 0, "hd95": None, "assd": None}),
        ("half overlap (10x10 shifted by 5 px)", box(20, 30, 20, 30), box(20, 30, 25, 35),
         {"dice": 0.5, "miou": 1 / 3, "hd95": None, "assd": None}),
        ("empty prediction", np.zeros((S, S), bool), sq, {"dice": 0, "miou": 0, "hd95": D, "assd": D}),
        ("empty ground truth", sq, np.zeros((S, S), bool), {"dice": 0, "miou": 0, "hd95": D, "assd": D}),
        ("both empty", np.zeros((S, S), bool), np.zeros((S, S), bool), {"dice": 1, "miou": 1, "hd95": 0, "assd": 0}),
    ]
    rng = np.random.default_rng(0)
    for k in range(5):
        p, g = random_blobs(rng, 3), random_blobs(rng, 3)
        inter, ps, gs = (p & g).sum(), p.sum(), g.sum()
        cases.append((f"random blobs #{k}", p, g, {"dice": 2 * inter / (ps + gs), "miou": inter / (ps + gs - inter),
                                                   "hd95": None, "assd": None}))

    ok = True
    print(f"{'case':44s} {'metric':6s} {'evaluator':>10s} {'expected':>10s}  ok")
    for name, p, g, exp in cases:
        got = metrics.case_metrics(p, g)
        ref = brute_force(p, g) if None in exp.values() else None
        for m in ["dice", "miou", "hd95", "assd"]:
            e = exp[m] if exp[m] is not None else ref[0 if m == "hd95" else 1]
            good = abs(got[m] - e) < 1e-6
            ok &= good
            print(f"{name:44s} {m:6s} {got[m]:10.4f} {e:10.4f}  {'yes' if good else 'NO'}")

    # Threshold: prob exactly 0.5 is background, anything above is lesion
    t = metrics.binarize(np.array([0.0, 0.4999, 0.5, 0.5001, 1.0]))
    good = t.tolist() == [False, False, False, True, True]
    ok &= good
    print(f"threshold prob > 0.5 on [0, .4999, .5, .5001, 1] -> {t.astype(int).tolist()}  {'yes' if good else 'NO'}")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
