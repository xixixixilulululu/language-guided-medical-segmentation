"""Phase 5 Part 2, Task 5.2.1: reconcile the U-Net test inference with the Phase 0 archive (exact rule).

Rule (configs/phase5_part2/spec.json "reconciliation"; deterministic tier of P2-2): for every seed, the mask
P > 0.5 of runs/phase3/test/img/unet/QaTa-COV19/seed<k>/probs.npz equals the archived Phase 0 PNG mask
(runs/unet/QaTa-COV19/seed<k>/eval_test/masks/<id>.png) pixel by pixel, the stored masks.npz equals
packbits(P > 0.5), and the per-case Dice equals the archived per_case.csv within 1e-6.
Output: results/phase5_part2_unet_img_test_per_case.csv.gz (keys, metrics, reconciliation columns; never
overwritten). Exit status 1 if the rule is not met (the file is still written as the record of the check).
Usage (in slurm/phase5_part2_unet_infer.slurm, after infer_img.py):  python src/phase5_part2/reconcile_unet.py
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from phase5 import guards  # noqa: E402

DATASET, SEEDS = "QaTa-COV19", [1001, 1002, 1003]
OUT = os.path.join("results", "phase5_part2_unet_img_test_per_case.csv.gz")
DICE_TOL = 1e-6
METRICS = ["dice", "miou", "hd95", "assd", "empty"]


def new_dir(seed):
    return os.path.join(REPO, "runs", "phase3", "test", "img", "unet", DATASET, f"seed{seed}")


def archive_dir(seed):
    return os.path.join(REPO, "runs", "unet", DATASET, f"seed{seed}", "eval_test")


def reconcile_rows(ids, probs, packed, archived_masks, new_dice, old_dice):
    """Per-image reconciliation: pixels differing from the archived mask, |Delta Dice|, stored-mask check."""
    rows = []
    for i in ids:
        m = np.asarray(probs[i]) > 0.5
        rows.append({"sample_id": i, "stored_mask_ok": bool(np.array_equal(np.packbits(m), packed[i])),
                     "mask_pixels_differing": int((m != (np.asarray(archived_masks[i]) > 0)).sum()),
                     "phase0_dice": float(old_dice[i]), "abs_dice_diff": abs(float(new_dice[i]) - float(old_dice[i]))})
    return pd.DataFrame(rows)


def rule_met(r):
    """Exact rule: identical masks, stored masks consistent, |Delta Dice| <= 1e-6 for every image."""
    return bool(r.stored_mask_ok.all() and (r.mask_pixels_differing == 0).all() and (r.abs_dice_diff <= DICE_TOL).all())


def summary(r, seed):
    d = r.abs_dice_diff
    return {"seed": seed, "n": len(r), "cases_mask_differs": int((r.mask_pixels_differing > 0).sum()),
            "max_pixels_differing": int(r.mask_pixels_differing.max()), "mean_abs_dice_diff": float(d.mean()),
            "max_abs_dice_diff": float(d.max()), "stored_mask_ok": bool(r.stored_mask_ok.all()), "rule_met": rule_met(r)}


def main():
    import cv2
    from phase2 import lock
    lock.require_open(what="Phase 5 Part 2 U-Net reconciliation")
    guards.outputs_absent(REPO, [OUT])
    parts, sums = [], []
    for s in SEEDS:
        nd, ad = new_dir(s), archive_dir(s)
        pc = pd.read_csv(os.path.join(nd, "per_case.csv"), dtype={"sample_id": str})
        old = pd.read_csv(os.path.join(ad, "per_case.csv"), dtype={"sample_id": str}).set_index("sample_id").dice
        ids = list(pc.sample_id)
        if set(ids) != set(old.index) or len(ids) != 2113:
            guards.stop(f"seed {s}: image ids differ from the Phase 0 archive or are not 2113")
        probs, packed = np.load(os.path.join(nd, "probs.npz")), np.load(os.path.join(nd, "masks.npz"))
        archived = {i: cv2.imread(os.path.join(ad, "masks", f"{i}.png"), cv2.IMREAD_GRAYSCALE) for i in ids}
        r = reconcile_rows(ids, probs, packed, archived, pc.set_index("sample_id").dice, old)
        sums.append(summary(r, s))
        parts.append(pc[["dataset", "arch", "seed", "sample_id"] + METRICS].merge(r, on="sample_id", validate="one_to_one"))
    guards.write_atomic(pd.concat(parts, ignore_index=True), REPO, OUT, float_format="%.17g")
    S = pd.DataFrame(sums)
    print(S.to_string(index=False))
    if not S.rule_met.all():
        print("RECONCILIATION RULE NOT MET: stop and report (spec 'reconciliation')", file=sys.stderr)
        sys.exit(1)
    print("reconciliation rule met for all seeds")


if __name__ == "__main__":
    main()
