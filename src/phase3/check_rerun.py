"""Phase 3: checks of the validation reruns (reports/PHASE3_SETUP.md, section 6 and item 16). Read only.

  1. Completeness: every IMG run (2 archs x 2 datasets x 3 seeds) and every LANG C2/C3 pass (3 methods x 2 datasets
     x 3 seeds x 4 conditions) exists, with probabilities, masks and per-case scores for the same ids.
  2. Consistency on ALL cases of every new file: the stored mask equals probability > 0.5; probabilities are float32.
  3. Reproducibility (item 16): LANG C2/C3 masks vs. the Phase 2 masks of the same pass (LViT-T and RecLMIS must be
     identical; DD-CMD differences are recorded); IMG vs. the Phase 0 archive (pixels and Dice, reported as is).

Usage:  python src/phase3/check_rerun.py   -> results/phase3_rerun_checks.csv (exit 1 if 1 or 2 fails, or if
        LViT-T / RecLMIS differ from Phase 2)
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = os.path.join(REPO, "runs", "phase3", "val")
DATASETS = ["QaTa-COV19", "MosMedData+"]
SEEDS = [1001, 1002, 1003]


def unpack(npz, k):
    return np.unpackbits(npz[k])[:224 * 224].reshape(224, 224).astype(bool)


def consistent(masks, probs, ids):
    """(all masks == prob > 0.5, all float32) over every id."""
    return (all((unpack(masks, i) == (probs[i] > 0.5)).all() for i in ids),
            all(probs[i].dtype == np.float32 for i in ids))


def main():
    rows = []
    for arch in ("unet", "nnunet"):
        for ds in DATASETS:
            for s in SEEDS:
                d = os.path.join(RUN, "img", arch, ds, f"seed{s}")
                r = {"kind": "IMG", "model": arch, "dataset": ds, "seed": s, "condition": "image only"}
                if not os.path.exists(os.path.join(d, "per_case.csv")):
                    rows.append({**r, "complete": False})
                    continue
                pc = pd.read_csv(os.path.join(d, "per_case.csv"), dtype={"sample_id": str})
                m, p = np.load(os.path.join(d, "masks.npz")), np.load(os.path.join(d, "probs.npz"))
                ok_mask, ok_dtype = consistent(m, p, pc.sample_id)
                rec = pd.read_csv(os.path.join(d, "reconcile_phase0.csv"))
                rows.append({**r, "complete": set(m.files) == set(p.files) == set(pc.sample_id), "n": len(pc),
                             "mask_equals_prob": ok_mask, "float32": ok_dtype,
                             "cases_differing": int((rec.mask_pixels_differing > 0).sum()),
                             "max_pixels_differing": int(rec.mask_pixels_differing.max()),
                             "max_abs_dice_diff": float(rec.abs_dice_diff.max()),
                             "mean_dice_new": float(pc.dice.mean()), "mean_dice_reference": float(rec.dice_phase0.mean())})
    for mth in ("lvit", "reclmis", "ddcmd"):
        for ds in DATASETS:
            for s in SEEDS:
                d = os.path.join(RUN, "lang", mth, ds, f"seed{s}")
                for c in ("C2", "C3a", "C3b", "C3c"):
                    r = {"kind": "LANG", "model": mth, "dataset": ds, "seed": s, "condition": c}
                    f = os.path.join(d, f"per_case_{c}_0.csv")
                    if not os.path.exists(f):
                        rows.append({**r, "complete": False})
                        continue
                    pc = pd.read_csv(f, dtype={"sample_id": str})
                    m, p = np.load(os.path.join(d, f"masks_{c}_0.npz")), np.load(os.path.join(d, f"probs_{c}_0.npz"))
                    ok_mask, ok_dtype = consistent(m, p, pc.sample_id)
                    diff = pd.read_csv(os.path.join(d, f"phase2_mask_diff_{c}.csv")).pixels_differing_from_phase2
                    old = pd.read_csv(os.path.join(REPO, "runs", "phase2", "val", mth, ds, f"seed{s}", f"per_case_{c}_0.csv"))
                    rows.append({**r, "complete": set(m.files) == set(p.files) == set(pc.sample_id), "n": len(pc),
                                 "mask_equals_prob": ok_mask, "float32": ok_dtype,
                                 "cases_differing": int((diff > 0).sum()), "max_pixels_differing": int(diff.max()),
                                 "max_abs_dice_diff": float((pc.set_index("sample_id").dice
                                                             - old.set_index("sample_id").dice).abs().max()),
                                 "mean_dice_new": float(pc.dice.mean()), "mean_dice_reference": float(old.dice.mean())})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(REPO, "results", "phase3_rerun_checks.csv"), index=False, float_format="%.6g")
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(df.groupby(["kind", "model", "dataset"]).agg(
            runs=("complete", "size"), complete=("complete", "all"), mask_eq=("mask_equals_prob", "all"),
            f32=("float32", "all"), cases_diff=("cases_differing", "sum"), max_px=("max_pixels_differing", "max"),
            max_dDice=("max_abs_dice_diff", "max")).to_string())
    det = df[(df.kind == "LANG") & df.model.isin(["lvit", "reclmis"])]
    bad = (not df.complete.all() or not df.mask_equals_prob.all() or not df.float32.all()
           or det.cases_differing.sum() > 0)
    if det.cases_differing.sum() > 0:
        print("STOP: LViT-T / RecLMIS masks differ from Phase 2 (item 16)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
