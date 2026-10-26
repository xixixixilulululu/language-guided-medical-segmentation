"""Summarise the post-hoc LViT-T sensitivity check (reports/PHASE1_SENSITIVITY_LVIT_PLAN.md).

Written and pushed before any sensitivity run started. Descriptive only, no significance test.
For LViT-T on MosMedData+ it reads, for seeds 1001/1002/1003:
  batch 24 (sensitivity): runs/lvit_b24/MosMedData+/seed<k>/{eval_test/summary.json, training_summary.json}
  batch 2 (main run):     runs/lvit/MosMedData+/seed<k>/{eval_test/summary.json, training_summary.json}
and writes results/phase1_sensitivity_lvit_b24.csv (one row per setting x seed, plus one mean row per
setting: mean and sample SD (ddof = 1) of the per-seed values, the same summary as the Phase 1 main table)
and prints the same table in Markdown. Dice and mIoU in %, HD95 and ASSD in pixels of the 224x224 image.
The paper value (LViT Table II, LViT-T on MosMedData+: Dice 74.57) is printed alongside.

Usage:  python src/phase1/summarize_sensitivity_lvit.py
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET = "MosMedData+"
SEEDS = [1001, 1002, 1003]
SETTINGS = [("batch 2 (main run)", "lvit"), ("batch 24 (sensitivity)", "lvit_b24")]
PAPER_DICE = 74.57                  # LViT paper, Table II (LViT-T, MosMedData+)
N_TEST = 273


def load_run(root, folder, seed):
    """Per-seed values of one run (test metrics of the validation-selected checkpoint + training record)."""
    d = os.path.join(root, folder, DATASET, f"seed{seed}")
    with open(os.path.join(d, "eval_test", "summary.json")) as f:
        s = json.load(f)
    with open(os.path.join(d, "training_summary.json")) as f:
        t = json.load(f)
    assert s["split"] == "test" and s["n_cases"] == N_TEST, (d, s["split"], s["n_cases"])
    return {"seed": seed, "dice_pct": 100 * s["dice"], "miou_pct": 100 * s["miou"], "hd95_px": s["hd95"],
            "assd_px": s["assd"], "empty_predictions": s["n_empty_predictions"], "best_epoch": t["best_epoch"],
            "best_val_dice_pct": 100 * t["best_val_dice"], "last_epoch": t["last_epoch"], "ended": t["ended"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default=os.path.join(REPO, "runs"))
    ap.add_argument("--out", default=os.path.join(REPO, "results", "phase1_sensitivity_lvit_b24.csv"))
    args = ap.parse_args()

    rows = []
    for name, folder in SETTINGS:
        per_seed = [dict(setting=name, **load_run(args.runs_root, folder, s)) for s in SEEDS]
        rows += per_seed
        df = pd.DataFrame(per_seed)
        mean = {"setting": name, "seed": "mean"}
        for c in ["dice_pct", "miou_pct", "hd95_px", "assd_px", "empty_predictions"]:
            mean[c] = df[c].mean()
            mean[c + "_sd"] = df[c].std(ddof=1)
        rows.append(mean)
    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False, float_format="%.4f")

    # Markdown: per seed, then mean ± SD; the paper value and the differences are descriptive only
    print("| Setting | Seed | Dice | mIoU | HD95 | ASSD | Empty predictions (of 273) | Best epoch (val. Dice) | Ended |")
    print("|---|---|---:|---:|---:|---:|---:|---|---|")
    for r in rows:
        if r["seed"] == "mean":
            f = lambda c, k=2: f"{r[c]:.{k}f} ± {r[c + '_sd']:.{k}f}"  # noqa: E731
            print(f"| {r['setting']} | mean ± SD | {f('dice_pct')} | {f('miou_pct')} | {f('hd95_px')} | "
                  f"{f('assd_px')} | {f('empty_predictions', 1)} | | |")
        else:
            print(f"| {r['setting']} | {r['seed']} | {r['dice_pct']:.2f} | {r['miou_pct']:.2f} | {r['hd95_px']:.2f} | "
                  f"{r['assd_px']:.2f} | {r['empty_predictions']} | {r['best_epoch']} ({r['best_val_dice_pct']:.2f}) | "
                  f"{r['ended']} (epoch {r['last_epoch']}) |")
    m = {r["setting"]: r["dice_pct"] for r in rows if r["seed"] == "mean"}
    b2, b24 = m[SETTINGS[0][0]], m[SETTINGS[1][0]]
    print(f"\nPaper (LViT Table II): Dice {PAPER_DICE:.2f}")
    print(f"Mean Dice, batch 24 - batch 2: {b24 - b2:+.2f} points; paper - batch 2: {PAPER_DICE - b2:+.2f}; "
          f"paper - batch 24: {PAPER_DICE - b24:+.2f}")
    print(f"written: {os.path.relpath(args.out, REPO)}")


if __name__ == "__main__":
    np.seterr(all="raise")
    main()
