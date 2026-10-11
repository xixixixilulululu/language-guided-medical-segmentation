"""Collect the Task 0.3 quick-check results (runs/quick_check/) into results/phase0_quick_check.json.

Reads:
  steps1-4/<method>_<dataset>.json         quick_check.py records
  subset/<method>_<dataset>/epochs.csv     small-subset training (U-Net, TransUNet)
  subset/nnunet_<dataset>/selection.csv    snapshot selection timing (nnU-Net)
  timing/*/epochs.csv                      full-size epoch times (U-Net, TransUNet)
  nnunet_results/.../fold_0/training_log_*.txt   nnU-Net "Epoch time" lines
Only measured values are written; nothing is estimated here (the schedule is derived in the report).
"""
import glob
import json
import os
import re

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(REPO, "runs", "quick_check")
DATASETS = ["QaTa-COV19", "MosMedData+"]


def epochs(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def main():
    out = {"steps1_4": {}, "subset_training": {}, "epoch_seconds": {}, "nnunet": {}}
    for f in sorted(glob.glob(os.path.join(Q, "steps1-4", "*.json"))):
        out["steps1_4"][os.path.basename(f)[:-5]] = json.load(open(f))

    for m in ["unet", "transunet"]:
        for ds in DATASETS:
            e = epochs(os.path.join(Q, "subset", f"{m}_{ds}", "epochs.csv"))
            if e is not None:
                out["subset_training"][f"{m}_{ds}"] = {
                    "epochs": int(len(e)), "first_train_loss": float(e.train_loss.iloc[0]),
                    "last_train_loss": float(e.train_loss.iloc[-1]),
                    "min_train_loss": float(e.train_loss.min()),
                    "best_val_dice_32_cases": float(e.val_dice.max())}

    # full-size epochs: mean over the measured epochs (epoch 1 includes cuDNN / worker start-up)
    for d in sorted(glob.glob(os.path.join(Q, "timing", "*"))):
        e = epochs(os.path.join(d, "epochs.csv"))
        if e is None:
            continue
        out["epoch_seconds"][os.path.basename(d)] = {
            "epochs_measured": int(len(e)),
            "train_s": [round(x, 1) for x in e.train_seconds], "val_s": [round(x, 1) for x in e.val_seconds],
            "last_epoch_total_s": round(float(e.train_seconds.iloc[-1] + e.val_seconds.iloc[-1]), 1)}
        ts = os.path.join(d, "training_summary.json")
        if os.path.exists(ts):
            out["epoch_seconds"][os.path.basename(d)]["max_memory_gb"] = round(json.load(open(ts))["max_memory_gb"], 2)

    for ds, name in [("QaTa-COV19", "Dataset501_QaTaCOV19"), ("MosMedData+", "Dataset502_MosMedDataPlus")]:
        rec = {}
        logs = glob.glob(os.path.join(Q, "nnunet_results", name, "*", "fold_0", "training_log_*.txt"))
        if logs:
            txt = open(sorted(logs)[-1]).read()
            rec["epoch_seconds"] = [float(x) for x in re.findall(r"Epoch time: ([\d.]+) s", txt)]
        sel = os.path.join(Q, "subset", f"nnunet_{ds}", "selection.csv")
        if os.path.exists(sel):
            s = pd.read_csv(sel)
            rec["val_prediction_seconds_per_snapshot"] = [round(x, 1) for x in s.seconds]
            rec["val_dice_per_snapshot"] = [round(x, 4) for x in s.val_dice]
        ev = os.path.join(Q, "subset", f"nnunet_{ds}", "eval_test", "summary.json")
        rec["final_pipeline_complete"] = os.path.exists(ev)
        out["nnunet"][ds] = rec

    path = os.path.join(REPO, "results", "phase0_quick_check.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "steps1_4"}, indent=2))


if __name__ == "__main__":
    main()
