"""Collect the Phase 1 quick-check outputs (runs/phase1_quick_check/) into results/phase1_quick_check.json.

Per method and dataset: the probes on the untrained model, the subset run (loss / validation Dice curve,
best epoch), the text-swap probe on the trained subset model, the resume test (6 epochs uninterrupted vs
3 + 3) and the measured seconds per epoch on the full data. For DD-CMD it also checks decision D3 from the
outputs: both callbacks monitor `val_case_dice`, and the selected best epoch is the epoch with the highest
per-case validation Dice in epochs.csv (the epoch with the highest official val_MIoU is shown next to it).
No test-set number is read or written (the quick checks never touch the test set).

Usage:  python src/phase1/summarize_quick_check.py
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1 import common  # noqa: E402

Q = os.path.join(common.REPO, "runs", "phase1_quick_check")
OUT = os.path.join(common.REPO, "results", "phase1_quick_check.json")
METHODS = ["lvit", "reclmis", "ddcmd"]
DATASETS = ["QaTa-COV19", "MosMedData+"]


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def jload(path):
    return json.load(open(path)) if os.path.exists(path) else None


def first_argmax(rs, col):
    """1-based epoch of the first maximum of a column (ties: the earliest, as Lightning / our selection)."""
    vals = [float(r[col]) for r in rs]
    return int(rs[vals.index(max(vals))]["epoch"])


def subset_summary(m, ds):
    d = os.path.join(Q, "subset", f"{m}_{ds}")
    if not os.path.exists(os.path.join(d, "epochs.csv")):
        return None
    rs = rows(os.path.join(d, "epochs.csv"))
    if not rs:
        return None   # run failed before its first epoch
    s = jload(os.path.join(d, "training_summary.json")) or {}
    out = {"epochs": len(rs), "train_loss_first": float(rs[0]["train_loss"]), "train_loss_last": float(rs[-1]["train_loss"]),
           "val_dice_first": float(rs[0]["val_dice"]), "val_dice_best": s.get("best_val_dice"),
           "best_epoch": s.get("best_epoch"), "argmax_val_dice_epoch": first_argmax(rs, "val_dice")}
    for k in ("steps", "rec_loss_zero_steps", "monitors"):
        if k in s:
            out[k] = s[k]
    if m == "reclmis":
        out["aux_losses_first_last"] = {k: [float(rs[0][k]), float(rs[-1][k])]
                                        for k in ("loss_ccl", "loss_text_rec", "loss_img_rec")}
    if m == "ddcmd":   # D3 check
        out["argmax_val_MIoU_official_epoch"] = first_argmax(rs, "val_MIoU_official")
        out["D3_selection_on_case_dice"] = bool(
            set((s.get("monitors") or {}).values()) == {"val_case_dice"} and s.get("best_epoch") == out["argmax_val_dice_epoch"])
    return out


def resume_summary(m):
    a, b = (os.path.join(Q, "resume", m, x, "epochs.csv") for x in ("uninterrupted", "segmented"))
    if not all(os.path.exists(p) for p in (a, b, os.path.join(os.path.dirname(b), "segments.jsonl"))):
        return None   # resume test not run (or failed before training started)
    ra, rb = rows(a), rows(b)
    seg = [json.loads(line) for line in open(os.path.join(Q, "resume", m, "segmented", "segments.jsonl"))]
    return {"epochs_uninterrupted": [int(r["epoch"]) for r in ra], "epochs_segmented": [int(r["epoch"]) for r in rb],
            "lr_identical": all(x["lr"] == y["lr"] for x, y in zip(ra, rb)),
            "max_abs_diff_train_loss": max(abs(float(x["train_loss"]) - float(y["train_loss"])) for x, y in zip(ra, rb)),
            "max_abs_diff_val_dice": max(abs(float(x["val_dice"]) - float(y["val_dice"])) for x, y in zip(ra, rb)),
            "segments": seg}


def timing_summary(m, ds):
    p = os.path.join(Q, "timing", f"{m}_{ds}", "epochs.csv")
    if not os.path.exists(p):
        return None
    rs = rows(p)
    if not rs:
        return None
    if "epoch_seconds" in rs[0]:                      # DD-CMD: one wall-clock time per epoch (train + val)
        sec = [float(r["epoch_seconds"]) for r in rs]
    else:
        sec = [float(r["train_seconds"]) + float(r["val_seconds"]) for r in rs]
    return {"seconds_per_epoch": sec, "seconds_per_epoch_last": sec[-1]}


def main():
    res = {}
    for m in METHODS:
        res[m] = {"resume": resume_summary(m)}
        for ds in DATASETS:
            res[m][ds] = {"probe_init": jload(os.path.join(Q, f"probe_init_{m}_{ds}.json")),
                          "subset": subset_summary(m, ds),
                          "probe_trained": jload(os.path.join(Q, f"probe_trained_{m}_{ds}.json")),
                          "timing": timing_summary(m, ds)}
            if res[m][ds]["probe_init"]:
                res[m][ds]["probe_init"].pop("pairing", None)   # long; kept in the probe file itself
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    for m in METHODS:
        for ds in DATASETS:
            r = res[m][ds]
            sub, tim, pt = r["subset"] or {}, r["timing"] or {}, r["probe_trained"] or {}
            print(f"{m:8s} {ds:12s} probes {'ok' if (r['probe_init'] or {}).get('passed') else '--'} | "
                  f"loss {sub.get('train_loss_first', float('nan')):.3f}->{sub.get('train_loss_last', float('nan')):.3f} | "
                  f"val Dice {sub.get('val_dice_first', float('nan')):.3f}->best {sub.get('val_dice_best') or float('nan'):.3f} | "
                  f"swap {pt.get('trained_val_dice_true_text', float('nan')):.4f}/{pt.get('trained_val_dice_fixed_text', float('nan')):.4f} | "
                  f"{tim.get('seconds_per_epoch_last', float('nan')):.0f} s/epoch"
                  + (f" | D3 {sub.get('D3_selection_on_case_dice')}" if m == "ddcmd" else ""))
    print("->", OUT)


if __name__ == "__main__":
    main()
