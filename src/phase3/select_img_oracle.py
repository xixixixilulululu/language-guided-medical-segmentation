"""Phase 3, Task 3.1C (choose one IMG architecture) and Task 3.1D (oracle diagnostics), QaTa GATE_FIT only.

Rules fixed in reports/PHASE3_SETUP.md, section 5 (items 1, 2, 8, 9, 10), before any Phase 3 result:
  3.1C  Clean IMG Dice from the Phase 3 float32 rerun (runs/phase3/val/img), GATE_FIT images only: seed mean per
        image -> mean per group -> mean of the group means. The higher architecture wins; |difference| < 1e-6 ->
        nnU-Net. One choice for all language models and conditions; seeds matched (LANG seed k <-> IMG seed k).
  3.1D  On GATE_FIT, conditions C4-C6, per row (image, seed, condition, donor): LANG Dice from Phase 2 (the C4-C6
        probabilities, masks and scores of one pass; item 1), IMG Dice of the chosen architecture, same seed:
          actual_gain = Dice(IMG) - Dice(LANG); oracle_gain = max(Dice(IMG), Dice(LANG)) - Dice(LANG)
        IMG better / LANG better / neutral with tolerance 1e-6 (item 10). Summaries per method x seed x condition,
        image-weighted and group-weighted (donors -> images -> groups), on S_FIT (FIT targets eligible for C4, C5
        and C6). Headroom of a language model = the group-weighted mean oracle gain over C4, C5, C6 on S_FIT
        (mean of the three conditions), averaged over the three seeds; < 0.005 -> "insufficient headroom".
The oracle uses the ground truth; it is a diagnostic, never a gate input.

Usage:  python src/phase3/select_img_oracle.py
        -> results/phase3_fallback_selection.csv, results/phase3_oracle_diagnostics.csv
Tests:  python src/phase3/test_select_img_oracle.py
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEEDS = [1001, 1002, 1003]
METHODS = ["lvit", "reclmis", "ddcmd"]
NAME = {"lvit": "LViT-T", "reclmis": "RecLMIS", "ddcmd": "DD-CMD", "unet": "U-Net", "nnunet": "nnU-Net"}
MISMATCH = ["C4", "C5", "C6"]
TIE, NEUTRAL, HEADROOM_MIN = 1e-6, 1e-6, 0.005


def choose_img(img, fit):
    """img: DataFrame arch, seed, sample_id, dice (QaTa validation); fit: DataFrame sample_id, group_id (GATE_FIT)."""
    rows = []
    for arch in ("unet", "nnunet"):
        t = img[(img.arch == arch) & img.sample_id.isin(fit.sample_id)]
        per_img = t.groupby("sample_id").dice.mean()
        g = fit.set_index("sample_id").group_id.reindex(per_img.index)
        r = {"arch": NAME[arch], "n_images": len(per_img), "n_groups": g.nunique(),
             "group_weighted_dice": float(per_img.groupby(g).mean().mean()), "image_weighted_dice": float(per_img.mean())}
        for s in SEEDS:
            r[f"image_weighted_dice_seed{s}"] = float(t[t.seed == s].dice.mean())
        rows.append(r)
    out = pd.DataFrame(rows).set_index("arch")
    d = out.loc["U-Net", "group_weighted_dice"] - out.loc["nnU-Net", "group_weighted_dice"]
    chosen = "nnU-Net" if abs(d) < TIE or d < 0 else "U-Net"
    out["chosen"] = out.index == chosen
    out["difference_unet_minus_nnunet"] = d
    return out.reset_index(), {"U-Net": "unet", "nnU-Net": "nnunet"}[chosen]


def oracle_rows(lang, img_arch, fit):
    """Per-row gains on GATE_FIT C4-C6. lang: per-case rows (dataset QaTa, method, seed, sample_id, condition,
    variant_id, dice); img_arch: DataFrame seed, sample_id, dice of the chosen architecture."""
    l = lang[lang.condition.isin(MISMATCH) & lang.sample_id.isin(fit.sample_id)].copy()
    i = img_arch.set_index(["seed", "sample_id"]).dice
    l["dice_img"] = i.reindex(pd.MultiIndex.from_arrays([l.seed, l.sample_id])).values
    if l.dice_img.isna().any():
        raise ValueError("IMG Dice missing for some LANG rows")
    l["actual_gain"] = l.dice_img - l.dice
    l["oracle_gain"] = np.maximum(l.dice_img, l.dice) - l.dice
    l["outcome"] = np.where(l.actual_gain.abs() <= NEUTRAL, "neutral", np.where(l.actual_gain > 0, "IMG better", "LANG better"))
    l["group_id"] = fit.set_index("sample_id").group_id.reindex(l.sample_id).values
    return l


def summaries(rows):
    """Per method x seed x condition on S_FIT, plus per method the headroom decision."""
    s_fit = set.intersection(*[set(rows[rows.condition == c].sample_id) for c in MISMATCH])
    r = rows[rows.sample_id.isin(s_fit)]
    out = []
    for (m, s, c), g in r.groupby(["method", "seed", "condition"]):
        per_img = g.groupby(["sample_id", "group_id"])[["oracle_gain", "actual_gain"]].mean().reset_index()
        q = g.actual_gain.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        out.append({"method": NAME[m], "seed": s, "condition": c, "n_rows": len(g), "n_images": len(per_img),
                    "n_groups": per_img.group_id.nunique(),
                    "oracle_gain_group_weighted": float(per_img.groupby("group_id").oracle_gain.mean().mean()),
                    "oracle_gain_image_weighted": float(per_img.oracle_gain.mean()),
                    "actual_gain_group_weighted": float(per_img.groupby("group_id").actual_gain.mean().mean()),
                    "actual_gain_image_weighted": float(per_img.actual_gain.mean()),
                    "frac_rows_img_better": float((g.outcome == "IMG better").mean()),
                    "frac_rows_lang_better": float((g.outcome == "LANG better").mean()),
                    "frac_rows_neutral": float((g.outcome == "neutral").mean()),
                    "actual_gain_p05": q[0.05], "actual_gain_p25": q[0.25], "actual_gain_median": q[0.5],
                    "actual_gain_p75": q[0.75], "actual_gain_p95": q[0.95],
                    "max_gain": float(g.actual_gain.max()), "max_loss": float(g.actual_gain.min())})
    t = pd.DataFrame(out)
    head = (t.groupby(["method", "seed"]).oracle_gain_group_weighted.mean()        # mean of C4, C5, C6
             .groupby("method").mean().rename("headroom_mean_over_seeds"))
    t = t.merge(head, on="method")
    t["insufficient_headroom"] = t.headroom_mean_over_seeds < HEADROOM_MIN
    t["n_S_fit_images"] = len(s_fit)
    return t


def main():
    fit = pd.read_csv(os.path.join(REPO, "results", "phase3_gate_partitions.csv"), dtype=str)
    fit = fit[fit.partition == "GATE_FIT"]
    img = pd.concat([pd.read_csv(os.path.join(REPO, "runs", "phase3", "val", "img", a, "QaTa-COV19", f"seed{s}",
                                              "per_case.csv"), dtype={"sample_id": str}) for a in ("unet", "nnunet")
                     for s in SEEDS], ignore_index=True)
    sel, arch = choose_img(img, fit)
    sel.to_csv(os.path.join(REPO, "results", "phase3_fallback_selection.csv"), index=False, float_format="%.6f")
    print(sel.to_string(index=False))
    pc = pd.read_csv(os.path.join(REPO, "results", "phase2_val_per_case.csv.gz"), dtype={"sample_id": str, "variant_id": str})
    lang = pc[pc.dataset == "QaTa-COV19"]
    rows = oracle_rows(lang, img[img.arch == arch][["seed", "sample_id", "dice"]], fit)
    t = summaries(rows)
    t.insert(0, "img_arch", NAME[arch])
    t.to_csv(os.path.join(REPO, "results", "phase3_oracle_diagnostics.csv"), index=False, float_format="%.6g")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(t[["method", "seed", "condition", "n_images", "n_groups", "oracle_gain_group_weighted",
                 "actual_gain_group_weighted", "frac_rows_img_better", "max_gain", "max_loss",
                 "headroom_mean_over_seeds", "insufficient_headroom"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
