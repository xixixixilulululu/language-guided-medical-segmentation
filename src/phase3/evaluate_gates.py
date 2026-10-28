"""Phase 3, Task 3.5: the fixed gates (as of commit b58cdf6) on the VALIDATION split; development data.

Nothing is fitted or tuned here. The gates are read from results/phase3_dgate_thresholds.json (D-Gate thresholds at full precision; the CSV
results/phase3_dgate_calibration.csv rounds them to 6 digits)
and results/phase3_lr_artifacts.json (LR scaler, coefficients, intercept, threshold) and applied to every
validation row (all nine conditions; QaTa GATE_FIT and GATE_CAL separately; MosMedData+ with the QaTa gate of the
same language model x seed, the MosMed checkpoints and the MosMed nnU-Net of the same seed).

Strategies per row (dataset x model x seed x sample x condition x donor):
  LANG    the language-guided prediction under the supplied text
  IMG     nnU-Net, same seed
  D-Gate  IMG if F1 > threshold else LANG          LR-Gate  IMG if LR score > threshold else LANG
  NULL    (reference) the C1 prediction of the same image and seed; under C1 it is LANG itself
  ORACLE  (reference, not deployable) IMG if Dice(IMG) > Dice(LANG) else LANG
A gate's metrics are those of the selected prediction (item 15). All strategies of a condition use the same rows.

Tables (results/):
  phase3_validation_results.csv   image-weighted, per dataset x part x method x condition x strategy: Dice, mIoU,
                                  HD95, ASSD, empty rate, fallback rate as mean +- SD over the 3 per-seed means
                                  (donors averaged within image and seed, as in Phase 2); plus the group-weighted
                                  paired differences gate - LANG (donors -> seeds -> images -> groups) with a 95%
                                  t-CI over groups: descriptive, no test
  phase3_fallback_rates.csv       switch rate per seed and gate
  phase3_routing_quality.csv      switched rows, beneficial / harmful / neutral (|dDice| <= 1e-6), net Dice gain
  phase3_mosmed_exploratory.csv   MosMed: per model x condition, units (11), oracle headroom, gate - LANG,
                                  unnecessary C0 / C2 switches
  phase3_val_gate_decisions.csv.gz  keys, partition, F1, LR score, D-Gate switch, LR-Gate switch
Test (--split test; Phase 2 lock open, both protocols frozen): the same gates, unchanged, on the test rows; LANG
from results/phase2_test_per_case.csv.gz (all conditions from the Phase 2 test benchmark; no rerun), IMG from
results/phase3_img_test_per_case.csv.gz; QaTa part "test". Outputs: phase3_test_results.csv,
phase3_test_paired_differences.csv, phase3_test_fallback_rates.csv, phase3_test_routing_quality.csv,
phase3_test_mosmed_exploratory.csv, phase3_test_gate_decisions.csv.gz, phase3_test_per_case.csv.gz (every row x
strategy with the metrics of the selected prediction). Inference: src/phase3/stats_phase3.py.
Usage:  python src/phase3/evaluate_gates.py [--split test]
Tests:  python src/phase3/test_evaluate_gates.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
R = lambda f: os.path.join(REPO, "results", f)  # noqa: E731
NAME = {"lvit": "LViT-T", "reclmis": "RecLMIS", "ddcmd": "DD-CMD"}
CODE = {v: k for k, v in NAME.items()}
CONDITIONS = ["C0", "C1", "C2", "C3a", "C3b", "C3c", "C4", "C5", "C6"]
STRATEGIES = ["LANG", "IMG", "D-Gate", "LR-Gate", "NULL (C1)", "ORACLE"]   # "NULL" alone is read as NaN by pandas
METRICS = ["dice", "miou", "hd95", "assd", "empty"]
FEATS = ["F1", "F2", "F3", "F4", "F5"]
KEYS = ["dataset", "method", "seed", "sample_id", "condition", "variant_id", "donor_sample_id"]
NEUTRAL = 1e-6
RERUN = {"C2", "C3a", "C3b", "C3c"}


def load_rows(split="val"):
    """One DataFrame: keys, partition, group, F1-F5, LANG / IMG / NULL metrics."""
    dt = {"sample_id": str, "variant_id": str, "donor_sample_id": str}
    sfx = "" if split == "val" else "_test"
    F = pd.read_csv(R(f"phase3_reliability_features{sfx}.csv.gz"), dtype=dt, keep_default_na=False)
    L = pd.read_csv(R(f"phase3_feature_labels{sfx}.csv.gz"), dtype=dt, keep_default_na=False)
    D = F.merge(L, on=KEYS, validate="one_to_one")
    p2 = pd.read_csv(R(f"phase2_{split}_per_case.csv.gz"), dtype=dt, keep_default_na=False)
    if split == "val":
        p3 = pd.read_csv(R("phase3_lang_rerun_val_per_case.csv.gz"), dtype=dt, keep_default_na=False)
        lang = pd.concat([p2[~p2.condition.isin(RERUN)], p3], ignore_index=True)   # item 1
    else:
        lang = p2                                                                    # test: one benchmark run
    lang = lang[KEYS + METRICS].rename(columns={m: f"lang_{m}" for m in METRICS})
    D = D.merge(lang, on=KEYS, validate="one_to_one")
    img = pd.read_csv(R(f"phase3_img_{split}_per_case.csv.gz"), dtype={"sample_id": str})
    arch = {"U-Net": "unet", "nnU-Net": "nnunet"}[pd.read_csv(R("phase3_fallback_selection.csv")).query("chosen").arch.iloc[0]]
    img = img[img.arch == arch][["dataset", "seed", "sample_id"] + METRICS].rename(columns={m: f"img_{m}" for m in METRICS})
    D = D.merge(img, on=["dataset", "seed", "sample_id"], validate="many_to_one")
    null = D[D.condition == "C1"][["dataset", "method", "seed", "sample_id"] + [f"lang_{m}" for m in METRICS]]
    D = D.merge(null.rename(columns={f"lang_{m}": f"null_{m}" for m in METRICS}),
                on=["dataset", "method", "seed", "sample_id"], validate="many_to_one")
    if not (np.allclose(D.lang_dice, D.dice_lang, atol=1e-6) and np.allclose(D.img_dice, D.dice_img, atol=1e-6)):
        raise ValueError("per-case Dice differs from the labels file (inconsistent artifacts)")
    for p in ("lang", "img", "null"):
        D[f"{p}_empty"] = D[f"{p}_empty"].astype(str).eq("True").astype(float)
    D["part"] = np.where(D.dataset == "QaTa-COV19", D.partition, "MosMedData+ (all)")
    return D, arch


def apply_gates(D):
    """D-Gate and LR-Gate decisions from the fixed artifacts (no fitting)."""
    dg = json.load(open(R("phase3_dgate_thresholds.json")))                 # full precision (not the rounded CSV)
    art = json.load(open(R("phase3_lr_artifacts.json")))
    D = D.copy()
    D["lr_score"], D["d_switch"], D["lr_switch"] = np.nan, False, False
    for (m, s), idx in D.groupby(["method", "seed"]).groups.items():
        t = dg[f"{NAME[m]}_seed{s}"]["threshold"]
        a = art[f"{NAME[m]}_seed{s}"]
        X = (D.loc[idx, FEATS].to_numpy() - np.array(a["scaler_mean"])) / np.array(a["scaler_scale"])
        score = X @ np.array(a["coef"]) + a["intercept"]
        D.loc[idx, "lr_score"] = score
        D.loc[idx, "d_switch"] = D.loc[idx, "F1"].to_numpy() > t
        D.loc[idx, "lr_switch"] = score > a["threshold"]
    return D


def strategy_metrics(D):
    """Long table: one row per input row x strategy, with the metrics of the selected prediction and a switch flag."""
    out = []
    base = D[["dataset", "part", "group", "method", "seed", "sample_id", "condition", "variant_id"]]
    oracle = D.img_dice > D.lang_dice
    sel = {"LANG": np.zeros(len(D), bool), "IMG": np.ones(len(D), bool), "D-Gate": D.d_switch.to_numpy(),
           "LR-Gate": D.lr_switch.to_numpy(), "ORACLE": oracle.to_numpy()}
    for s, use_img in sel.items():
        t = base.copy()
        t["strategy"], t["switch"] = s, use_img.astype(float)
        for m in METRICS:
            t[m] = np.where(use_img, D[f"img_{m}"], D[f"lang_{m}"])
        out.append(t)
    t = base.copy()
    t["strategy"], t["switch"] = "NULL (C1)", np.nan
    for m in METRICS:
        t[m] = D[f"null_{m}"].to_numpy()
    out.append(t)
    return pd.concat(out, ignore_index=True)


def image_weighted(S):
    """mean +- SD over the 3 per-seed means (donors averaged within image and seed first)."""
    k = ["dataset", "part", "method", "condition", "strategy"]
    per_img = S.groupby(k + ["seed", "sample_id"])[METRICS + ["switch"]].mean()
    per_seed = per_img.groupby(k + ["seed"]).mean()
    n = S.groupby(k).agg(n_images=("sample_id", "nunique"), n_groups=("group", "nunique"), n_rows=("sample_id", "size"))
    res = per_seed.groupby(k).agg(["mean", "std"])
    res.columns = [f"{a}_{b}" if b == "mean" else f"{a}_sd" for a, b in res.columns]
    res = res.rename(columns={f"{m}_mean": m for m in METRICS + ["switch"]})
    return n.join(res).reset_index().rename(columns={"switch": "fallback_rate", "switch_sd": "fallback_rate_sd"})


def paired_group(S, gate):
    """Group-weighted gate - LANG per dataset x part x method x condition, 95% t-CI over groups (descriptive)."""
    import analyze_phase0
    k = ["dataset", "part", "method", "condition"]
    a = S[S.strategy == gate].set_index(k + ["group", "seed", "sample_id", "variant_id"]).dice
    b = S[S.strategy == "LANG"].set_index(k + ["group", "seed", "sample_id", "variant_id"]).dice
    d = (a - b).groupby(k + ["group", "seed", "sample_id"]).mean().groupby(k + ["group", "sample_id"]).mean()
    rows = []
    for key, g in d.groupby(k):
        u = g.groupby("group").mean()
        r = dict(zip(k, key), comparison=f"{gate} - LANG", n_groups=len(u), group_weighted_delta=float(u.mean()))
        if len(u) >= 2 and u.std(ddof=1) > 0:
            t = analyze_phase0.paired_t(u.values, np.zeros(len(u)))
            r.update(ci95_low=t["ci_low"], ci95_high=t["ci_high"])
        rows.append(r)
    return pd.DataFrame(rows)


def routing(D):
    rows = []
    for gate, col in (("D-Gate", "d_switch"), ("LR-Gate", "lr_switch")):
        for key, g in D.groupby(["dataset", "part", "method", "condition"]):
            sw = g[g[col]]
            gain = sw.img_dice - sw.lang_dice
            net = np.where(g[col], g.img_dice - g.lang_dice, 0.0)
            per_img = pd.Series(net, index=g.index).groupby([g.seed, g.sample_id]).mean().groupby(level=1).mean()
            rows.append(dict(zip(["dataset", "part", "method", "condition"], key), gate=gate, n_rows=len(g),
                             n_switched=len(sw), switch_rate=len(sw) / len(g),
                             n_beneficial=int((gain > NEUTRAL).sum()), n_harmful=int((gain < -NEUTRAL).sum()),
                             n_neutral=int((gain.abs() <= NEUTRAL).sum()),
                             pct_beneficial_of_switched=float((gain > NEUTRAL).mean()) if len(sw) else np.nan,
                             pct_harmful_of_switched=float((gain < -NEUTRAL).mean()) if len(sw) else np.nan,
                             net_dice_gain_image_weighted=float(per_img.mean()),
                             mean_gain_per_switch=float(gain.mean()) if len(sw) else np.nan))
    return pd.DataFrame(rows)


OUT = {"val": {"res": "phase3_validation_results.csv", "pg": "phase3_validation_paired_differences.csv",
               "fr": "phase3_fallback_rates.csv", "rq": "phase3_routing_quality.csv",
               "mo": "phase3_mosmed_exploratory.csv", "dec": "phase3_val_gate_decisions.csv.gz"},
       "test": {"res": "phase3_test_results.csv", "pg": "phase3_test_paired_differences.csv",
                "fr": "phase3_test_fallback_rates.csv", "rq": "phase3_test_routing_quality.csv",
                "mo": "phase3_test_mosmed_exploratory.csv", "dec": "phase3_test_gate_decisions.csv.gz"}}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    split = ap.parse_args().split
    if split == "test":
        from phase2 import lock
        lock.require_open(what="Phase 3 test evaluation")
    o = OUT[split]
    D, arch = load_rows(split)
    D = apply_gates(D)
    S = strategy_metrics(D)
    res = image_weighted(S)
    pg = pd.concat([paired_group(S, g) for g in ("D-Gate", "LR-Gate", "IMG", "NULL (C1)", "ORACLE")], ignore_index=True)
    res.to_csv(R(o["res"]), index=False, float_format="%.6g")
    pg.to_csv(R(o["pg"]), index=False, float_format="%.6g")
    fr = S[S.strategy.isin(["D-Gate", "LR-Gate"])].groupby(["dataset", "part", "method", "condition", "strategy", "seed"]
                                                          ).switch.mean().rename("fallback_rate").reset_index()
    fr.to_csv(R(o["fr"]), index=False, float_format="%.6g")
    rq = routing(D)
    rq.to_csv(R(o["rq"]), index=False, float_format="%.6g")
    mo = res[res.dataset == "MosMedData+"].merge(pg[pg.dataset == "MosMedData+"].pivot_table(
        index=["method", "condition"], columns="comparison", values="group_weighted_delta").reset_index(),
        on=["method", "condition"])
    mo.to_csv(R(o["mo"]), index=False, float_format="%.6g")
    if split == "test":                                                   # Task 3.6 deliverable (per case x strategy)
        S.to_csv(R("phase3_test_per_case.csv.gz"), index=False, float_format="%.6f")
    D[KEYS + ["part", "F1", "lr_score", "d_switch", "lr_switch"]].to_csv(
        R(o["dec"]), index=False, float_format="%.17g")      # full precision
    print("rows", len(D), "IMG", arch)


if __name__ == "__main__":
    main()
