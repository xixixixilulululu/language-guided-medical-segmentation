"""Phase 2 analysis (Tasks 2.4, 2.5) and the Phase 3 go / no-go rule (PHASE2_PROTOCOL.md section 5.1).

Input: the per-case files of src/phase2/infer.py, runs/phase2/<split>/<method>/<dataset>/seed<k>/per_case_*.csv
(Dice in [0, 1]). Units (patient / CT volume, D1 / D2) come from the sample ids (src/phase2/texts.groups); no
text is read except, for the D6 sensitivity check, the parse flags of the split's attributes.

Aggregation (PHASE2_PLAN Task 2.3): per image, method, seed and condition, the METRICS are averaged over the
donors (variants d1-d3) first; then, for paired analysis, over the three seeds within the image. Probability
maps are never averaged. Every comparison uses the common eligible image set of its two conditions.

Outputs (results/; one split per run, the split is a column):
  phase2_<split>_per_case.csv.gz        all per-case rows + unit (no text)
  phase2_main_results_<split>.csv       Task 2.4: mean +- sample SD over the 3 per-seed image means (donor-averaged),
                                        per-seed means, eligible images and units, mIoU / HD95 / ASSD / empty rate
  phase2_pairwise_results_<split>.csv   Task 2.5: the 11 comparisons per method x dataset; image-weighted delta;
                                        unit_weighted_delta with the one-sample t-test over unit differences
                                        (t-CI, raw p, Holm across the 11, alpha 0.05); exploratory flag (MosMed
                                        always, QaTa < 20 units, D3); mIoU / HD95 / ASSD unit-weighted deltas with
                                        a paired unit bootstrap CI (10,000, default_rng(0))
  phase2_d6_sensitivity_<split>.csv     MosMed without the scope-conflict images (D6): image- and unit-weighted
                                        deltas of the 11 comparisons, descriptive, no test
  phase2_phase3_rule_<split>.csv        section 5.1: per method x dataset x comparison x seed, delta_img and
                                        delta_unit, and the classification of the comparison
  phase2_phase3_per_donor_<split>.csv   the same per donor variant (descriptive)
The test split is refused (exit code 5) unless the Phase 2 lock is open.

Usage:  python src/phase2/analyze_phase2.py --split val
Tests:  python src/phase2/test_analyze_phase2.py   (synthetic data)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from phase2 import lock  # noqa: E402

METHODS = ["lvit", "reclmis", "ddcmd"]
NAME = {"lvit": "LViT-T", "reclmis": "RecLMIS", "ddcmd": "DD-CMD"}
DATASETS = ["QaTa-COV19", "MosMedData+"]
SEEDS = [1001, 1002, 1003]
CONDITIONS = ["C0", "C1", "C2", "C3a", "C3b", "C3c", "C4", "C5", "C6"]
COMPARISONS = [("C0", "C1"), ("C0", "C2"), ("C0", "C3a"), ("C0", "C3b"), ("C0", "C3c"), ("C0", "C4"),
               ("C0", "C5"), ("C0", "C6"), ("C1", "C4"), ("C1", "C5"), ("C1", "C6")]
METRICS = ["dice", "miou", "hd95", "assd"]
ALPHA = 0.05
N_BOOT, BOOT_SEED = 10_000, 0
MIN_UNITS_CONFIRMATORY = 20                       # D3: QaTa exploratory below this; MosMed always exploratory
# Phase 3 rule (section 5.1): comparisons and thresholds on the [0, 1] Dice scale
PHASE3 = {("C0", "C5"): 0.02, ("C0", "C6"): 0.02, ("C1", "C5"): 0.01, ("C1", "C6"): 0.01}


def stats_module():
    """Phase 0 statistics (holm, paired_t, bootstrap_mean_ci), imported unchanged."""
    import analyze_phase0
    return analyze_phase0


# ---------------------------------------------------------------------------------------------
# Loading and aggregation
# ---------------------------------------------------------------------------------------------
def load(root, split):
    """All per-case rows of one split, with the unit of every sample."""
    from phase2 import texts as T
    files = glob.glob(os.path.join(root, split, "*", "*", "seed*", "per_case_*.csv"))
    if not files:
        raise FileNotFoundError(f"no per-case files under {os.path.join(root, split)}")
    df = pd.concat([pd.read_csv(f, dtype={"sample_id": str, "variant_id": str, "donor_sample_id": str},
                                keep_default_na=False) for f in files], ignore_index=True)
    units = {(d, s): T.groups(d, s)[2:4] for d, s in df[["dataset", "sample_id"]].drop_duplicates().itertuples(index=False)}
    df["unit"] = [units[(d, s)][0] for d, s in zip(df.dataset, df.sample_id)]
    df["unit_type"] = [units[(d, s)][1] for d, s in zip(df.dataset, df.sample_id)]
    df["empty"] = df["empty"].astype(str).eq("True")
    df["seed"] = df["seed"].astype(int)
    return df


def image_seed_table(df):
    """Donor-averaged metrics per (dataset, method, seed, condition, sample)."""
    g = df.groupby(["dataset", "method", "seed", "condition", "sample_id", "unit"], sort=False)
    out = g[METRICS].mean()
    out["empty"] = g["empty"].mean()
    out["n_variants"] = g.size()
    return out.reset_index()


def wide(img, dataset, method, metric="dice", seed=None):
    """sample_id x condition table: one seed, or the mean over the three seeds (seed=None)."""
    t = img[(img.dataset == dataset) & (img.method == method)]
    if seed is not None:
        t = t[t.seed == seed]
        return t.pivot_table(index=["sample_id", "unit"], columns="condition", values=metric)
    if not set(t.groupby(["condition", "sample_id"]).seed.nunique()) <= {len(SEEDS)}:
        raise ValueError(f"{dataset} {method}: not every image has all three seeds")
    return t.pivot_table(index=["sample_id", "unit"], columns="condition", values=metric, aggfunc="mean")


def paired(w, a, b):
    """(image differences a - b on the common eligible set, unit means of those differences)."""
    if a not in w or b not in w:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    d = (w[a] - w[b]).dropna()
    return d, d.groupby(level="unit").mean()


# ---------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------
def main_table(img, split):
    rows = []
    for ds in DATASETS:
        for m in METHODS:
            t = img[(img.dataset == ds) & (img.method == m)]
            if t.empty:
                continue
            for c in CONDITIONS:
                tc = t[t.condition == c]
                if tc.empty:
                    rows.append({"split": split, "dataset": ds, "method": NAME[m], "condition": c,
                                 "n_images": 0, "n_units": 0})
                    continue
                per_seed = tc.groupby("seed")[METRICS + ["empty"]].mean()
                r = {"split": split, "dataset": ds, "method": NAME[m], "condition": c,
                     "n_images": tc.sample_id.nunique(), "n_units": tc.unit.nunique()}
                for k in METRICS + ["empty"]:
                    r[f"{k}_mean"], r[f"{k}_sd"] = per_seed[k].mean(), per_seed[k].std(ddof=1)
                for s in SEEDS:
                    r[f"dice_seed{s}"] = per_seed["dice"].get(s, np.nan)
                rows.append(r)
    return pd.DataFrame(rows)


def exploratory(dataset, n_units):
    if dataset == "MosMedData+":
        return True, "MosMedData+: unit-level inference exploratory (11 validation / 9 test units)"
    if n_units < MIN_UNITS_CONFIRMATORY:
        return True, f"fewer than {MIN_UNITS_CONFIRMATORY} units"
    return False, ""


def pairwise_table(img, split, exclude=None):
    """The 11 comparisons per method x dataset. exclude: {dataset: set(sample ids)} left out (D6 check)."""
    S = stats_module()
    rows = []
    for ds in DATASETS:
        for m in METHODS:
            if img[(img.dataset == ds) & (img.method == m)].empty:
                continue
            ws = {k: wide(img, ds, m, k) for k in METRICS}
            if exclude and ds in exclude:
                ws = {k: w[~w.index.get_level_values("sample_id").isin(exclude[ds])] for k, w in ws.items()}
            block = []
            for a, b in COMPARISONS:
                d, u = paired(ws["dice"], a, b)
                r = {"split": split, "dataset": ds, "method": NAME[m], "comparison": f"{a}-{b}",
                     "n_images": len(d), "n_units": len(u), "unit_type": "patient / image" if ds == "QaTa-COV19"
                     else "CT volume (+ pooled bjorke)"}
                if len(d):
                    common = d.index
                    r[f"mean_{a}"] = float(ws["dice"].loc[common, a].mean())
                    r[f"mean_{b}"] = float(ws["dice"].loc[common, b].mean())
                    r["image_weighted_delta"] = float(d.mean())
                    r["unit_weighted_delta"] = float(u.mean())
                if len(u) >= 2 and u.std(ddof=1) > 0:
                    t = S.paired_t(u.values, np.zeros(len(u)))
                    r.update(ci95_low=t["ci_low"], ci95_high=t["ci_high"], t=t["t"], p_raw=t["p"])
                else:
                    r.update(ci95_low=np.nan, ci95_high=np.nan, t=np.nan, p_raw=np.nan)
                for k in ["miou", "hd95", "assd"]:
                    _, uk = paired(ws[k], a, b)
                    r[f"{k}_unit_weighted_delta"] = float(uk.mean()) if len(uk) else np.nan
                    if len(uk) >= 2:
                        r[f"{k}_boot_ci95_low"], r[f"{k}_boot_ci95_high"] = S.bootstrap_mean_ci(
                            uk.values, n_boot=N_BOOT, seed=BOOT_SEED)
                expl, why = exploratory(ds, len(u))
                r["exploratory"], r["caveat"] = expl, why if len(u) >= 2 else "inference undefined (fewer than 2 units)"
                block.append(r)
            # Holm across the 11 comparisons; an unavailable test enters as p = 1 (internal), shown as NA
            p = [x["p_raw"] if np.isfinite(x.get("p_raw", np.nan)) else 1.0 for x in block]
            for x, h in zip(block, S.holm(p)):
                x["p_holm"] = float(h) if np.isfinite(x.get("p_raw", np.nan)) else np.nan
                x["significant_holm_0.05"] = bool(np.isfinite(x["p_holm"]) and x["p_holm"] < ALPHA)
            rows += block
    return pd.DataFrame(rows)


def d6_sensitivity(img, split, attrs):
    """MosMed without the D6 scope-conflict images: image- and unit-weighted deltas, descriptive."""
    from phase2 import texts as T
    conflict = set(attrs[(attrs.dataset == "MosMedData+") & attrs.parse_note.fillna("").map(T.is_scope_conflict)].sample_id)
    full = pairwise_table(img[img.dataset == "MosMedData+"], split)
    sub = pairwise_table(img[img.dataset == "MosMedData+"], split, exclude={"MosMedData+": conflict})
    key = ["dataset", "method", "comparison"]
    keep = ["n_images", "n_units", "image_weighted_delta", "unit_weighted_delta"]
    out = full[key + keep].merge(sub[key + keep], on=key, suffixes=("_all", "_without_d6"))
    out.insert(0, "split", split)
    out["d6_images_excluded"] = len(conflict & set(img[img.dataset == "MosMedData+"].sample_id))
    return out


def phase3_rule(img, split):
    """PHASE2_PROTOCOL.md section 5.1, per seed; returns (per seed table, per donor table)."""
    rows, donors = [], []
    for ds in DATASETS:
        for m in METHODS:
            if img[(img.dataset == ds) & (img.method == m)].empty:
                continue
            for (a, b), thr in PHASE3.items():
                per_seed = []
                for s in SEEDS:
                    d, u = paired(wide(img, ds, m, "dice", seed=s), a, b)
                    di, du = float(d.mean()), float(u.mean())
                    per_seed.append((s, di, du, di >= thr, di >= thr and du > 0))
                n_img = sum(x[3] for x in per_seed)
                n_both = sum(x[4] for x in per_seed)
                cls = "repeated signal" if n_both >= 2 else ("unstable signal" if n_img >= 2 else "no signal")
                for s, di, du, hit_img, hit_both in per_seed:
                    rows.append({"split": split, "dataset": ds, "method": NAME[m], "comparison": f"{a}-{b}",
                                 "threshold": thr, "seed": s, "n_images": len(d), "delta_img": di, "delta_unit": du,
                                 "img_meets_threshold": hit_img, "img_and_unit_meet": hit_both,
                                 "classification": cls})
    t = pd.DataFrame(rows)
    return t


def phase3_per_donor(df, split):
    """Descriptive: delta_img / delta_unit per seed and donor variant (d1, d2, d3)."""
    rows = []
    for (ds, m, s), g in df.groupby(["dataset", "method", "seed"]):
        base = {c: g[g.condition == c].set_index("sample_id") for c in ("C0", "C1")}
        for (a, b), thr in PHASE3.items():
            for v, gv in g[g.condition == b].groupby("variant_id"):
                gv = gv.set_index("sample_id")
                d = (base[a].dice.reindex(gv.index) - gv.dice).dropna()
                u = d.groupby(gv.unit.reindex(d.index)).mean()
                rows.append({"split": split, "dataset": ds, "method": NAME[m], "comparison": f"{a}-{b}", "seed": s,
                             "variant_id": v, "n_images": len(d), "delta_img": float(d.mean()),
                             "delta_unit": float(u.mean())})
    return pd.DataFrame(rows)


def phase3_summary(rule):
    """One row per method x dataset x comparison, and the overall outcome."""
    s = rule.drop_duplicates(["dataset", "method", "comparison"])[["dataset", "method", "comparison", "classification"]]
    per_md = s.groupby(["dataset", "method"]).classification.apply(lambda c: "repeated signal" in set(c))
    outcome = "basis for Phase 3" if per_md.any() else "no-go"
    return s, per_md, outcome


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--runs-root", default=os.path.join(REPO, "runs", "phase2"))
    ap.add_argument("--results-dir", default=os.path.join(REPO, "results"))
    args = ap.parse_args()
    if args.split == "test":
        lock.require_open(what="Phase 2 test analysis")              # before anything is read
    from phase2 import texts as T
    df = load(args.runs_root, args.split)
    img = image_seed_table(df)
    R = lambda name: os.path.join(args.results_dir, name)  # noqa: E731
    cols = ["dataset", "method", "seed", "sample_id", "condition", "variant_id", "donor_sample_id", "unit",
            "unit_type", "dice", "miou", "hd95", "assd", "empty"]
    df.sort_values(["dataset", "method", "seed", "condition", "variant_id", "sample_id"])[cols].to_csv(
        R(f"phase2_{args.split}_per_case.csv.gz"), index=False, float_format="%.6f")
    main_table(img, args.split).to_csv(R(f"phase2_main_results_{args.split}.csv"), index=False, float_format="%.6f")
    pw = pairwise_table(img, args.split)
    pw.to_csv(R(f"phase2_pairwise_results_{args.split}.csv"), index=False, float_format="%.6g")
    attrs = pd.concat([T.attributes(ds, args.split) for ds in DATASETS])
    d6_sensitivity(img, args.split, attrs).to_csv(R(f"phase2_d6_sensitivity_{args.split}.csv"), index=False,
                                                 float_format="%.6g")
    rule = phase3_rule(img, args.split)
    rule.to_csv(R(f"phase2_phase3_rule_{args.split}.csv"), index=False, float_format="%.6g")
    phase3_per_donor(df, args.split).to_csv(R(f"phase2_phase3_per_donor_{args.split}.csv"), index=False,
                                            float_format="%.6g")
    s, per_md, outcome = phase3_summary(rule)
    print(s.to_string(index=False))
    print("method x dataset with a repeated signal:", [k for k, v in per_md.items() if v] or "none")
    print("Phase 3 rule outcome (validation):", outcome)
    with open(R(f"phase2_phase3_outcome_{args.split}.json"), "w") as f:
        json.dump({"split": args.split, "outcome": outcome,
                   "repeated_signal": [list(k) for k, v in per_md.items() if v],
                   "classification": s.to_dict("records")}, f, indent=2)


if __name__ == "__main__":
    main()
