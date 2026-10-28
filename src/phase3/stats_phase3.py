"""Phase 3, Task 3.7: paired statistical comparison of the fixed gates (QaTa primary / secondary families).

Rows come from src/phase3/evaluate_gates.py (load_rows -> apply_gates -> strategy_metrics): the same images,
donors and seeds for every strategy of a condition. Nothing is fitted or tuned here.

Aggregation of one comparison A - B (PHASE3_PLAN Task 3.7), Dice:
  1. donor differences averaged within image and seed (C4-C6: up to 3 donors; other conditions: 1 row)
  2. the 3 seed differences averaged within image
  3. image differences averaged within group (QaTa: Phase 2 patient units, covid_* images their own unit;
     MosMed: CT volumes)
  4. equally weighted group differences: mean, 95% t-CI and two-sided one-sample t-test p over groups
Image set: all images eligible for the condition (the Task 3.5 rows), per condition.

Families (QaTa only; two-sided alpha 0.05):
  primary     LR-Gate - LANG  under C4, C5, C6 x LViT-T, RecLMIS, DD-CMD   9 comparisons, Holm across the 9
  secondary   LR-Gate - D-Gate under C4, C5, C6 x the same methods           9 comparisons, Holm across the 9
Descriptive (mean and CI only, no p): D-Gate - LANG (all conditions); LR-Gate - LANG and LR-Gate - D-Gate for
  C0, C1, C2, C3a-c. C0 and C2 rows of gate - LANG carry the development margin -0.005 and the flag
  ci95_low_above_margin (a CI that does not reach the margin; a nonsignificant difference does not prove
  preservation). MosMedData+: every comparison exploratory (few volumes, known volume overlap across splits).
Validation (--split val, default): QaTa GATE_FIT and GATE_CAL are analysed separately; every row is labelled
  "descriptive (validation)": the gates were fitted / calibrated on these images, so no p-value here is a test.
Test (--split test; Phase 2 lock open, both protocols frozen): QaTa part "test".

Outputs: results/phase3_pairwise_results_val.csv  |  results/phase3_pairwise_results.csv (test)
Usage:   python src/phase3/stats_phase3.py [--split test]
Tests:   python src/phase3/test_stats_phase3.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
R = lambda f: os.path.join(REPO, "results", f)  # noqa: E731
QATA = "QaTa-COV19"
MISMATCH = ["C4", "C5", "C6"]
MARGIN_CONDS = ["C0", "C2"]
MARGIN = -0.005
COMPARISONS = [("LR-Gate", "LANG"), ("LR-Gate", "D-Gate"), ("D-Gate", "LANG")]
METHOD_ORDER = ["LViT-T", "RecLMIS", "DD-CMD"]
NAME = {"lvit": "LViT-T", "reclmis": "RecLMIS", "ddcmd": "DD-CMD"}
OUT = {"val": "phase3_pairwise_results_val.csv", "test": "phase3_pairwise_results.csv"}


def group_differences(S, a, b, metric="dice"):
    """{(dataset, part, method, condition): Series of group differences} for A - B (steps 1-3)."""
    k = ["dataset", "part", "method", "condition"]
    idx = k + ["group", "seed", "sample_id", "variant_id"]
    x = S[S.strategy == a].set_index(idx)[metric]
    y = S[S.strategy == b].set_index(idx)[metric]
    if len(x) != len(y) or not x.index.is_unique or y.reindex(x.index).isna().any():
        raise ValueError(f"{a} and {b} are not paired on the same rows")
    d = x - y.reindex(x.index)
    d = d.groupby(k + ["group", "seed", "sample_id"]).mean()          # 1. donors within image and seed
    d = d.groupby(k + ["group", "sample_id"]).mean()                  # 2. seeds within image
    n_img = d.groupby(k).size()
    d = d.groupby(k + ["group"]).mean()                               # 3. images within group
    return {key: (g.droplevel(k), int(n_img[key])) for key, g in d.groupby(k)}


def t_over_groups(u, level=0.95):
    """Mean, t-CI and two-sided one-sample t-test p over the group differences u (step 4)."""
    from scipy import stats
    u = np.asarray(u, float)
    n = len(u)
    r = {"n_groups": n, "delta": float(u.mean()), "ci95_low": np.nan, "ci95_high": np.nan, "t": np.nan, "p_raw": np.nan}
    if n >= 2:
        se = u.std(ddof=1) / np.sqrt(n)
        if se > 0:
            half = stats.t.ppf(1 - (1 - level) / 2, n - 1) * se
            res = stats.ttest_1samp(u, 0.0)
            r.update(ci95_low=float(u.mean() - half), ci95_high=float(u.mean() + half),
                     t=float(res.statistic), p_raw=float(res.pvalue))
        else:                                                             # all groups identical: no spread
            r.update(ci95_low=float(u.mean()), ci95_high=float(u.mean()))
    return r


def holm(p):
    """Holm step-down adjusted p-values (same order as the input); the Phase 0 implementation."""
    import analyze_phase0
    return analyze_phase0.holm(p)


def family(dataset, condition, a, b):
    if dataset != QATA:
        return "exploratory (MosMed)"
    if condition in MISMATCH and (a, b) == ("LR-Gate", "LANG"):
        return "primary"
    if condition in MISMATCH and (a, b) == ("LR-Gate", "D-Gate"):
        return "secondary"
    return "descriptive"


def pairwise(S, label):
    """All comparisons of the long strategy table S; Holm within each part for the two families."""
    rows = []
    for a, b in COMPARISONS:
        for (ds, part, m, c), (u, n_img) in group_differences(S, a, b).items():
            fam = family(ds, c, a, b)
            r = {"dataset": ds, "part": part, "method": m, "condition": c, "comparison": f"{a} - {b}", "family": fam,
                 "n_images": n_img, **t_over_groups(u.values)}
            if fam not in ("primary", "secondary"):
                r["t"] = r["p_raw"] = np.nan                              # descriptive: no test reported
            if ds == QATA and c in MARGIN_CONDS and b == "LANG":
                r["margin"] = MARGIN
                r["ci95_low_above_margin"] = bool(r["ci95_low"] >= MARGIN) if np.isfinite(r["ci95_low"]) else np.nan
            rows.append(r)
    P = pd.DataFrame(rows)
    P["p_holm"] = np.nan
    for (part, fam), g in P[P.family.isin(["primary", "secondary"])].groupby(["part", "family"]):
        if len(g) != 9:
            raise ValueError(f"{part} {fam}: {len(g)} comparisons, expected 9")
        P.loc[g.index, "p_holm"] = holm(g.p_raw.to_numpy())
    P["significant_holm_0.05"] = np.where(P.p_holm.notna(), P.p_holm < 0.05, np.nan)
    P.insert(0, "status", label)
    for c in ("margin", "ci95_low_above_margin"):
        if c not in P:
            P[c] = np.nan
    fam_order = {"primary": 0, "secondary": 1, "descriptive": 2, "exploratory (MosMed)": 3}
    P = P.assign(_f=P.family.map(fam_order), _m=P.method.map({m: i for i, m in enumerate(METHOD_ORDER)}))
    return P.sort_values(["dataset", "part", "_f", "comparison", "condition", "_m"]).drop(columns=["_f", "_m"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    split = ap.parse_args().split
    if split == "test":
        from phase2 import lock
        lock.require_open(what="Phase 3 test statistics")
    from phase3 import evaluate_gates as E
    D, _ = E.load_rows(split)
    S = E.strategy_metrics(E.apply_gates(D))
    S = S[S.strategy.isin(["LANG", "D-Gate", "LR-Gate"])].copy()
    S["method"] = S.method.map(NAME)
    label = ("descriptive (validation: gates fitted / calibrated on these images; p-values are not tests)"
             if split == "val" else "test (frozen protocol)")
    P = pairwise(S, label)
    P.to_csv(R(OUT[split]), index=False, float_format="%.6g")
    show = P[P.family.isin(["primary", "secondary"]) | P.margin.notna()]
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(show[["part", "family", "comparison", "method", "condition", "n_groups", "delta", "ci95_low",
                    "ci95_high", "p_raw", "p_holm", "ci95_low_above_margin"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
