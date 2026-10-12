"""Tests for src/analyze_phase0.py -- synthetic data and the VALIDATION split only (never the test split).

Part A, unit tests of the statistics:
  * holm(): a hand-worked example and statsmodels (if installed);
  * paired_t(): scipy.stats.ttest_rel, the explicit t-interval formula, and simulated CI coverage;
  * wilcoxon_p(): scipy.stats.wilcoxon;
  * bootstrap_mean_ci(): reproducibility (fixed seed), agreement with the normal-theory interval for large
    n, and simulated coverage;
  * select_cases(): known quantile positions on a toy table.
Part B, end-to-end on synthetic runs:
  * 3 methods x 2 datasets x 3 seeds of synthetic predictions on the validation split. Each prediction is
    the ground truth shifted by a method-specific number of pixels in a random direction; 2% of the cases
    get an empty prediction. They are scored by the real common evaluator (src/evaluate.py) and written in
    the same layout as real runs.
  * analyze_phase0.py is run with --split val into a scratch folder.
  * Every number in the two CSV files is then recomputed independently (plain pandas / scipy /
    statsmodels) and compared. The method with the smallest shift must come out best.

Usage:  python src/test_analyze_phase0.py --scratch <dir>      (exit status 1 if any check fails)
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
import analyze_phase0 as A  # noqa: E402
import evaluate as ev  # noqa: E402
from lgms import data  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


# ------------------------------------------------------------------------------------------------
# Part A: statistics
# ------------------------------------------------------------------------------------------------
def part_a():
    # Holm: p = [0.01, 0.04, 0.03] -> sorted 0.01*3 = 0.03, 0.03*2 = 0.06, max(0.06, 0.04*1) = 0.06
    adj = A.holm([0.01, 0.04, 0.03])
    check("holm hand example", np.allclose(adj, [0.03, 0.06, 0.06]), str(adj))
    check("holm caps at 1", np.allclose(A.holm([0.5, 0.6, 0.9]), [1.0, 1.0, 1.0]))
    try:
        from statsmodels.stats.multitest import multipletests
        rng = np.random.default_rng(1)
        ok = all(np.allclose(A.holm(p), multipletests(p, method="holm")[1])
                 for p in rng.uniform(0, 0.2, size=(200, 3)))
        check("holm == statsmodels (200 random triples)", ok)
    except ImportError:
        print("[skip] statsmodels not installed")

    rng = np.random.default_rng(2)
    a, b = rng.normal(0.80, 0.1, 500), rng.normal(0.78, 0.1, 500)
    r = A.paired_t(a, b)
    d = a - b
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    check("paired_t mean / CI formula", np.isclose(r["delta"], d.mean()) and np.isclose(r["ci_low"], d.mean() - half)
          and np.isclose(r["ci_high"], d.mean() + half))
    check("paired_t p == scipy ttest_rel", np.isclose(r["p"], stats.ttest_rel(a, b).pvalue))
    cover = np.mean([(lambda rr: rr["ci_low"] <= 0.02 <= rr["ci_high"])(
        A.paired_t(rng.normal(0.02, 0.1, 60), np.zeros(60))) for _ in range(2000)])
    check("paired_t CI coverage ~95% (2000 simulations)", 0.935 <= cover <= 0.965, f"{cover:.3f}")
    check("wilcoxon == scipy", np.isclose(A.wilcoxon_p(a, b)[1], stats.wilcoxon(a, b).pvalue))

    x = rng.beta(8, 2, 2000)
    c1, c2 = A.bootstrap_mean_ci(x), A.bootstrap_mean_ci(x)
    check("bootstrap reproducible with the fixed seed", c1 == c2, str(c1))
    se = x.std(ddof=1) / np.sqrt(len(x))
    check("bootstrap ~ normal CI for n = 2000", abs(c1[0] - (x.mean() - 1.96 * se)) < 0.15 * se * 1.96
          and abs(c1[1] - (x.mean() + 1.96 * se)) < 0.15 * se * 1.96, f"{c1} vs ±{1.96 * se:.4f}")
    true_mean = 8 / 10
    cover = np.mean([(lambda ci: ci[0] <= true_mean <= ci[1])(
        A.bootstrap_mean_ci(rng.beta(8, 2, 273), n_boot=2000, seed=k)) for k in range(400)])
    check("bootstrap CI coverage ~95% (400 simulations, n = 273)", 0.92 <= cover <= 0.975, f"{cover:.3f}")

    # select_cases on a toy table: 11 cases with scores 0.00, 0.10, ..., 1.00 -> positions 1, 3, 5, 7, 9
    ids = [f"c{i:02d}" for i in range(11)]
    toy = {}
    for m, _ in A.METHODS:
        for s in A.SEEDS:
            toy[(m, "toy", s)] = pd.DataFrame({"dice": np.linspace(0, 1, 11)[::-1]}, index=ids[::-1])
    picks = A.select_cases(toy, "toy")
    check("select_cases quantile positions", [p["sample_id"] for p in picks] == ["c01", "c03", "c05", "c07", "c09"],
          str([p["sample_id"] for p in picks]))


# ------------------------------------------------------------------------------------------------
# Part B: synthetic end-to-end run on the validation split
# ------------------------------------------------------------------------------------------------
SHIFT = {"unet": 3, "transunet": 5, "nnunet": 1}       # pixels; nnU-Net must come out best, TransUNet worst


def make_runs(root):
    """Synthetic predictions (shifted ground truth), scored by the real evaluator, in the real run layout."""
    for ds in A.DATASETS:
        ids = data.split_ids(ds, "val")
        gts = {i: data.read_mask(ds, "val", i) for i in ids}
        for mi, (m, _) in enumerate(A.METHODS):
            for s in A.SEEDS:
                rng = np.random.default_rng([mi, s, len(ds)])
                preds = {}
                for i in ids:
                    if rng.random() < 0.02:
                        preds[i] = np.zeros((224, 224), np.float32)            # empty prediction
                        continue
                    ang = rng.uniform(0, 2 * np.pi)
                    k = SHIFT[m] + rng.integers(0, 2)
                    preds[i] = np.roll(gts[i], (int(round(k * np.sin(ang))), int(round(k * np.cos(ang)))), (0, 1)).astype(np.float32)
                out = os.path.join(A.run_dir(root, m, ds, s), "eval_val")
                os.makedirs(os.path.join(out, "masks"), exist_ok=True)
                ev.evaluate(ds, "val", preds, workers=16).to_csv(os.path.join(out, "per_case.csv"), index=False)
                if s == A.FIGURE_SEED:
                    for i, p in preds.items():
                        cv2.imwrite(os.path.join(out, "masks", f"{i}.png"), (p > 0.5).astype(np.uint8) * 255)
                print(f"  synthetic run written: {m} {ds} seed{s}")


def independent_check(root, out):
    """Recompute every CSV value with separate, plain code and compare."""
    main = pd.read_csv(os.path.join(out, "results", "phase0_main_results.csv"))
    pw = pd.read_csv(os.path.join(out, "results", "phase0_pairwise_comparisons.csv"))
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        multipletests = None
    worst = 0.0
    for ds in A.DATASETS:
        case = {}
        for m, name in A.METHODS:
            dfs = [pd.read_csv(os.path.join(A.run_dir(root, m, ds, s), "eval_val", "per_case.csv"))
                   .set_index("sample_id").sort_index() for s in A.SEEDS]
            row = main[(main.dataset == ds) & (main.method == name)].iloc[0]
            for met, col, sc in [("dice", "dice", 100), ("miou", "miou", 100), ("hd95", "hd95", 1), ("assd", "assd", 1)]:
                v = np.array([d[met].mean() for d in dfs]) * sc
                unit = "pct" if sc == 100 else "px"
                worst = max(worst, abs(row[f"{col}_mean_{unit}"] - v.mean()), abs(row[f"{col}_sd_{unit}"] - np.std(v, ddof=1)))
            case[m] = sum(d["dice"] for d in dfs) / 3 * 100
            lo, hi = row.dice_ci95_low_pct, row.dice_ci95_high_pct
            check(f"{ds} {name}: CI contains the mean and is narrow", lo < row.dice_mean_pct < hi and hi - lo < 6,
                  f"{row.dice_mean_pct:.2f} [{lo:.2f}, {hi:.2f}]")
            check(f"{ds} {name}: point estimate = mean of seed-averaged per-case Dice",
                  np.isclose(case[m].mean(), row.dice_mean_pct, atol=1e-3))
        sub = pw[pw.dataset == ds].reset_index(drop=True)
        praw = []
        for k, (a, b) in enumerate(A.PAIRS):
            d = (case[a] - case[b]).values
            r = sub.iloc[k]
            p = stats.ttest_rel(case[a].values, case[b].values).pvalue
            praw.append(p)
            check(f"{ds} {A.NAME[a]} vs {A.NAME[b]}: ΔDice, CI, p, Wilcoxon",
                  np.isclose(r.delta_dice_pct, d.mean(), atol=1e-4)
                  and np.isclose(r.ci95_low_pct, d.mean() - stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)), atol=1e-4)
                  and np.isclose(r.p_raw, p, rtol=1e-4)
                  and np.isclose(r.wilcoxon_p_raw, stats.wilcoxon(case[a].values, case[b].values).pvalue, rtol=1e-4))
        if multipletests is not None:
            check(f"{ds}: Holm == statsmodels", np.allclose(sub.p_holm, multipletests(praw, method="holm")[1], rtol=1e-4))
        order = main[main.dataset == ds].sort_values("dice_mean_pct").method.tolist()
        check(f"{ds}: synthetic effect recovered (TransUNet < U-Net < nnU-Net)", order == ["TransUNet", "U-Net", "nnU-Net"], str(order))
    check("main table matches independent recomputation (max abs diff < 1e-3)", worst < 1e-3, f"{worst:.2e}")
    for f in ["phase0_dice_ci_QaTa-COV19.png", "phase0_dice_ci_MosMedDataplus.png", "phase0_seed_stability.png",
              "phase0_cases_QaTa-COV19.png", "phase0_cases_MosMedDataplus.png"]:
        check(f"figure written: {f}", os.path.getsize(os.path.join(out, "figures", f)) > 10_000)
    rep = open(os.path.join(out, "PHASE0_RESULTS.md")).read()
    check("report has the methods text, tables and placeholders",
          all(s in rep for s in ["Methods (fixed analysis plan)", "| Dataset | Method | Dice Mean ± SD", "Holm p",
                                 "Wilcoxon", "seed-1001", "[placeholder"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True, help="folder for the synthetic runs and outputs (not in the repo)")
    ap.add_argument("--skip-make", action="store_true", help="reuse synthetic runs already in --scratch")
    args = ap.parse_args()
    print("== Part A: statistics")
    part_a()
    print("== Part B: synthetic end-to-end run on the validation split")
    root, out = os.path.join(args.scratch, "runs"), os.path.join(args.scratch, "out")
    if not args.skip_make:
        make_runs(root)
    subprocess.run([sys.executable, os.path.join(REPO, "src", "analyze_phase0.py"), "--runs-root", root, "--split", "val",
                    "--results-dir", os.path.join(out, "results"), "--figures-dir", os.path.join(out, "figures"),
                    "--report", os.path.join(out, "PHASE0_RESULTS.md"), "--quiet"], check=True)
    independent_check(root, out)
    print("ALL CHECKS PASSED" if not FAILED else f"FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
