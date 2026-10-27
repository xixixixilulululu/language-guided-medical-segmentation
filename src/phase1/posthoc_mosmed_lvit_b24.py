"""POST-HOC supplementary analysis (decided after the Phase 1 main results had been seen; does NOT replace them).

MosMedData+ only: the six-method table and the 15 pairwise comparisons, computed exactly as in the fixed Phase 1
analysis, but with the three LViT-T batch-24 sensitivity runs (runs/lvit_b24/MosMedData+/seed<k>) in place of
the main LViT-T runs. The other five methods use their main-run results. Method:
reports/PHASE1_SENSITIVITY_LVIT.md, section 5.

src/analyze_phase1.py is fixed and is not modified: this script imports it and calls its own functions
(load_per_case, main_table, pairwise_table; statistics from src/analyze_phase0.py through it), with the same
constants (seeds, 10,000 bootstrap resamples with rng 0, paired t-test, Holm across the 15 pairs, alpha 0.05,
Wilcoxon with its own Holm as a supplement).

Check: with --no-substitute the script must reproduce the MosMedData+ rows of the main analysis
(results/final_six_method_results.csv, results/final_six_method_pairwise.csv) exactly; it compares them itself.

Outputs (MosMedData+ rows only):
  results/phase1_posthoc_mosmed_lvit_b24_results.csv
  results/phase1_posthoc_mosmed_lvit_b24_pairwise.csv

Usage:  python src/phase1/posthoc_mosmed_lvit_b24.py [--no-substitute]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
import analyze_phase1 as A  # noqa: E402  (fixed analysis, imported unchanged)

DATASET = "MosMedData+"
SUBSTITUTE_DIR = "lvit_b24"          # runs/lvit_b24/MosMedData+/seed<k>/eval_test/per_case.csv


def substitute(per_case, runs_root):
    """Replace the LViT-T MosMedData+ entries by the batch-24 runs, with the same id checks as load_per_case."""
    for s in A.SEEDS:
        ref = per_case[("lvit", DATASET, s)]
        f = os.path.join(A.run_dir(runs_root, SUBSTITUTE_DIR, DATASET, s), "eval_test", "per_case.csv")
        df = pd.read_csv(f, dtype={"sample_id": str}).set_index("sample_id")
        if sorted(df.index) != sorted(ref.index) or df.index.duplicated().any():
            raise ValueError(f"{f}: sample ids do not match the test split of {DATASET}")
        if df[A.METRICS].isna().any().any():
            raise ValueError(f"{f}: missing metric values")
        per_case[("lvit", DATASET, s)] = df.loc[ref.index, A.METRICS]
    return per_case


def compare_with_main(table, pairs, results_dir):
    """--no-substitute: the MosMed rows must equal the main analysis outputs (as written, 4 / 6 significant digits)."""
    for new, name in [(table, "final_six_method_results.csv"), (pairs, "final_six_method_pairwise.csv")]:
        old = pd.read_csv(os.path.join(results_dir, name))
        old = old[old["dataset"] == DATASET].reset_index(drop=True)
        new = new.reset_index(drop=True)[old.columns]
        num = old.select_dtypes("number").columns
        same = np.allclose(old[num].values, new[num].values.astype(float), rtol=1e-3, atol=1e-4) \
            and old.drop(columns=num).astype(str).equals(new.drop(columns=num).astype(str))
        print(f"{name}: MosMedData+ rows {'IDENTICAL to the main analysis' if same else 'DIFFER from the main analysis'}")
        if not same:
            sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default=os.path.join(REPO, "runs"))
    ap.add_argument("--results-dir", default=os.path.join(REPO, "results"))
    ap.add_argument("--no-substitute", action="store_true",
                    help="check only: keep the main LViT-T runs and compare with the main analysis outputs")
    args = ap.parse_args()

    per_case = A.load_per_case(args.runs_root, "test")           # all main runs, ids checked
    if not args.no_substitute:
        per_case = substitute(per_case, args.runs_root)
    table = A.main_table(per_case)
    pairs = A.pairwise_table(per_case)
    table = table[table["dataset"] == DATASET]                    # QaTa-COV19 is not affected
    pairs = pairs[pairs["dataset"] == DATASET]                    # Holm is per dataset, so unaffected by QaTa
    if args.no_substitute:
        compare_with_main(table, pairs, args.results_dir)
        return
    table = table.assign(lvit_t_runs="batch 24 (sensitivity, runs/lvit_b24)")
    table.to_csv(os.path.join(args.results_dir, "phase1_posthoc_mosmed_lvit_b24_results.csv"), index=False,
                 float_format="%.4f")
    pairs.to_csv(os.path.join(args.results_dir, "phase1_posthoc_mosmed_lvit_b24_pairwise.csv"), index=False,
                 float_format="%.6g")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(table[["method", "dice_mean_pct", "dice_sd_pct", "dice_ci95_low_pct", "dice_ci95_high_pct",
                     "miou_mean_pct", "hd95_mean_px", "assd_mean_px"]].to_string(index=False))
        print(pairs[["method_a", "method_b", "delta_dice_pct", "dice_ci95_low_pct", "dice_ci95_high_pct", "p_raw",
                     "p_holm", "significant_holm_alpha_0.05", "wilcoxon_p_holm"]].to_string(index=False))
    print("written: results/phase1_posthoc_mosmed_lvit_b24_{results,pairwise}.csv")


if __name__ == "__main__":
    main()
